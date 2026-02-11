import copy
import logging
import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from backbones.pretrained_backbone import get_pretrained_backbone
from backbones.linears import SimpleLinear, SplitCosineLinear, CosineLinear
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
from torch.nn import functional as F
import scipy.stats as stats
import timm
import random
# [ADDED] Import autocast để kiểm soát precision thủ công
from torch.cuda.amp import autocast 

class BaseIncNet(nn.Module):
    def __init__(self, args: dict):
        super(BaseIncNet, self).__init__()
        self.args = args
        self.backbone = get_pretrained_backbone(args)
        self.feature_dim = self.backbone.out_dim
        self.fc = None

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    @staticmethod
    def generate_fc(in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        hyper_features = self.backbone(x)
        logits = self.fc(hyper_features)['logits']
        return {
            'features': hyper_features,
            'logits': logits
        }


class RandomBuffer(torch.nn.Linear):
    def __init__(self, in_features: int, buffer_size: int, device):
        super(torch.nn.Linear, self).__init__()
        self.bias = None
        self.in_features = in_features
        self.out_features = buffer_size
        
        # [MODIFIED] Dùng float32 thay vì double (tiết kiệm 50% VRAM)
        factory_kwargs = {"device": device, "dtype": torch.float32}
        
        self.W = torch.empty((self.in_features, self.out_features), **factory_kwargs)
        self.register_buffer("weight", self.W)

        self.reset_parameters()

    # @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # [ADDED] Đảm bảo input cùng kiểu với weight (tránh lỗi FP16 vs FP32)
        X = X.to(self.weight.dtype)
        return F.relu(X @ self.W)


class MiNbaseNet(nn.Module):
    def __init__(self, args: dict):
        super(MiNbaseNet, self).__init__()
        self.args = args
        self.backbone = get_pretrained_backbone(args)
        self.device = args['device']
        # initiate params
        self.gamma = args['gamma']
        self.buffer_size = args['buffer_size']
        self.feature_dim = self.backbone.out_dim  # dim of backbone
        self.task_prototypes = []

        self.buffer = RandomBuffer(in_features=self.feature_dim, buffer_size=self.buffer_size, device=self.device)

        # [MODIFIED] Chuyển toàn bộ sang float32
        factory_kwargs = {"device": self.device, "dtype": torch.float32}

        weight = torch.zeros((self.buffer_size, 0), **factory_kwargs)
        self.register_buffer("weight", weight)

        self.R: torch.Tensor
        R = torch.eye(self.weight.shape[0], **factory_kwargs) / self.gamma
        self.register_buffer("R", R)

        self.Pinoise_list = nn.ModuleList()

        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0

        self.fc2 = nn.ModuleList()

    # [ADDED] Hỗ trợ Gradient Checkpointing (Cứu cánh cho OOM)
    def set_grad_checkpointing(self, enable=True):
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(enable)
        elif hasattr(self.backbone, 'model') and hasattr(self.backbone.model, 'set_grad_checkpointing'):
             self.backbone.model.set_grad_checkpointing(enable)
        elif hasattr(self.backbone, 'grad_checkpointing'):
            self.backbone.grad_checkpointing = enable

    def forward_fc(self, features):
        features = features.to(self.weight)
        return features @ self.weight

    @property
    def in_features(self) -> int:
        return self.weight.shape[0]

    @property
    def out_features(self) -> int:
        return self.weight.shape[1]

    def update_fc(self, nb_classes):
        self.cur_task += 1
        self.known_class += nb_classes
        if self.cur_task > 0:
            fc = SimpleLinear(self.buffer_size, self.known_class, bias=False)
        else:
            fc = SimpleLinear(self.buffer_size, nb_classes, bias=True)
        if self.normal_fc is None:
            self.normal_fc = fc
        else:
            nn.init.constant_(fc.weight, 0.)

            del self.normal_fc
            self.normal_fc = fc

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        # [MODIFIED] Tắt Autocast ở đây. Ma trận nghịch đảo cần chạy FP32.
        with autocast(enabled=False):
            X = self.backbone(X).float() # Ép về float32
            X = self.buffer(X) # Buffer đã sửa thành float32 ở trên

            X, Y = X.to(self.weight.device), Y.to(self.weight.device).float()

            num_targets = Y.shape[1]
            if num_targets > self.out_features:
                increment_size = num_targets - self.out_features
                tail = torch.zeros((self.weight.shape[0], increment_size)).to(self.weight)
                self.weight = torch.cat((self.weight, tail), dim=1)
            elif num_targets < self.out_features:
                increment_size = self.out_features - num_targets
                tail = torch.zeros((Y.shape[0], increment_size)).to(Y)
                Y = torch.cat((Y, tail), dim=1)

            # [MODIFIED] Thêm Jitter để tránh Singular Matrix (vì dùng float32 kém chính xác hơn double)
            I = torch.eye(X.shape[0]).to(X)
            term = I + X @ self.R @ X.T
            jitter = 1e-6 * torch.eye(term.shape[0], device=term.device)
            
            K = torch.inverse(term + jitter)
            
            self.R -= self.R @ X.T @ K @ X @ self.R
            self.weight += self.R @ X.T @ (Y - X @ self.weight)

    def forward(self, x, new_forward: bool = False):
        
        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        if self.training:
            print("!!! NOISE IS RUNNING !!!")
        # [ADDED] Cast về dtype của weight (thường là FP16 nếu đang autocast)
        hyper_features = hyper_features.to(self.weight.dtype)
        logits = self.forward_fc(self.buffer(hyper_features))
        return {
            'logits': logits
        }
    
    def update_task_prototype(self, prototype):
        # [MODIFIED] Lưu về CPU
        if isinstance(prototype, torch.Tensor):
            self.task_prototypes[-1] = prototype.detach().cpu()
        else:
            self.task_prototypes[-1] = prototype # Giả sử đã xử lý ở min.py

    def extend_task_prototype(self, prototype):
        # [MODIFIED] Lưu về CPU
        if isinstance(prototype, torch.Tensor):
            self.task_prototypes.append(prototype.detach().cpu())
        else:
            self.task_prototypes.append(prototype)

    def extract_feature(self, x):
        hyper_features = self.backbone(x)
        return hyper_features

    def forward_normal_fc(self, x, new_forward: bool = False):

        if new_forward:
            hyper_features = self.backbone(x, new_forward=True)
        else:
            hyper_features = self.backbone(x)
        
        # [MODIFIED] Logic ép kiểu an toàn
        hyper_features = self.buffer(hyper_features)
        hyper_features = hyper_features.to(self.normal_fc.weight.dtype) 
        
        logits = self.normal_fc(hyper_features)['logits']
        return {
            "logits": logits
        }

    def update_noise(self):
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].update_noise()
            self.backbone.noise_maker[j].init_weight_noise(self.task_prototypes)

    def unfreeze_noise(self):
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].unfreeze_noise()

    def init_unfreeze(self):
        for j in range(self.backbone.layer_num):
            for param in self.backbone.noise_maker[j].parameters():
                param.requires_grad = True
            for p in self.backbone.blocks[j].norm1.parameters():
                p.requires_grad = True
            for p in self.backbone.blocks[j].norm2.parameters():
                p.requires_grad = True
        for p in self.backbone.norm.parameters():
            p.requires_grad = True
    # --- TRONG CLASS MiNbaseNet ---

    # [ADDED] Hàm điều khiển mode cho PiNoise
    def set_noise_mode(self, mode):
        if hasattr(self.backbone, 'noise_maker'):
            for m in self.backbone.noise_maker:
                m.active_task_idx = mode

    # [ADDED] Logic tìm kết quả tự tin nhất (Max Logit)
    # Sửa hàm này trong MiNbaseNet
    def forward_tuna_combined(self, x, debug_targets=None):
        # debug_targets: Nhãn (Label) thật để kiểm tra xem router chọn có đúng task không
        was_training = self.training
        self.eval()
        
        batch_size = x.shape[0]
        num_tasks = len(self.backbone.noise_maker[0].mu)
        
        # 1. Universal
        self.set_noise_mode(-2)
        with torch.no_grad():
            l_uni = self.forward_fc(self.buffer(self.backbone(x)))

        # 2. Specific Selection
        best_l_spec = torch.zeros_like(l_uni)
        min_entropy = torch.full((batch_size,), float('inf'), device=x.device)
        selected_task_indices = torch.full((batch_size,), -1, device=x.device, dtype=torch.long)

        # Lưu lại entropy của tất cả các task để so sánh
        debug_entropy_matrix = [] 

        with torch.no_grad():
            for t in range(num_tasks):
                self.set_noise_mode(t)
                l_t = self.forward_fc(self.buffer(self.backbone(x)))
                
                prob = torch.softmax(l_t, dim=1)
                entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=1)
                
                if debug_targets is not None:
                    debug_entropy_matrix.append(entropy.unsqueeze(1))

                mask = entropy < min_entropy
                min_entropy[mask] = entropy[mask]
                best_l_spec[mask] = l_t[mask]
                selected_task_indices[mask] = t # Lưu lại task nào được chọn

        final_logits = l_uni + best_l_spec
        self.set_noise_mode(-2)
        if was_training: self.train()

        # --- PHẦN LOG DEBUG ---
        debug_info = {}
        if debug_targets is not None:
            # Giả sử mỗi task có increment class (ví dụ 10 class/task)
            # Cần tính Task ID thật của ảnh dựa trên Label
            # Ví dụ: Label 15 (increment=10) -> Task ID = 1
            increment = self.args['increment'] if 'increment' in self.args else 10 # Check args
            init_cls = self.args['init_class'] if 'init_class' in self.args else 10
            
            # Logic tính Task ID từ Label (đơn giản hóa)
            gt_task_ids = torch.zeros_like(debug_targets)
            gt_task_ids[debug_targets < init_cls] = 0
            for k in range(1, num_tasks):
                lower = init_cls + (k-1)*increment
                upper = init_cls + k*increment
                mask = (debug_targets >= lower) & (debug_targets < upper)
                gt_task_ids[mask] = k
            
            # 1. Routing Accuracy: Tỷ lệ chọn đúng chuyên gia
            correct_routing = (selected_task_indices == gt_task_ids).float().mean().item()
            
            # 2. Entropy Gap: Entropy của Task đúng vs Entropy thấp nhất tìm được
            # (Nếu Task đúng mà Entropy cao -> Model ngu ở task đó)
            entropies = torch.cat(debug_entropy_matrix, dim=1) # [Batch, Num_Task]
            # Lấy entropy của task đúng
            gt_entropies = entropies.gather(1, gt_task_ids.unsqueeze(1)).squeeze()
            avg_gt_entropy = gt_entropies.mean().item()
            avg_min_entropy = min_entropy.mean().item()

            debug_info = {
                "routing_acc": correct_routing * 100,
                "avg_entropy_selected": avg_min_entropy,
                "avg_entropy_groundtruth": avg_gt_entropy,
                "task_distribution": [
                    (selected_task_indices == t).sum().item() for t in range(num_tasks)
                ]
            }

        return {'logits': final_logits, 'debug_info': debug_info}
