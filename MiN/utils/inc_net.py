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
        self.fc_uni = None
        self.fc_spec = None
        self.task_class_indices = {}
    def update_fc(self, nb_classes):
        self.cur_task += 1
        start_class = self.known_class
        self.known_class += nb_classes
        
        # Lưu lại phạm vi class của task mới (VD: Task 1 là [10, 20])
        self.task_class_indices[self.cur_task] = list(range(start_class, self.known_class))

        # Update cả 3 mạng: Uni, Spec và Normal
        self.fc_uni = self.generate_fc(self.buffer_size, self.known_class, self.fc_uni)
        self.fc_spec = self.generate_fc(self.buffer_size, self.known_class, self.fc_spec)
        self.update_normal_fc(self.known_class)

    def generate_fc(self, in_dim, out_dim, old_fc=None):
        new_fc = SimpleLinear(in_dim, out_dim, bias=True)
        if old_fc is not None:
            nb_output = old_fc.out_features
            weight = copy.deepcopy(old_fc.weight.data)
            bias = copy.deepcopy(old_fc.bias.data)
            new_fc.weight.data[:nb_output] = weight
            new_fc.bias.data[:nb_output] = bias
        return new_fc

    def update_normal_fc(self, nb_classes):
        if self.cur_task == 0:
            self.normal_fc = SimpleLinear(self.buffer_size, nb_classes, bias=True)
        else:
            # Normal FC mở rộng và giữ weight cũ (cho SGD tiếp tục)
            new_fc = SimpleLinear(self.buffer_size, nb_classes, bias=True)
            if self.normal_fc is not None:
                nb_old = self.normal_fc.out_features
                new_fc.weight.data[:nb_old] = self.normal_fc.weight.data
                new_fc.bias.data[:nb_old] = self.normal_fc.bias.data
                # Init phần mới bằng 0
                nn.init.constant_(new_fc.weight.data[nb_old:], 0.)
                nn.init.constant_(new_fc.bias.data[nb_old:], 0.)
            self.normal_fc = new_fc

    # --- HÀM FIT 1: UNIVERSAL (RECURSIVE RLS) ---
    @torch.no_grad()
    def fit_uni(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        with autocast(enabled=False): # RLS cần FP32
            X = self.buffer(self.backbone(X)).float()
            Y = Y.float()

            # Mở rộng Y nếu cần
            if Y.shape[1] < self.fc_uni.out_features:
                tail = torch.zeros((Y.shape[0], self.fc_uni.out_features - Y.shape[1])).to(Y)
                Y = torch.cat((Y, tail), dim=1)

            # RLS Update dùng self.R_uni
            W = self.fc_uni.weight.data.T 
            
            term = torch.eye(X.shape[0]).to(X) + X @ self.R_uni @ X.T
            jitter = 1e-6 * torch.eye(term.shape[0], device=term.device)
            K = torch.inverse(term + jitter)
            
            self.R_uni -= self.R_uni @ X.T @ K @ X @ self.R_uni
            W += self.R_uni @ X.T @ (Y - X @ W)
            
            self.fc_uni.weight.data = W.T

    # --- HÀM FIT 2: SPECIFIC (RESET RLS) ---
    @torch.no_grad()
    def fit_spec(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        with autocast(enabled=False):
            X = self.buffer(self.backbone(X)).float()
            Y = Y.float() # Y ở đây là one-hot của toàn bộ known_classes
            
            # Chỉ lấy các cột target thuộc task hiện tại
            task_idxs = self.task_class_indices[self.cur_task]
            Y_task = Y[:, task_idxs]

            # Independent RLS (Giải trực tiếp bằng Ridge Regression)
            # W* = (X^T X + lambda I)^-1 X^T Y
            lambda_reg = 1e-2
            I = torch.eye(X.shape[1], device=X.device)
            
            Cov = X.T @ X + lambda_reg * I
            W_task = torch.linalg.pinv(Cov) @ (X.T @ Y_task)
            
            # Update CHỈ các cột của task hiện tại trong fc_spec
            # Các cột khác giữ nguyên (thực ra là rác hoặc 0, ta không quan tâm)
            self.fc_spec.weight.data[task_idxs, :] = W_task.T

    # --- HÀM TRAIN NOISE (Dùng Normal FC + SGD) ---
    # Min.py gọi hàm này trong run()
    def forward_normal_fc(self, x, new_forward: bool = False):
        hyper_features = self.backbone(x)
        # Đi qua Buffer và Normal FC
        out = self.normal_fc(self.buffer(hyper_features))
        return out

    # Hàm forward mặc định (ít dùng, chủ yếu để tương thích)
    def forward(self, x):
        return self.forward_tuna_combined(x)

    # --- CORE: HYBRID INFERENCE (DUAL RLS + TUNA) ---
    def forward_tuna_combined(self, x):
        was_training = self.training
        self.eval()
        
        batch_size = x.shape[0]
        num_tasks = len(self.backbone.noise_maker[0].mu)
        
        # 1. Nhánh Universal (Base Prediction)
        self.set_noise_mode(-2)
        with torch.no_grad():
            feat_uni = self.buffer(self.backbone(x))
            logits_uni = self.fc_uni(feat_uni)['logits'] 

        # 2. Nhánh Specific (Routing - Selection)
        min_entropy = torch.full((batch_size,), float('inf'), device=x.device)
        best_task_ids = torch.zeros((batch_size,), dtype=torch.long, device=x.device)
        
        # Lưu logits để dùng sau
        saved_task_logits = [] 

        with torch.no_grad():
            for t in range(num_tasks):
                self.set_noise_mode(t)
                feat_t = self.buffer(self.backbone(x))
                # Dùng fc_spec
                l_t = self.fc_spec(feat_t)['logits'] 
                saved_task_logits.append(l_t)
                
                # [FIX BUG 1] MASKED ENTROPY
                # Chỉ tính entropy dựa trên các class của task t
                if t in self.task_class_indices:
                    task_cols = self.task_class_indices[t]
                    l_t_masked = l_t[:, task_cols] 
                    
                    prob = torch.softmax(l_t_masked, dim=1)
                    entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=1)
                    
                    mask = entropy < min_entropy
                    min_entropy[mask] = entropy[mask]
                    best_task_ids[mask] = t

        # 3. Sparse Logit Injection
        # Final = Base + Expert(Correct Task Only)
        final_logits = logits_uni.clone()
        
        for t in range(num_tasks):
            if t in self.task_class_indices:
                class_idxs = self.task_class_indices[t]
                mask_t = (best_task_ids == t)
                
                if mask_t.sum() > 0:
                    expert_l = saved_task_logits[t][mask_t]
                    cols = torch.tensor(class_idxs, device=self.device)
                    # Chỉ cộng phần tinh hoa của task t
                    final_logits[mask_t][:, cols] += expert_l[:, cols]

        self.set_noise_mode(-2)
        if was_training: self.train()
        
        return {'logits': final_logits}

    # --- CÁC HÀM PHỤ TRỢ ---
    def set_noise_mode(self, mode):
        if hasattr(self.backbone, 'noise_maker'):
            for m in self.backbone.noise_maker:
                m.active_task_idx = mode

    def extract_feature(self, x):
        return self.buffer(self.backbone(x))

    def update_task_prototype(self, prototype):
        if isinstance(prototype, torch.Tensor): self.task_prototypes[-1] = prototype.detach().cpu()
        else: self.task_prototypes[-1] = prototype

    def extend_task_prototype(self, prototype):
        if isinstance(prototype, torch.Tensor): self.task_prototypes.append(prototype.detach().cpu())
        else: self.task_prototypes.append(prototype)

    def update_noise(self):
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].update_noise()
            self.backbone.noise_maker[j].init_weight_noise(self.task_prototypes)

    def unfreeze_noise(self):
        for j in range(self.backbone.layer_num):
            self.backbone.noise_maker[j].unfreeze_noise()

    def init_unfreeze(self):
        for j in range(self.backbone.layer_num):
            for param in self.backbone.noise_maker[j].parameters(): param.requires_grad = True
            for p in self.backbone.blocks[j].norm1.parameters(): p.requires_grad = True
            for p in self.backbone.blocks[j].norm2.parameters(): p.requires_grad = True
        for p in self.backbone.norm.parameters(): p.requires_grad = True


