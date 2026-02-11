import math
import random
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import copy
import gc 
import matplotlib.pyplot as plt
import numpy as np
import os

from utils.inc_net import MiNbaseNet
from torch.utils.data import WeightedRandomSampler
from utils.toolkit import tensor2numpy, count_parameters
from data_process.data_manger import DataManger
from utils.training_tool import get_optimizer, get_scheduler
from utils.toolkit import calculate_class_metrics, calculate_task_metrics

# [ADDED] Import Mixed Precision
from torch.amp import autocast, GradScaler

EPSILON = 1e-8

class MinNet(object):
    def __init__(self, args, loger):
        super().__init__()
        self.args = args
        self.logger = loger
        self._network = MiNbaseNet(args)
        self.device = args['device']
        self.num_workers = args["num_workers"]

        self.init_epochs = args["init_epochs"]
        self.init_lr = args["init_lr"]
        self.init_weight_decay = args["init_weight_decay"]
        self.init_batch_size = args["init_batch_size"]

        self.lr = args["lr"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.epochs = args["epochs"]

        self.init_class = args["init_class"]
        self.increment = args["increment"]

        self.buffer_size = args["buffer_size"]
        self.buffer_batch = args["buffer_batch"]
        self.gamma = args['gamma']
        self.fit_epoch = args["fit_epochs"]

        self.known_class = 0
        self.cur_task = -1
        self.total_acc = []
        self.class_acc = []
        self.task_acc = []
        
        # [ADDED] Scaler cho Mixed Precision
        self.scaler = GradScaler('cuda')

    # [ADDED] Hàm gộp Noise (TUNA Logic)
    def merge_noise_experts(self):
        print(f"\n>>> Merging Noise Experts (TUNA EMR) for Task {self.cur_task}...")
        if hasattr(self._network.backbone, 'noise_maker'):
            for m in self._network.backbone.noise_maker:
                m.merge_noise()
        self._clear_gpu()

    def _clear_gpu(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def after_train(self, data_manger):
        if self.cur_task == 0:
            self.known_class = self.init_class
        else:
            self.known_class += self.increment

        _, test_list, _ = data_manger.get_task_list(self.cur_task)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)
        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False,
                                 num_workers=self.num_workers)
        eval_res = self.eval_task(test_loader)
        self.total_acc.append(round(float(eval_res['all_class_accy']*100.), 2))
        self.logger.info('total acc: {}'.format(self.total_acc))
        self.logger.info('avg_acc: {:.2f}'.format(np.mean(self.total_acc)))
        self.logger.info('task_confusion_metrix:\n{}'.format(eval_res['task_confusion']))
        print('total acc: {}'.format(self.total_acc))
        print('avg_acc: {:.2f}'.format(np.mean(self.total_acc)))
        del test_set
        self.analyze_entropy_accuracy(test_loader)

    def save_check_point(self, path_name):
        torch.save(self._network.state_dict(), path_name)

    # [MODIFIED] Compute Acc dùng Hybrid Inference
    def compute_test_acc(self, test_loader):
        model = self._network.eval()
        correct, total = 0, 0
        with torch.no_grad(), autocast('cuda'):
            for i, (_, inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(self.device)
                outputs = model.forward_tuna_combined(inputs)
                logits = outputs["logits"]
                predicts = torch.max(logits, dim=1)[1]
                correct += (predicts.cpu() == targets).sum()
                total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    @staticmethod
    def cat2order(targets, datamanger):
        for i in range(len(targets)):
            targets[i] = datamanger.map_cat2order(targets[i])
        return targets

    def init_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, train_list_name = data_manger.get_task_list(0)
        self.logger.info("task_list: {}".format(train_list_name))
        self.logger.info("task_order: {}".format(train_list))
        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)
        train_loader = DataLoader(train_set, batch_size=self.init_batch_size, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.init_batch_size, shuffle=False, num_workers=self.num_workers)
        self.test_loader = test_loader
        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = True

        self._network.update_fc(self.init_class)
        self._network.update_noise()
        self._clear_gpu()
        prototype = self.get_task_prototype(self._network, train_loader)
        self._network.extend_task_prototype(prototype)
        
        # 1. PRE-FIT
        self.fit_fc(train_loader, test_loader)
        # 2. RUN
        self._network.set_noise_mode(0)
        self.run(train_loader) 
        # 3. MERGE
        self.merge_noise_experts()
        self._clear_gpu()
        prototype = self.get_task_prototype(self._network, train_loader)
        self._network.update_task_prototype(prototype)
        # 4. CHỐT HẠ
        train_set_clean = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_clean.labels = self.cat2order(train_set_clean.labels, data_manger)
        train_loader_clean = DataLoader(train_set_clean, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = False
        self.re_fit(train_loader_clean, test_loader)
        del train_set, test_set, train_set_clean
        self._clear_gpu()

    def increment_train(self, data_manger):
        self.cur_task += 1
        train_list, test_list, train_list_name = data_manger.get_task_list(self.cur_task)
        self.logger.info("task_list: {}".format(train_list_name))
        self.logger.info("task_order: {}".format(train_list))
        train_set = data_manger.get_task_data(source="train", class_list=train_list)
        train_set.labels = self.cat2order(train_set.labels, data_manger)
        test_set = data_manger.get_task_data(source="test", class_list=test_list)
        test_set.labels = self.cat2order(test_set.labels, data_manger)
        train_loader = DataLoader(train_set, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.buffer_batch, shuffle=False, num_workers=self.num_workers)
        self.test_loader = test_loader
        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = False
        
        self._network.update_fc(self.increment)
        self._network.update_noise()
        self._clear_gpu()
        prototype = self.get_task_prototype(self._network, train_loader)
        self._network.extend_task_prototype(prototype)

        # 1. PRE-FIT
        self.fit_fc(train_loader, test_loader)
        # 2. RUN
        train_loader_run = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self._network.set_noise_mode(self.cur_task)
        self.run(train_loader_run)
        # 3. MERGE
        self.merge_noise_experts()
        self._clear_gpu()
        prototype = self.get_task_prototype(self._network, train_loader)
        self._network.update_task_prototype(prototype)
        del train_set
        # 4. CHỐT HẠ
        train_set_clean = data_manger.get_task_data(source="train_no_aug", class_list=train_list)
        train_set_clean.labels = self.cat2order(train_set_clean.labels, data_manger)
        train_loader_clean = DataLoader(train_set_clean, batch_size=self.buffer_batch, shuffle=True, num_workers=self.num_workers)
        if self.args['pretrained']:
            for param in self._network.backbone.parameters(): param.requires_grad = False
        self.re_fit(train_loader_clean, test_loader)
        del train_set_clean, test_set
        self._clear_gpu()

    def fit_fc(self, train_loader, test_loader):
        self._network.eval()
        self._network.to(self.device)
        self._network.reset_R_spec()

        # --- BƯỚC 1: TRÍCH XUẤT FEATURE 1 LẦN DUY NHẤT ---
        print(">>> Fast Fitting: Extracting features once...")
        all_feats = []
        all_targets = []
        
        with torch.no_grad():
            for _, inputs, targets in tqdm(train_loader, desc="Caching features"):
                inputs = inputs.to(self.device)
                # Chạy qua Backbone + Buffer đúng 1 lần
                # Lưu ý: fit_fc chạy trước Run nên Noise mode thường là 0 hoặc -2
                feat = self._network.extract_feature(inputs) 
                all_feats.append(feat.cpu()) # Đẩy về CPU để tránh OOM GPU
                all_targets.append(targets.cpu())

        all_feats = torch.cat(all_feats, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # --- BƯỚC 2: CHẠY RLS TRÊN TẬP FEATURE ĐÃ LƯU ---
        num_samples = all_feats.shape[0]
        # Thường RLS chỉ cần 1 epoch là đủ hội tụ
        for epoch in range(self.fit_epoch):
            # Shuffle indices để học khách quan hơn
            indices = torch.randperm(num_samples)
            
            for start_idx in range(0, num_samples, self.buffer_batch):
                end_idx = min(start_idx + self.buffer_batch, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Đưa mini-batch feature trở lại GPU
                f_batch = all_feats[batch_indices].to(self.device)
                t_batch = all_targets[batch_indices].to(self.device)
                y_onehot = torch.nn.functional.one_hot(t_batch, num_classes=self._network.known_class)

                # FIT TRỰC TIẾP (Cần hàm fit_spec_direct trong inc_net.py)
                self._network.set_noise_mode(self.cur_task)
                self._network.fit_spec_direct(f_batch, y_onehot)

                self._network.set_noise_mode(-2)
                self._network.fit_uni_direct(f_batch, y_onehot)

        self._clear_gpu()
        print(f">>> Task {self.cur_task} --> Fast Fit FC Done!")

    def re_fit(self, train_loader, test_loader):
        self._network.eval()
        self._network.to(self.device)
        self._network.reset_R_spec()

        print(">>> Fast Refitting: Extracting clean features...")
        all_feats = []
        all_targets = []
        
        with torch.no_grad():
            for _, inputs, targets in tqdm(train_loader, desc="Caching clean features"):
                feat = self._network.extract_feature(inputs.to(self.device))
                all_feats.append(feat.cpu())
                all_targets.append(targets.cpu())

        all_feats = torch.cat(all_feats, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        num_samples = all_feats.shape[0]
        # Re-fit thường chỉ cần 1 lượt qua dữ liệu
        indices = torch.arange(num_samples)
        
        for start_idx in range(0, num_samples, self.buffer_batch):
            end_idx = min(start_idx + self.buffer_batch, num_samples)
            f_batch = all_feats[indices[start_idx:end_idx]].to(self.device)
            t_batch = all_targets[indices[start_idx:end_idx]].to(self.device)
            y_onehot = torch.nn.functional.one_hot(t_batch, num_classes=self._network.known_class)

            self._network.set_noise_mode(self.cur_task)
            self._network.fit_spec_direct(f_batch, y_onehot)

            self._network.set_noise_mode(-2)
            self._network.fit_uni_direct(f_batch, y_onehot)

        self._clear_gpu()

    def run(self, train_loader):
        if self.cur_task == 0:
            epochs = self.init_epochs
            lr = self.init_lr
            weight_decay = self.init_weight_decay
        else:
            epochs = self.epochs
            lr = self.lr
            weight_decay = self.weight_decay

        for param in self._network.parameters(): param.requires_grad = False
        for param in self._network.normal_fc.parameters(): param.requires_grad = True
        if self.cur_task == 0: self._network.init_unfreeze()
        else: self._network.unfreeze_noise()
            
        params = filter(lambda p: p.requires_grad, self._network.parameters())
        optimizer = get_optimizer(self.args['optimizer_type'], params, lr, weight_decay)
        scheduler = get_scheduler(self.args['scheduler_type'], optimizer, epochs)

        self._network.train()
        self._network.to(self.device)
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with autocast('cuda'):
                    if self.cur_task > 0:
                        with torch.no_grad():
                            outputs1 = self._network(inputs, new_forward=False)
                            logits1 = outputs1['logits']
                        outputs2 = self._network.forward_normal_fc(inputs, new_forward=False)
                        logits2 = outputs2['logits']
                        logits2 = logits2 + logits1
                        loss = F.cross_entropy(logits2, targets.long())
                        logits_final = logits2
                    else:
                        outputs = self._network.forward_normal_fc(inputs, new_forward=False)
                        logits = outputs["logits"]
                        loss = F.cross_entropy(logits, targets.long())
                        logits_final = logits

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                losses += loss.item()
                _, preds = torch.max(logits_final, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                del inputs, targets, loss, logits_final

            scheduler.step()
            train_acc = 100. * correct / total
            info = "Task {} --> SGD Run: Loss {:.3f}, Acc {:.2f}".format(self.cur_task, losses/len(train_loader), train_acc)
            self.logger.info(info)
            prog_bar.set_description(info)

    def eval_task(self, test_loader):
        model = self._network.eval()
        pred, label = [], []
        with torch.no_grad(), autocast('cuda'):
            for i, (_, inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(self.device)
                outputs = model.forward_tuna_combined(inputs)
                logits = outputs["logits"]
                predicts = torch.max(logits, dim=1)[1]
                pred.extend([int(predicts[i].cpu().numpy()) for i in range(predicts.shape[0])])
                label.extend(int(targets[i].cpu().numpy()) for i in range(targets.shape[0]))
        class_info = calculate_class_metrics(pred, label)
        task_info = calculate_task_metrics(pred, label, self.init_class, self.increment)
        return {
            "all_class_accy": class_info['all_accy'],
            "class_accy": class_info['class_accy'],
            "class_confusion": class_info['class_confusion_matrices'],
            "task_accy": task_info['all_accy'],
            "task_confusion": task_info['task_confusion_matrices'],
            "all_task_accy": task_info['task_accy'],
        }

    def get_task_prototype(self, model, train_loader):
        model = model.eval()
        model.to(self.device)
        features = []
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                with autocast('cuda'):
                    feature = model.extract_feature(inputs)
                features.append(feature.detach().cpu())
        
        prototype = torch.mean(torch.cat(features, dim=0), dim=0).to(self.device)
        self._clear_gpu()
        return prototype


    def analyze_entropy_accuracy(self, test_loader):
        self._network.eval()
        all_entropies = []
        all_correct_flags = []
    
        print(">>> Fast Analysis: Caching features for proof...")
        # 1. Trích xuất feature thô (Base features) 1 lần duy nhất
        raw_features = []
        targets_list = []
        with torch.no_grad():
            for _, inputs, targets in test_loader:
                # Chỉ chạy qua backbone 1 lần
                feat = self._network.backbone(inputs.to(self.device))
                raw_features.append(feat.cpu())
                targets_list.append(targets.cpu())
        
        raw_features = torch.cat(raw_features, dim=0)
        targets_all = torch.cat(targets_list, dim=0)
    
        # 2. Chạy qua các Expert (Chỉ là các lớp Linear nhỏ, cực nhanh)
        num_tasks = self.cur_task + 1
        for t in range(num_tasks):
            self._network.set_noise_mode(t)
            with torch.no_grad():
                # Chạy qua PiNoise (Expert) + Buffer + FC
                # PiNoise nhận raw_features đã cache
                f_t = self._network.buffer(self._network.backbone.noise_maker[0].forward_from_feat(raw_features.to(self.device)))
                l_t = self._network.fc_spec(f_t)['logits']
                
                if t in self._network.task_class_indices:
                    task_cols = self._network.task_class_indices[t]
                    l_t_masked = l_t[:, task_cols]
                    prob = torch.softmax(l_t_masked, dim=1)
                    entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=1)
                    
                    predicts = torch.max(l_t_masked, dim=1)[1]
                    global_preds = torch.tensor([task_cols[p] for p in predicts.cpu().numpy()]).to(self.device)
                    correct = (global_preds == targets_all.to(self.device)).float()
    
                    all_entropies.extend(entropy.cpu().numpy())
                    all_correct_flags.extend(correct.cpu().numpy())
    
        self._plot_tuna_proof(all_entropies, all_correct_flags)
    
    def _plot_tuna_proof(self, entropies, correct_flags, num_bins=10):
        entropies = np.array(entropies)
        correct_flags = np.array(correct_flags)
    
        # Chia dữ liệu thành các nhóm (bins) theo mức độ Entropy 
        bin_edges = np.linspace(entropies.min(), entropies.max(), num_bins + 1)
        accuracies = []
        avg_entropies = []
    
        for i in range(num_bins):
            mask = (entropies >= bin_edges[i]) & (entropies < bin_edges[i+1])
            if mask.any():
                accuracies.append(correct_flags[mask].mean() * 100)
                avg_entropies.append((bin_edges[i] + bin_edges[i+1]) / 2)
    
        # Vẽ biểu đồ tương tự Figure 2 trong TUNA [cite: 246]
        plt.figure(figsize=(8, 6))
        plt.plot(accuracies, avg_entropies, marker='o', color='black', linewidth=2)
        plt.xlabel('Accuracy (%)', fontsize=12)
        plt.ylabel('Entropy', fontsize=12)
        plt.title(f'Entropy vs Accuracy Correlation (Task {self.cur_task})', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Đảo ngược trục Y: Entropy thấp (tự tin) nằm ở trên cùng tương ứng Acc cao 
        plt.gca().invert_yaxis() 
        
        plt.savefig(f'routing_proof_task_{self.cur_task}.png')
        plt.close()
        print(f">>> Chứng minh đã được lưu tại: routing_proof_task_{self.cur_task}.png")
