import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import math
import cv2
from dataloaders.data_utils import get_points_from_mask, get_bboxes_from_mask
import pickle
import re
import random
import logging
from collections import OrderedDict
from utils import DiceLoss

model_logger = logging.getLogger('model')
model_logger.setLevel(logging.DEBUG)
model_logger.propagate = True

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        inputs = torch.clamp(inputs, -10.0, 10.0)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        pt = torch.clamp(pt, 1e-8, 1.0 - 1e-8)
        focal_loss = self.alpha * ((1 - pt) + 1e-4) ** self.gamma * bce_loss
        focal_loss = torch.nan_to_num(focal_loss, nan=0.0, posinf=1.0, neginf=0.0)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BoundaryLoss(nn.Module):
    """边界损失，用于增强边界分割精度"""
    def __init__(self, sigma=2.0):
        super().__init__()
        self.sigma = sigma
    
    def forward(self, pred, gt):
        pred_prob = torch.sigmoid(pred)
        gt_binary = (gt > 0).float()
        
        kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], device=pred.device).view(1, 1, 3, 3).float()
        
        gt_boundary = torch.abs(F.conv2d(gt_binary, kernel, padding=1, groups=1))
        gt_boundary = (gt_boundary > 0).float()
        
        b, c, h, w = gt_binary.shape
        total_loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
        
        for i in range(b):
            for j in range(c):
                gt_boundary_i = gt_boundary[i, j]
                if gt_boundary_i.sum() > 0:
                    boundary_coords = torch.nonzero(gt_boundary_i)
                    if boundary_coords.numel() > 0:
                        max_boundary_points = 1000  
                        if boundary_coords.shape[0] > max_boundary_points:
                            idx = torch.randperm(boundary_coords.shape[0], device=pred.device)[:max_boundary_points]
                            boundary_coords = boundary_coords[idx]
                        
                        x = torch.arange(h, device=pred.device).unsqueeze(1).repeat(1, w)
                        y = torch.arange(w, device=pred.device).unsqueeze(0).repeat(h, 1)
                        
                        min_dist = torch.full((h, w), float('inf'), device=pred.device)
                        for coord in boundary_coords:
                            dist = torch.sqrt((x - coord[0])**2 + (y - coord[1])**2)
                            min_dist = torch.min(min_dist, dist)
                        
                        lambda_b = 1.0
                        weight = 1.0 + lambda_b * torch.exp(-min_dist**2 / (2 * self.sigma**2))
                        ce_term = -(gt_binary[i, j] * torch.log(pred_prob[i, j] + 1e-8) +
                                    (1 - gt_binary[i, j]) * torch.log(1 - pred_prob[i, j] + 1e-8))
                        weighted_ce = weight * ce_term
                        w_sum = weight.sum().clamp(min=1.0)
                        boundary_loss = weighted_ce.sum() / w_sum
                        total_loss = total_loss + boundary_loss.to(pred.dtype)
                        del x, y, min_dist, weight, ce_term, weighted_ce, boundary_loss
                        torch.cuda.empty_cache()

        if (b * c) > 0:
            return total_loss / (b * c)
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

class TemporalAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss(alpha=0.75, gamma=2.0)
        self.consistency_loss = nn.MSELoss(reduction='none')  
        self.boundary_loss = BoundaryLoss()
        self.lambda_q = 1.0
        self.consistency_warmup_start = 50
        self.consistency_warmup_end = 120
        self.max_consistency_weight = 0.3
    def forward(self, pred_masks, gt_masks, iou_pred=None, previous_pred=None, interaction_round=0, global_epoch=0,
                new_click_coords=None, image_size=(256, 256), click_exclude_radius=5):
        if isinstance(pred_masks, dict):
            if iou_pred is None and 'iou_pred' in pred_masks:
                iou_pred = pred_masks['iou_pred']
            pred_masks = pred_masks['masks']
        if pred_masks.shape[2:] != gt_masks.shape[2:]:
            pred_masks = F.interpolate(pred_masks, size=gt_masks.shape[2:], mode='bilinear', align_corners=False)
        if gt_masks.dtype == torch.long:
            gt_masks = gt_masks.to(dtype=pred_masks.dtype)
        else:
            gt_masks = gt_masks.to(dtype=pred_masks.dtype)
        dice_loss = self.dice_loss(pred_masks, gt_masks)
        focal_loss = self.focal_loss(pred_masks, gt_masks)
        bce_loss = self.bce_loss(pred_masks, gt_masks)
        zero_scalar = torch.zeros((), device=pred_masks.device, dtype=pred_masks.dtype)
        boundary_loss = self.boundary_loss(pred_masks, gt_masks) if global_epoch >= 5 else zero_scalar
        w_dice = torch.tensor(1.0, device=pred_masks.device, dtype=pred_masks.dtype)
        w_focal = torch.tensor(0.5, device=pred_masks.device, dtype=pred_masks.dtype)
        w_bce = torch.tensor(0.1, device=pred_masks.device, dtype=pred_masks.dtype)
        w_boundary = torch.tensor(0.3 if global_epoch >= 5 else 0.0, device=pred_masks.device, dtype=pred_masks.dtype)
        main_loss = w_dice * dice_loss + w_focal * focal_loss + w_bce * bce_loss + w_boundary * boundary_loss
        pred_prob = torch.sigmoid(pred_masks)
        pred_binary = (pred_prob > 0.5).to(dtype=pred_masks.dtype)
        gt_binary = (gt_masks > 0).to(dtype=pred_masks.dtype)
        intersection = (pred_binary * gt_binary).sum(dim=(1, 2, 3))
        union = (pred_binary + gt_binary - pred_binary * gt_binary).sum(dim=(1, 2, 3))
        raw_iou = (intersection + 1e-7) / (union + 1e-7)
        quality_loss = zero_scalar.clone()
        if iou_pred is not None:
            q = iou_pred
            if q.dim() > 1:
                q = q[:, 0]
            q = q.squeeze()
            if q.min() < 0 or q.max() > 1:
                q = torch.sigmoid(q)
            else:
                q = q.clamp(0, 1)
            quality_loss = self.lambda_q * F.mse_loss(q, raw_iou.detach()).to(dtype=pred_masks.dtype)
        consistency_loss = zero_scalar.clone()
        interaction_round = (
            interaction_round.item() if isinstance(interaction_round, torch.Tensor) else interaction_round
        )
        if previous_pred is not None and interaction_round > 0:
            if previous_pred.device != pred_masks.device:
                previous_pred = previous_pred.to(pred_masks.device)
            if previous_pred.shape[2:] != pred_masks.shape[2:]:
                previous_pred = F.interpolate(previous_pred, size=pred_masks.shape[2:], mode='bilinear', align_corners=False)
            prev_prob = torch.sigmoid(previous_pred).detach().to(dtype=pred_masks.dtype)
            cur_dice = raw_iou.mean()
            if cur_dice > 0.65:
                if global_epoch < self.consistency_warmup_start:
                    consistency_weight = 0.0
                elif global_epoch < self.consistency_warmup_end:
                    consistency_weight = self.max_consistency_weight * (global_epoch - self.consistency_warmup_start) / (self.consistency_warmup_end - self.consistency_warmup_start)
                else:
                    consistency_weight = self.max_consistency_weight
                b, c, h, w = pred_prob.shape
                mse_term = (pred_prob - prev_prob) ** 2
                valid_mask = torch.ones((b, 1, h, w), device=pred_prob.device, dtype=pred_masks.dtype)
                if new_click_coords is not None and click_exclude_radius > 0:
                    coords = new_click_coords
                    if isinstance(coords, torch.Tensor):
                        coords = coords.to(dtype=pred_masks.dtype)
                        img_h, img_w = image_size
                        if coords.shape[-1] == 2:
                            ys = torch.arange(h, device=pred_prob.device).to(dtype=pred_masks.dtype)
                            xs = torch.arange(w, device=pred_prob.device).to(dtype=pred_masks.dtype)
                            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
                            grid_y = grid_y.unsqueeze(0).unsqueeze(0)
                            grid_x = grid_x.unsqueeze(0).unsqueeze(0)
                            for bi in range(b):
                                for pi in range(coords.shape[1]):
                                    cx = coords[bi, pi, 0]
                                    cy = coords[bi, pi, 1]
                                    if cx.detach().item() == 0 and cy.detach().item() == 0:
                                        continue
                                    scale_x = w / img_w
                                    scale_y = h / img_h
                                    px = cx * scale_x
                                    py = cy * scale_y
                                    r_sq = float(click_exclude_radius ** 2)
                                    d2 = (grid_x - px) ** 2 + (grid_y - py) ** 2
                                    valid_mask[bi, 0] = valid_mask[bi, 0] * (d2[0, 0] > r_sq).to(dtype=pred_masks.dtype)
                valid_count = valid_mask.sum().clamp(min=1.0)
                consistency_loss = (mse_term * valid_mask).sum() / valid_count * consistency_weight
        return (main_loss + quality_loss + consistency_loss).to(dtype=pred_masks.dtype)

class QualityGateStrategy:
    """
    Quality-Guided Selection (QGS) Strategy - 按照论文描述实现
    公式 (4): 质量门控机制
    
    动态阈值策略：
    1. 预热期 (epoch < warmup_epochs): 全部存入，不进行质量筛选
    2. 上升期 (warmup_epochs ≤ epoch < warmup_epochs + ramp_epochs): 线性提升阈值
    3. 稳定期 (epoch ≥ warmup_epochs + ramp_epochs): 使用最大阈值
    
    TCI 计算按照论文公式：
    TCI_t = 1 - min(1, σ({ΔD_i}) / δ)
    其中 σ 是 Dice 变化的标准差，δ = 0.15
    """
    
    def __init__(self, base_dice_threshold=0.5, base_tci_threshold=0.6, 
                 max_dice_threshold=0.75, max_tci_threshold=0.8, 
                 warmup_epochs=50, ramp_epochs=70, delta=0.15):
        """
        初始化质量门控策略
        
        Args:
            base_dice_threshold: 基础 Dice 阈值（上升期起始），默认 0.5
            base_tci_threshold: 基础 TCI 阈值（上升期起始），默认 0.6
            max_dice_threshold: 最大 Dice 阈值（稳定期使用），默认 0.75
            max_tci_threshold: 最大 TCI 阈值（稳定期使用），默认 0.8
            warmup_epochs: 预热期 epoch 数（全部存入），默认 50
            ramp_epochs: 阈值上升期 epoch 数（50-120），默认 70
            delta: δ, TCI 计算中的归一化参数，默认 0.15
        """
        self.base_dice_threshold = base_dice_threshold
        self.base_tci_threshold = base_tci_threshold
        self.max_dice_threshold = max_dice_threshold
        self.max_tci_threshold = max_tci_threshold
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.delta = delta
    
    def _get_dynamic_thresholds(self, global_epoch):
        """
        根据当前 epoch 动态计算阈值
        
        策略：
        - epoch < warmup_epochs: 返回极低阈值（等效于全部通过）
        - warmup_epochs ≤ epoch < warmup_epochs + ramp_epochs: 线性插值提升阈值
        - epoch ≥ warmup_epochs + ramp_epochs: 使用最大阈值（严格）
        
        Args:
            global_epoch: 当前全局 epoch
            
        Returns:
            tuple: (current_dice_threshold, current_tci_threshold, is_warmup)
        """
        epoch = max(0, global_epoch)
        
        if epoch < self.warmup_epochs:
            return 0.0, 0.0, True
        elif epoch < self.warmup_epochs + self.ramp_epochs:
            progress = (epoch - self.warmup_epochs) / self.ramp_epochs
            dice_threshold = self.base_dice_threshold + progress * (self.max_dice_threshold - self.base_dice_threshold)
            tci_threshold = self.base_tci_threshold + progress * (self.max_tci_threshold - self.base_tci_threshold)
            return dice_threshold, tci_threshold, False
        else:
            return self.max_dice_threshold, self.max_tci_threshold, False
    
    def calculate_tci(self, dice_history):
        """
        计算时序一致性指数 (TCI)
        
        按照论文公式：
        TCI_t = 1 - min(1, σ({ΔD_i}) / δ)
        
        其中 ΔD_i = D_i - D_{i-1} 是相邻轮次的 Dice 变化
        
        Args:
            dice_history: 历史 Dice 值列表 [D_1, D_2, ..., D_t]
            
        Returns:
            float: TCI 值，范围 [0, 1]
        """
        if len(dice_history) < 2:
            return 1.0
        
        dice_changes = []
        for i in range(1, len(dice_history)):
            delta_d = dice_history[i] - dice_history[i - 1]
            dice_changes.append(delta_d)
        
        if len(dice_changes) < 2:
            return 1.0
        
        mean_change = sum(dice_changes) / len(dice_changes)
        variance = sum((dc - mean_change) ** 2 for dc in dice_changes) / len(dice_changes)
        std_dev = variance ** 0.5  
        
        normalized_std = std_dev / self.delta
        tci = 1.0 - min(1.0, normalized_std)
        
        return tci
    
    def check_quality(self, current_dice, dice_history, global_epoch=0):
        """
        检查当前分割是否满足质量门控要求（支持动态阈值）
        
        按照论文公式 (4):
        B_t = { {Z_t, h_t}, if Dice > τ_dice(epoch) and TCI > τ_tci(epoch)
              { B_{t-1}, otherwise
        
        预热期特殊处理：epoch < warmup_epochs 时全部通过
        
        Args:
            current_dice: 当前轮次的 Dice 系数 D_t
            dice_history: 历史 Dice 值列表 [D_1, D_2, ..., D_{t-1}]
            global_epoch: 当前全局 epoch，用于动态调整阈值
            
        Returns:
            tuple: (bool, float, float, float, float, bool)
                   (是否合格, 当前Dice, 当前TCI, 当前Dice阈值, 当前TCI阈值, 是否预热期)
        """
        dice_threshold, tci_threshold, is_warmup = self._get_dynamic_thresholds(global_epoch)
        
        if is_warmup:
            tci = self.calculate_tci(dice_history + [current_dice]) if len(dice_history) >= 1 else 1.0
            return True, current_dice, tci, dice_threshold, tci_threshold, True
        
        full_history = dice_history + [current_dice]
        tci = self.calculate_tci(full_history)
        
        dice_passed = current_dice > dice_threshold
        tci_passed = tci > tci_threshold
        
        return (dice_passed and tci_passed), current_dice, tci, dice_threshold, tci_threshold, False

class TemporalMemoryModule(nn.Module):
    MAX_HISTORY = 8  
    
    def __init__(self, max_history_length=8, feature_dim=256):
        super().__init__()
        self.max_history_length = max_history_length
        self.feature_dim = feature_dim
        self.history_buffer = OrderedDict()
        
        self.quality_gate = QualityGateStrategy(
            base_dice_threshold=0.68,
            base_tci_threshold=0.70,
            max_dice_threshold=0.75,
            max_tci_threshold=0.80,
            delta=0.15
        )
    
    def _safe_convert_id(self, patient_id):
        """安全地转换patient_id为字符串"""
        if isinstance(patient_id, torch.Tensor):
            if patient_id.numel() == 1:
                return str(patient_id.item())
            else:
                return str(id(patient_id))
        elif not isinstance(patient_id, str):
            return str(patient_id)
        return patient_id
    
    def _safe_convert_category(self, category):
        """安全地转换category为字符串"""
        if isinstance(category, torch.Tensor):
            if category.numel() == 1:
                return str(category.item())
            else:
                return str(id(category))
        elif not isinstance(category, str):
            return str(category)
        return category
    def store(self, patient_id, category, data, global_epoch=0): 
        """按类别存储病人历史记录 - 内存优化版"""
        patient_id_str = self._safe_convert_id(patient_id)
        category_str = self._safe_convert_category(category)
        
        print(f"[STOREDEBUG] store called: patient={patient_id_str}, cat={category_str}, epoch={global_epoch}, data_keys={list(data.keys())}")
        
        try:
            if patient_id_str not in self.history_buffer:
                self.history_buffer[patient_id_str] = {}
            if category_str not in self.history_buffer[patient_id_str]:
                self.history_buffer[patient_id_str][category_str] = []
            print(f"[STOREDEBUG] history_buffer initialized, len={len(self.history_buffer[patient_id_str][category_str])}")
        except Exception as e:
            print(f"[STOREDEBUG ERROR] history_buffer init failed: {e}")
            return
        
        try:
            def ultra_safe_scalar(value):
                """确保任何输入都转换为 Python 标量"""
                if isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        return value.item()
                    else:
                        return value.mean().item()
                elif isinstance(value, (list, tuple)):
                    if len(value) > 0:
                        return ultra_safe_scalar(value[0])
                    return 0.0
                elif isinstance(value, (int, float)):
                    return value
                else:
                    return 0.0
            
            quality = ultra_safe_scalar(
                data.get('quality_score', data.get('iou_predictions', data.get('dice', 0.0)))
            )
            dice = ultra_safe_scalar(data.get('dice', 0.0))  
            global_epoch = ultra_safe_scalar(global_epoch)
            label_sum = ultra_safe_scalar(data.get('label_sum', None))
            print(f"[STOREDEBUG] converted values: quality(gate)={quality:.4f}, dice(log)={dice:.4f}, epoch={global_epoch}, label_sum={label_sum}")

            if isinstance(label_sum, (int, float)) and label_sum is not None and label_sum <= 0:
                model_logger.debug(f"[QUALITYGATE] Skip empty organ slice: label_sum={label_sum}")
                print(f"[QUALITYGATE] Skip empty organ slice: label_sum={label_sum}")
                return

            history_list = self.history_buffer[patient_id_str].get(category_str, [])
            quality_history = [h['quality'] for h in history_list if 'quality' in h]
            
            is_qualified, current_quality, current_tci, quality_threshold, tci_threshold, is_warmup = (
                self.quality_gate.check_quality(quality, quality_history, global_epoch)
            )
            
            stage = "WARMUP" if is_warmup else ("RAMP" if global_epoch < self.quality_gate.warmup_epochs + self.quality_gate.ramp_epochs else "STABLE")
            print(f"[QUALITYGATE] patient={patient_id_str}, cat={category_str}, stage={stage}, quality={current_quality:.4f} (thresh={quality_threshold:.4f}), tci={current_tci:.4f} (thresh={tci_threshold:.4f}), qualified={is_qualified}")
            
            reduced_data = {
                'quality': quality,     
                'dice': dice,           
                'tci': current_tci,
                'epoch': global_epoch,
                'qualified': is_qualified
            }
            
            training_flag = False
            if hasattr(self, 'training'):
                if isinstance(self.training, bool):
                    training_flag = self.training
                elif isinstance(self.training, torch.Tensor):
                    training_flag = self.training.mean().item() if self.training.numel() > 1 else self.training.item()
            print(f"[STOREDEBUG] training_flag={training_flag}")
            
            if not is_qualified:
                print(f"[QUALITYGATE] Record rejected: quality(gate)={quality:.4f} (threshold={quality_threshold:.4f}), tci={current_tci:.4f} (threshold={tci_threshold:.4f})")
                return
            
            if 'features' in data and isinstance(data['features'], torch.Tensor):
                has_nan = torch.isnan(data['features']).any().item()
                has_inf = torch.isinf(data['features']).any().item()
                print(f"[STOREDEBUG] patient={patient_id_str}, cat={category_str}, epoch={global_epoch}, quality(gate)={quality:.4f}, dice(log)={dice:.4f}, has_nan={has_nan}, has_inf={has_inf}")
                if has_nan or has_inf:
                    model_logger.error(f"[FATAL] NaN/Inf detected, skip storing")
                    return
                    
                if data['features'].shape[-1] != 64 or data['features'].shape[-2] != 64:
                    data['features'] = F.interpolate(
                        data['features'].unsqueeze(0) if data['features'].dim() == 3 else data['features'],
                        size=(64, 64),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                
                spatial_pooler = nn.AdaptiveAvgPool2d((8, 8))
                reduced_data['features_spatial'] = spatial_pooler(data['features'].unsqueeze(0)).squeeze(0).cpu()
            
            for key in list(data.keys()):
                if key not in ['features_spatial'] and key in reduced_data:
                    reduced_data[key] = ultra_safe_scalar(reduced_data[key])
            
            print(f"[STOREDEBUG] training_flag={training_flag}, history_len_before={len(self.history_buffer.get(patient_id_str, {}).get(category_str, []))}")
        
            self.history_buffer[patient_id_str][category_str].append(reduced_data)
        
            if len(self.history_buffer[patient_id_str][category_str]) > self.MAX_HISTORY:
                self.history_buffer[patient_id_str][category_str] = self.history_buffer[patient_id_str][category_str][-self.MAX_HISTORY:]
            
            print(f"[STOREDEBUG] after store, history_len={len(self.history_buffer[patient_id_str][category_str])}")
        except Exception as e:
            print(f"[STOREDEBUG ERROR] store process failed: {e}")
            import traceback
            traceback.print_exc()
    def update_history(self, patient_id, data, global_epoch=0):
        """更新病人历史记录，每个病人最多保存8条历史"""
        category = data.get('category', 'default')
        self.store(patient_id, category, data, global_epoch)
    
    def clear_patient_history(self, patient_id):
        """清除指定病人的历史记录，避免不同病人之间的信息混乱"""
        patient_id_str = self._safe_convert_id(patient_id)
        if patient_id_str in self.history_buffer:
            del self.history_buffer[patient_id_str]
            print(f"[INFO] Cleared history for patient {patient_id_str}")
    
    def get_history(self, patient_id, category=None):
        """获取病人的历史记录"""
        patient_id_str = self._safe_convert_id(patient_id)
        
        if patient_id_str not in self.history_buffer:
            return []
        
        if category is None:
            all_history = []
            for cat_history in self.history_buffer[patient_id_str].values():
                all_history.extend(cat_history)
            return all_history
        else:
            category_str = self._safe_convert_category(category)
            return self.history_buffer[patient_id_str].get(category_str, [])
    def clear_all_history(self):
        """清空所有历史记录，用于Epoch结束时"""
        self.history_buffer.clear()
    
    def clear_history(self, patient_id, category=None):
        """清除指定病人或指定类别的历史记录"""
        patient_id_str = self._safe_convert_id(patient_id)
        
        if category is not None:
            category_str = self._safe_convert_category(category)
            if patient_id_str in self.history_buffer and category_str in self.history_buffer[patient_id_str]:
                del self.history_buffer[patient_id_str][category_str]
                if not isinstance(self.history_buffer[patient_id_str], dict) or len(self.history_buffer[patient_id_str]) == 0:
                    del self.history_buffer[patient_id_str]
        else:
            if patient_id_str in self.history_buffer:
                del self.history_buffer[patient_id_str]
    
    def clear_expired_history(self, max_age_epochs=50):
        """清理过期的历史记录（超过max_age_epochs的记录）"""
        current_epoch = getattr(self, 'current_epoch', 0)
        for patient_id in list(self.history_buffer.keys()):
            for category in list(self.history_buffer[patient_id].keys()):
                self.history_buffer[patient_id][category] = [
                    record for record in self.history_buffer[patient_id][category]
                    if current_epoch - record.get('epoch', 0) <= max_age_epochs
                ]
                if not self.history_buffer[patient_id][category]:
                    del self.history_buffer[patient_id][category]
            if not self.history_buffer[patient_id]:
                del self.history_buffer[patient_id]
    
    def on_epoch_end(self, current_epoch):
        """在训练循环中定期调用（如每个epoch结束）"""
        self.current_epoch = current_epoch
        self.clear_expired_history()
    
    def get_temporal_context(self, patient_id, category=None):
        """改进的时序获取逻辑：即使历史记录少于2条也返回有效序列"""
        patient_id_str = self._safe_convert_id(patient_id)
        if patient_id_str not in self.history_buffer:
            return None
        history_list = self.get_history(patient_id, category)
        if not isinstance(history_list, (list, tuple)) or len(history_list) < 1:
            return None
        features = []
        for data in history_list:
            if not isinstance(data, dict):
                continue
            feat = data.get('features_spatial', None) or data.get('features', None)
            if feat is None:
                continue
            if not isinstance(feat, torch.Tensor):
                continue
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            feat = feat.to(device, non_blocking=True)
            if len(feat.shape) == 4:
                feat = torch.mean(feat, dim=(2, 3))
            elif len(feat.shape) == 3:
                feat = F.interpolate(feat.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0)
                feat = torch.mean(feat, dim=(1, 2))
            if len(features) == 0:
                features.append(feat)
            else:
                if feat.shape == features[0].shape:
                    features.append(feat)
        if len(features) > 0:
            sequence_tensor = torch.stack(features, dim=1)
            return sequence_tensor
        return None
    def clear_history(self, patient_id=None, category=None):
        """清空历史记录，支持清空特定病人或特定类别的历史"""
        if patient_id is not None:
            patient_id_str = self._safe_convert_id(patient_id)
            if patient_id_str in self.history_buffer:
                if category is not None:
                    category_str = self._safe_convert_category(category)
                    if category_str in self.history_buffer[patient_id_str]:
                        del self.history_buffer[patient_id_str][category_str]
                    if not isinstance(self.history_buffer[patient_id_str], dict) or len(self.history_buffer[patient_id_str]) == 0:
                        del self.history_buffer[patient_id_str]
                else:
                    del self.history_buffer[patient_id_str]
        else:
            self.history_buffer.clear()

class LSTMTemporalModule(nn.Module):
    """LSTM时序记忆模块"""
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.feature_align = nn.ModuleDict()
        self.pre_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, groups=feature_dim),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
        )
        self.temporal_memory = nn.GRU(
            input_size=feature_dim,
            hidden_size=feature_dim,
            num_layers=2,
            batch_first=True
        )
        self.pattern_recognizer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),  
            nn.GELU(),  
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.attn = nn.Linear(feature_dim, 1)  
        
        self._init_weights()
    def _init_weights(self):
        """优化权重初始化，提高训练稳定性"""
        for name, param in self.temporal_memory.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        for name, param in self.pattern_recognizer.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def normalize_coordinates(self, coords, image_size):
        """归一化坐标到 [0, 1] 范围
        
        Args:
            coords: 坐标张量，形状为 [N, 2]
            image_size: 图像尺寸 (height, width)
        
        Returns:
            归一化后的坐标，形状为 [N, 2]
        """
        if coords is None:
            return None
        
        coords = coords.float()
        
        height, width = image_size
        coords[:, 0] /= width - 1  
        coords[:, 1] /= height - 1  
        
        coords = torch.clamp(coords, 0.0, 1.0)
        
        return coords
    def process_labels(self, labels, num_classes=None, use_embedding=False, embedding_dim=32):
        """处理类别标签，支持 One-hot 编码或 Embedding
        
        Args:
            labels: 类别标签张量，形状为 [N]
            num_classes: 类别数量，用于 One-hot 编码
            use_embedding: 是否使用 Embedding 而非 One-hot
            embedding_dim: Embedding 维度
        
        Returns:
            处理后的标签向量，形状为 [N, D]，其中 D 是编码维度
        """
        if labels is None:
            return None
        
        device = labels.device
        
        if use_embedding:
            if not hasattr(self, 'label_embedding'):
                max_class = 100
                self.label_embedding = nn.Embedding(max_class, embedding_dim).to(device)
            
            labels = labels.long()
            labels = torch.clamp(labels, 0, 99)
            label_embeddings = self.label_embedding(labels)
            return label_embeddings
        else:
            if num_classes is None:
                num_classes = labels.max().item() + 1
            
            labels = labels.long()
            one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes)
            return one_hot.float()
    def pad_sequences(self, sequences, max_length=None, padding_value=0.0):
        """填充序列到相同长度
        
        Args:
            sequences: 序列列表，每个元素是形状为 [T_i, D] 的张量
            max_length: 最大序列长度，默认使用最长序列的长度
            padding_value: 填充值
        
        Returns:
            填充后的序列张量，形状为 [B, max_length, D]
            序列长度列表，形状为 [B]
        """
        if not isinstance(sequences, (list, tuple)) or len(sequences) == 0:
            return torch.empty(0), []
        
        seq_lengths = [len(seq) for seq in sequences]
        
        if max_length is None:
            max_length = max(seq_lengths)
        
        feature_dim = sequences[0].shape[1]
        device = sequences[0].device
        
        padded_sequences = torch.full((len(sequences), max_length, feature_dim), 
                                     padding_value, device=device)
        
        for i, seq in enumerate(sequences):
            length = min(len(seq), max_length)
            padded_sequences[i, :length] = seq[:length]
        
        return padded_sequences, seq_lengths
    def construct_interaction_sequence(self, interactions, image_size=(256, 256)):
        """构造交互序列张量 - 恢复真实图像特征版"""
        if interactions is None or (isinstance(interactions, torch.Tensor) and interactions.numel() == 0) or (not isinstance(interactions, (list, tuple))):
            device = next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cpu')
            return torch.zeros(1, 0, self.feature_dim, device=device), [0]
        
        if len(interactions) < 1:
            device = next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cpu')
            return torch.zeros(1, 0, self.feature_dim, device=device), [0]
        device = next(self.parameters()).device
        sequence_features = []
        for interaction in interactions:
            if isinstance(interaction, dict):
                feat = None
                if 'features_spatial' in interaction and interaction['features_spatial'] is not None:
                    feat = interaction['features_spatial'].to(device)
                    if feat.dim() == 5:
                        feat = feat.mean(dim=(0, 1))  
                    if feat.dim() == 4:
                        feat = torch.mean(feat, dim=(2, 3))  
                    if feat.dim() == 3:
                        feat = feat.mean(dim=(1, 2))  
                    if feat.dim() == 1:
                        feat = feat.unsqueeze(0)  
                elif 'features' in interaction and interaction['features'] is not None:
                    feat = interaction['features'].to(device)
                    if feat.dim() == 5:
                        feat = feat.mean(dim=(0, 1))
                    if feat.dim() == 4:
                        feat = torch.mean(feat, dim=(2, 3))
                    if feat.dim() == 3:
                        feat = feat.mean(dim=(1, 2))
                    if feat.dim() == 1:
                        feat = feat.unsqueeze(0)
                    if feat.dim() == 2 and feat.shape[1] != self.feature_dim:
                        current_dim = feat.shape[1]
                        if current_dim > self.feature_dim:
                            key = str(current_dim)
                            if key not in self.feature_align:
                                self.feature_align[key] = nn.Linear(current_dim, self.feature_dim).to(device)
                            feat = self.feature_align[key](feat)
                        else:
                            padding = torch.zeros(feat.shape[0], self.feature_dim - current_dim, device=device)
                            feat = torch.cat([feat, padding], dim=1)
                    elif feat.dim() == 3 and feat.shape[2] != self.feature_dim:
                        current_dim = feat.shape[2]
                        if current_dim > self.feature_dim:
                            key = str(current_dim)
                            if key not in self.feature_align:
                                self.feature_align[key] = nn.Linear(current_dim, self.feature_dim).to(device)
                            feat = self.feature_align[key](feat)
                        else:
                            padding = torch.zeros(feat.shape[0], feat.shape[1], self.feature_dim - current_dim, device=device)
                            feat = torch.cat([feat, padding], dim=2)
                else:
                    feat = torch.zeros(1, self.feature_dim, device=device)
            else:
                feat = torch.zeros(1, self.feature_dim, device=device)
            sequence_features.append(feat)
        sequence_features = [f.squeeze(0) for f in sequence_features]  
        sequence_tensor = torch.stack(sequence_features, dim=0)       
        sequence_tensor = sequence_tensor.unsqueeze(0)                
        seq_lengths = [len(sequence_features)]
        return sequence_tensor, seq_lengths
    def extract_trajectory_features(self, interaction_sequence, patient_id=None):
        """提取交互轨迹特征 - 标准LSTM前向传播版"""
        device = next(self.parameters()).device
        self.device = device
        
        if interaction_sequence is None:
            return torch.zeros(1, self.feature_dim, device=device)
        if isinstance(interaction_sequence, torch.Tensor):
            if interaction_sequence.numel() == 0:
                return torch.zeros(1, self.feature_dim, device=device)
            if interaction_sequence.dim() < 2:
                return torch.zeros(1, self.feature_dim, device=device)
        elif not isinstance(interaction_sequence, (list, tuple)):
            return torch.zeros(1, self.feature_dim, device=device)
        
        if isinstance(interaction_sequence, torch.Tensor):
            sequence_tensor = interaction_sequence
        else:
            sequence_tensor, seq_lengths = self.construct_interaction_sequence(interaction_sequence)
        
        if sequence_tensor.dim() == 5:
            sequence_tensor = sequence_tensor.mean(dim=(3, 4))  
        if sequence_tensor.dim() == 4:
            sequence_tensor = sequence_tensor.mean(dim=(2, 3))
        
        if sequence_tensor.dim() == 4:
            sequence_tensor = sequence_tensor.mean(dim=(2, 3))
        elif sequence_tensor.dim() == 2:
            sequence_tensor = sequence_tensor.unsqueeze(1)
        
        if sequence_tensor.dim() == 3 and sequence_tensor.shape[2] != self.feature_dim:
            current_dim = sequence_tensor.shape[2]
            if current_dim > self.feature_dim:
                key = str(current_dim)
                if key not in self.feature_align:
                    self.feature_align[key] = nn.Linear(current_dim, self.feature_dim).to(device)
                sequence_tensor = self.feature_align[key](sequence_tensor)
            else:
                padding = torch.zeros(sequence_tensor.shape[0], sequence_tensor.shape[1], self.feature_dim - current_dim, device=device)
                sequence_tensor = torch.cat([sequence_tensor, padding], dim=2)
        elif sequence_tensor.dim() == 2 and sequence_tensor.shape[1] != self.feature_dim:
            current_dim = sequence_tensor.shape[1]
            if current_dim > self.feature_dim:
                key = str(current_dim)
                if key not in self.feature_align:
                    self.feature_align[key] = nn.Linear(current_dim, self.feature_dim).to(device)
                sequence_tensor = self.feature_align[key](sequence_tensor)
            else:
                padding = torch.zeros(sequence_tensor.shape[0], self.feature_dim - current_dim, device=device)
                sequence_tensor = torch.cat([sequence_tensor, padding], dim=1)
        
        if sequence_tensor.dim() < 3 or sequence_tensor.shape[1] == 0:
            return torch.zeros(1, self.feature_dim, device=device)
        
        batch_size = sequence_tensor.shape[0]
        sequence_tensor = sequence_tensor.to(device)
        
        max_seq_length = 8  
        if sequence_tensor.shape[1] > max_seq_length:
            sequence_tensor = sequence_tensor[:, -max_seq_length:, :]
        
        if sequence_tensor.dim() != 3:
            model_logger.error(f"[ERROR] 序列维度错误: {sequence_tensor.dim()}")
            model_logger.error(f"[ERROR] 序列形状: {sequence_tensor.shape}")
            return torch.zeros(1, self.feature_dim, device=device)
        
        h_0 = torch.zeros(2, batch_size, self.feature_dim, device=device)
        
        try:
            if not isinstance(sequence_tensor, torch.Tensor):
                sequence_tensor = torch.zeros(batch_size, 1, self.feature_dim, device=device)
            
            gru_output = self.temporal_memory(sequence_tensor, h_0)
            
            if isinstance(gru_output, tuple):
                temporal_out = gru_output[0]
                if torch.isnan(temporal_out).any() or torch.isinf(temporal_out).any():
                    model_logger.warning(f"[WARNING] GRU输出包含NaN/Inf，使用零输出替代")
                    temporal_out = torch.zeros_like(temporal_out)
                    gru_output = (temporal_out, gru_output[1])
            else:
                if torch.isnan(gru_output).any() or torch.isinf(gru_output).any():
                    model_logger.warning(f"[WARNING] GRU输出包含NaN/Inf，使用零输出替代")
                    gru_output = torch.zeros_like(gru_output)
            
            if isinstance(gru_output, torch.Tensor):
                temporal_out = gru_output
            elif isinstance(gru_output, (tuple, list)):
                temporal_out = gru_output[0] if len(gru_output) > 0 else torch.zeros(batch_size, sequence_tensor.shape[1] if sequence_tensor.dim() == 3 else 1, self.feature_dim, device=device)
            else:
                temporal_out = torch.zeros(batch_size, 1, self.feature_dim, device=device)
            
            if not isinstance(temporal_out, torch.Tensor) or temporal_out.dim() != 3:
                model_logger.error(f"[ERROR] GRU输出类型/维度错误: type={type(temporal_out)}, dim={temporal_out.dim() if isinstance(temporal_out, torch.Tensor) else 'N/A'}")
                return torch.zeros(1, self.feature_dim, device=device)
            
            attn = torch.softmax(self.attn(temporal_out), dim=1)
            
            if torch.isnan(attn).any() or torch.isinf(attn).any():
                model_logger.warning(f"[WARNING] 注意力权重包含NaN/Inf，使用均匀分布替代")
                attn = torch.ones_like(attn) / attn.shape[1]
            
            attn_feature = (temporal_out * attn).sum(dim=1)
            
            pattern_out = self.pattern_recognizer(attn_feature)
            
            if torch.isnan(pattern_out).any() or torch.isinf(pattern_out).any():
                model_logger.warning(f"[WARNING] 模式识别器输出包含NaN/Inf，使用零特征替代")
                pattern_out = torch.zeros_like(pattern_out)
            
            return pattern_out
        except Exception as e:
            model_logger.error(f"[ERROR] LSTM前向传播错误: {str(e)}")
            model_logger.error(f"[ERROR] 序列形状: {sequence_tensor.shape if isinstance(sequence_tensor, torch.Tensor) else 'N/A'}")
            model_logger.error(f"[ERROR] 序列维度: {sequence_tensor.dim() if isinstance(sequence_tensor, torch.Tensor) else 'N/A'}")
            return torch.zeros(1, self.feature_dim, device=device)
    def forward(self, interaction_sequence):
        """前向传播"""
        features = self.extract_trajectory_features(interaction_sequence)
        if not isinstance(features, torch.Tensor) or torch.isnan(features).any() or torch.isinf(features).any():
            model_logger.error("[ERROR] LSTMTemporalModule 提取的特征包含 NaN/Inf！")
            features = torch.zeros(1, self.feature_dim, device=next(self.parameters()).device)
        if features.dim() == 1:
            features = features.unsqueeze(0)
        return features  

class CrossScaleAttention(nn.Module):
    def __init__(self, feature_dim=768, num_scales=4):
        super().__init__()
        self.num_scales = num_scales
        self.feature_dim = feature_dim
        self.channel_adapters = nn.ModuleDict({
            "96": nn.Conv2d(96, feature_dim, 1),
            "192": nn.Conv2d(192, feature_dim, 1),
            "384": nn.Conv2d(384, feature_dim, 1),
            "768": nn.Conv2d(768, feature_dim, 1),
        })
        self.q_proj = nn.Conv2d(feature_dim, feature_dim, 1)
        
        self.scale = feature_dim ** -0.5
        
        self.out_proj = nn.Conv2d(feature_dim, feature_dim, 1)
        
        self.k_proj = nn.Conv2d(feature_dim, feature_dim, 1)
        self.v_proj = nn.Conv2d(feature_dim, feature_dim, 1)
    def forward(self, feats, mem_state):
        if feats is None or (isinstance(feats, (list, tuple)) and len(feats) == 0):
            return mem_state
        if torch.isnan(mem_state).any() or torch.isinf(mem_state).any():
            model_logger.error("[ERROR] CrossScaleAttention 输入 mem_state 包含 NaN/Inf！")
            return torch.zeros_like(mem_state)
        feats = list(feats)
        for i in range(len(feats)):
            if feats[i].dim() == 2:
                feats[i] = feats[i].unsqueeze(-1).unsqueeze(-1)
            elif feats[i].dim() == 3:
                feats[i] = feats[i].unsqueeze(-1)
            
            if torch.isnan(feats[i]).any() or torch.isinf(feats[i]).any():
                model_logger.error(f"[ERROR] CrossScaleAttention 输入 feats[{i}] 包含 NaN/Inf！")
                feats[i] = torch.zeros_like(feats[i])
        if feats[0].dim() != 4:
            feats[0] = feats[0].unsqueeze(-1).unsqueeze(-1)
        B, C = feats[0].shape[:2]
        target_size = (16, 16)
        if not isinstance(mem_state, torch.Tensor):
            return mem_state
        if mem_state.dim() == 4:
            mem_state = mem_state.mean(dim=(2, 3))
        elif mem_state.dim() == 2:
            mem_state = mem_state
        else:
            mem_state = mem_state.view(B, -1)
        if mem_state.dim() != 2:
            mem_state = mem_state.view(B, -1)
        mem_context = self.q_proj(mem_state.unsqueeze(-1).unsqueeze(-1)).view(B, self.feature_dim, 1, 1)
        
        weighted_feats = []
        
        for f in feats:
            in_channels = f.shape[1]
            if isinstance(in_channels, torch.Tensor):
                in_channels = in_channels.item()
            aligned_f = f
            if in_channels != self.feature_dim:
                key = str(in_channels)
                key = str(in_channels)
                if key in self.channel_adapters:
                    aligned_f = self.channel_adapters[key](f)
                else:
                    aligned_f = f  
            
            k = self.k_proj(aligned_f)
            spatial_attn = torch.sigmoid((k * mem_context).sum(dim=1, keepdim=True))
            weighted_f = aligned_f * spatial_attn
            
            weighted_f = F.interpolate(weighted_f, size=target_size, mode='bilinear', align_corners=False)
            weighted_feats.append(weighted_f)
        
        if len(weighted_feats) == 1:
            output = self.out_proj(weighted_feats[0])
        else:
            output = self.out_proj(torch.cat(weighted_feats, dim=1))
        
        if torch.isnan(output).any() or torch.isinf(output).any():
            model_logger.error("[ERROR] CrossScaleAttention 输出包含 NaN/Inf！")
            return torch.zeros_like(output)
        
        return output

class MultiScaleFeatureFusion(nn.Module):
    """Multi-Scale Feature Temporal Fusion module
    按照论文描述实现：在四个空间尺度 s ∈ {8, 16, 32, 64} 上进行自适应特征融合
    """
    def __init__(self, image_size=256, feature_dims=[768, 384, 192, 96], in_dim=768):
        super().__init__()
        self.in_dim = in_dim
        self.scale_adapters = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(8), nn.Conv2d(in_dim, feature_dims[0], 1), nn.BatchNorm2d(feature_dims[0]), nn.ReLU()),
            nn.Sequential(nn.AdaptiveAvgPool2d(16), nn.Conv2d(in_dim, feature_dims[1], 1), nn.BatchNorm2d(feature_dims[1]), nn.ReLU()),
            nn.Sequential(nn.AdaptiveAvgPool2d(32), nn.Conv2d(in_dim, feature_dims[2], 1), nn.BatchNorm2d(feature_dims[2]), nn.ReLU()),
            nn.Sequential(nn.AdaptiveAvgPool2d(64), nn.Conv2d(in_dim, feature_dims[3], 1), nn.BatchNorm2d(feature_dims[3]), nn.ReLU()),
        ])
        
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dims[0] * 2, feature_dims[0], kernel_size=1),  
                nn.BatchNorm2d(feature_dims[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dims[0], 2, kernel_size=1, bias=False),  
                nn.Softmax(dim=1)  
            ),
            nn.Sequential(
                nn.Conv2d(feature_dims[1] * 2, feature_dims[1], kernel_size=1),
                nn.BatchNorm2d(feature_dims[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dims[1], 2, kernel_size=1, bias=False),
                nn.Softmax(dim=1)
            ),
            nn.Sequential(
                nn.Conv2d(feature_dims[2] * 2, feature_dims[2], kernel_size=1),
                nn.BatchNorm2d(feature_dims[2]),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dims[2], 2, kernel_size=1, bias=False),
                nn.Softmax(dim=1)
            ),
            nn.Sequential(
                nn.Conv2d(feature_dims[3] * 2, feature_dims[3], kernel_size=1),
                nn.BatchNorm2d(feature_dims[3]),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dims[3], 2, kernel_size=1, bias=False),
                nn.Softmax(dim=1)
            ),
        ])
    def extract_multi_scale_features(self, x):
        """提取多尺度特征"""
        feats = []
        for layer in self.scale_adapters:
            feats.append(layer(x))
        return feats
    def fuse_features(self, current_img_feat, historical_img_feat, scale_weights=None):
        """
        按照论文公式 (3) 实现自适应融合:
        F̃_t^s = W_s[0] ⊙ F_t^s + W_s[1] ⊙ C_t^s
        其中 W_s 通过公式 (2) 的门控网络计算得到
        """
        curr_list = self.extract_multi_scale_features(current_img_feat)
        hist_list = self.extract_multi_scale_features(historical_img_feat)
        fused = []
        for i, (c, h) in enumerate(zip(curr_list, hist_list)):
            if c.shape[2:] != h.shape[2:]:
                h = F.interpolate(h, size=c.shape[2:], mode='bilinear', align_corners=False)
            concatenated = torch.cat([c, h], dim=1)  
            
            weights = self.gates[i](concatenated)  
            
            fused_feat = weights[:, 0:1] * c + weights[:, 1:2] * h
            fused.append(fused_feat)
        target_size = fused[-1].shape[2:]  
        resized_fused = []
        for f in fused:
            if f.shape[2:] != target_size:
                resized_fused.append(F.interpolate(f, size=target_size, mode='bilinear', align_corners=False))
            else:
                resized_fused.append(f)
        combined = torch.cat(resized_fused, dim=1)
        return combined

class MultiScaleTemporalFusion(nn.Module):
    """Multi-scale temporal interaction fusion main module - Complete version (including LSTM temporal modeling and multi-scale fusion)"""
    def __init__(self, image_size=256, feature_dim=768, num_scales=4, 
                 ablation_no_multi_scale=False, ablation_no_trajectory=False):
        super().__init__()
        self.feature_dim = feature_dim  
        self.ablation_no_multi_scale = ablation_no_multi_scale
        self.ablation_no_trajectory = ablation_no_trajectory
        self.temporal_memory = TemporalMemoryModule(feature_dim=feature_dim)
        
        if not ablation_no_multi_scale:
            self.multi_scale_fusion = MultiScaleFeatureFusion(
                image_size=image_size,
                feature_dims=[feature_dim // (2 ** i) for i in range(num_scales)],
                in_dim=feature_dim
            )
        
        total_scale_channels = sum([feature_dim // (2 ** i) for i in range(num_scales)])  
        self.feature_projector = nn.Sequential(
            nn.Conv2d(total_scale_channels, feature_dim, kernel_size=1),  
            nn.GroupNorm(8, feature_dim),
            nn.ReLU(inplace=True)
        )
        
        self.temporal_projector = nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1)
        if not ablation_no_trajectory:
            self.trajectory_analyzer = LSTMTemporalModule(feature_dim)
        self.fusion_weights = nn.Parameter(torch.ones(num_scales) / num_scales, requires_grad=True)
        
        self.save_weights = True
        
        self.trajectory_fusion_weight = nn.Parameter(torch.tensor(-2.5, dtype=torch.float32), requires_grad=True)
        
        self.fusion_warmup_epochs = 50  
        self.max_fusion_weight = 0.5  
        
        self.temporal_gate = nn.Parameter(torch.tensor(-3.0, dtype=torch.float32), requires_grad=True)  
        self.stage_factor = 0.0  
        self.stage_1_end = 10  
        self.stage_2_end = 30  
        self.output_adapter = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 1)
        )
        self.cross_scale_attention = CrossScaleAttention(feature_dim=feature_dim, num_scales=num_scales)
    def get_temporal_context(self, patient_id, category=None): 
        if self.ablation_no_trajectory or not hasattr(self, 'trajectory_analyzer'):
            return None
        
        patient_id_str = self.temporal_memory._safe_convert_id(patient_id) 
        if patient_id_str not in self.temporal_memory.history_buffer: 
            return None 
        history_list = self.temporal_memory.get_history(patient_id, category)
        if not isinstance(history_list, (list, tuple)) or len(history_list) < 1: 
            return torch.zeros(1, 0, self.trajectory_analyzer.feature_dim, device=next(self.parameters()).device) 
        
        recent_history = history_list[-15:]  
        interaction_sequences = [] 
        for data in recent_history: 
            if not isinstance(data, dict):
                continue
            feat_spatial = data.get('features_spatial', None)
            feat = feat_spatial if feat_spatial is not None else data.get('features', None) 
            seq_entry = { 
                'features': feat, 
                'point_coords': data.get('point_coords', None), 
                'point_labels': data.get('point_labels', None), 
                'interaction_round': data.get('interaction_round', 0) 
            } 
            interaction_sequences.append(seq_entry) 
        
        if len(interaction_sequences) > 10:
            interaction_sequences = interaction_sequences[-10:]
        
        sequence_tensor, _ = self.trajectory_analyzer.construct_interaction_sequence(interaction_sequences) 
        
        if sequence_tensor.dim() == 3 and sequence_tensor.shape[2] != self.trajectory_analyzer.feature_dim:
            device = next(self.parameters()).device
            current_dim = sequence_tensor.shape[2]
            if current_dim > self.trajectory_analyzer.feature_dim:
                if not hasattr(self, 'temporal_projection_weight'):
                    self.temporal_projection_weight = nn.Parameter(torch.randn(self.trajectory_analyzer.feature_dim, current_dim, device=device))
                    self.temporal_projection_bias = nn.Parameter(torch.zeros(self.trajectory_analyzer.feature_dim, device=device))
                sequence_tensor = torch.matmul(sequence_tensor, self.temporal_projection_weight.t()) + self.temporal_projection_bias
            else:
                padding = torch.zeros(sequence_tensor.shape[0], sequence_tensor.shape[1], self.trajectory_analyzer.feature_dim - current_dim, device=device)
                sequence_tensor = torch.cat([sequence_tensor, padding], dim=2)
        
        return sequence_tensor.to(next(self.parameters()).device)
    @staticmethod
    def _safe_get_value(value):
        """彻底安全的值获取方法 - 确保永远返回 Python 标量"""
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            else:
                return value.mean().item()
        elif isinstance(value, (list, tuple)):
            if len(value) > 0:
                return MultiScaleTemporalFusion._safe_get_value(value[0])
            else:
                return 0.0
        elif isinstance(value, (int, float)):
            return value
        else:
            return 0.0
    def model_temporal_dependencies(self, patient_id, category, current_prompt): 
        """按类别和交互轮次动态加权，突出有效历史信息 - 彻底修复版"""
        try:
            patient_id_str = self.temporal_memory._safe_convert_id(patient_id)
            category_str = self.temporal_memory._safe_convert_category(category)
            if patient_id_str not in self.temporal_memory.history_buffer:
                return current_prompt, None
            patient_history = self.temporal_memory.history_buffer[patient_id_str]
            history = patient_history.get(category_str, [])
            if not isinstance(history, (list, tuple)) or len(history) == 0:
                return current_prompt, None
            weights = []
            useful_history = []
            for i, hist in enumerate(reversed(history)):
                if i < len(self.fusion_weights):
                    reversed_weights = torch.flip(self.fusion_weights, dims=[0])
                    base_weight_tensor = reversed_weights[i]
                    base_weight = base_weight_tensor.item()  
                else:
                    base_weight_tensor = torch.tensor(0.05, device=self.fusion_weights.device)
                    base_weight = 0.05
                
                dice_change = MultiScaleTemporalFusion._safe_get_value(hist.get('dice_change', 0.0))
                if hasattr(dice_change, 'item'):
                    dice_change = dice_change.item()
                
                if dice_change > 0.02:
                    base_weight_tensor = base_weight_tensor * 2.0
                    base_weight = base_weight * 2.0
                
                weights.append(base_weight_tensor)  
                useful_history.append(hist)
        
            if isinstance(current_prompt, dict):
                for hist, w in zip(useful_history, weights):
                    w_value = MultiScaleTemporalFusion._safe_get_value(w)
                    if hasattr(w_value, 'item'):
                        w_value = w_value.item()
                    
                    prompt_type = MultiScaleTemporalFusion._safe_get_value(hist.get('prompt_type', ''))
                    if not isinstance(prompt_type, str):
                        prompt_type = str(prompt_type)
                    
                    dice_change = MultiScaleTemporalFusion._safe_get_value(hist.get('dice_change', 0.0))
                    if isinstance(dice_change, torch.Tensor):
                        dice_change = dice_change.item() if dice_change.numel() == 1 else float(dice_change.mean())
                    
                    if category_str == 'RV' and isinstance(hist, dict) and prompt_type == 'points' and w_value > 0.3:
                        current_prompt['type'] = 'points'
                        break
                    elif category_str == 'Myo' and isinstance(hist, dict) and prompt_type == 'points':
                        if dice_change > 0.03:
                            current_prompt['type'] = 'points+text'
                            break
            return current_prompt, weights
        except Exception as e:
            model_logger.debug(f"[DEBUG] model_temporal_dependencies detailed error: {str(e)}")
            model_logger.debug(f"[DEBUG] category type: {type(category)}, value: {category}")
            model_logger.debug(f"[DEBUG] patient_id type: {type(patient_id)}, value: {patient_id}")
            import traceback
            model_logger.debug(f"[DEBUG] Stack trace: {traceback.format_exc()}")
            return current_prompt, None
    def update_interaction_history(self, patient_id, current_data, global_epoch=0):
        """更新历史记录 - 修复版：支持传入epoch"""
        self.temporal_memory.update_history(patient_id, current_data, global_epoch)
    def _refine_points(self, mask_correction, current_points, image_size):
        """根据掩码修正量优化点提示位置"""
        if isinstance(mask_correction, torch.Tensor) and mask_correction.numel() > 0:
            correction_value = mask_correction.mean().item()
            if isinstance(current_points, torch.Tensor):
                adjustment = torch.tensor([correction_value * 10, correction_value * 10],
                                          device=current_points.device,
                                          dtype=current_points.dtype)
                refined_points = current_points + adjustment
                refined_points = torch.clamp(refined_points, 0.0, image_size)
                return refined_points
        return current_points
    def update_fusion_weight(self, current_epoch):
        """根据epoch动态调整融合权重，实现平滑过渡"""
        if current_epoch < self.fusion_warmup_epochs:
            progress = current_epoch / self.fusion_warmup_epochs
            target_weight = -5.0 + progress * (5.0 + 0.0)  
            self.trajectory_fusion_weight.data.fill_(target_weight)
        return torch.sigmoid(self.trajectory_fusion_weight).item()
    def forward(self, current_prompt, history, image_features):
        """前向传播 - 优化版（支持消融实验）"""
        self.fusion_weights.requires_grad = True
        self.trajectory_fusion_weight.requires_grad = True
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(-1).unsqueeze(-1)
        elif image_features.dim() == 3:
            image_features = image_features.unsqueeze(-1)
        
        def _safe_scalar(val, default=0):
            if val is None:
                return default
            if isinstance(val, torch.Tensor):
                if val.numel() == 1:
                    return val.item()
                else:
                    return val.float().mean().item()
            if isinstance(val, (list, tuple)):
                return _safe_scalar(val[0]) if len(val) > 0 else default
            if isinstance(val, (int, float, bool, str)):
                return val
            return default
        interaction_round = _safe_scalar(current_prompt.get('interaction_round', 1))
        patient_id = current_prompt.get('patient_id', "default")
        if isinstance(patient_id, (list, tuple)):
            patient_id = patient_id[0] if len(patient_id) > 0 else "default"
        patient_id = str(patient_id)
        
        category = current_prompt.get('category', 'unknown')
        if isinstance(category, (list, tuple)):
            category = category[0] if len(category) > 0 else "unknown"
        category = str(category)
        
        global_epoch = _safe_scalar(current_prompt.get('global_epoch', 0))
        dice_val = _safe_scalar(current_prompt.get('dice', 0.0))
        tci_val = _safe_scalar(current_prompt.get('tci', 0.0))
        labels_tensor = current_prompt.get('labels', None) if isinstance(current_prompt, dict) else None
        label_sum_val = None
        if labels_tensor is not None:
            if isinstance(labels_tensor, torch.Tensor):
                label_sum_val = labels_tensor.sum().item()
            elif isinstance(labels_tensor, (int, float)):
                label_sum_val = labels_tensor
            elif isinstance(labels_tensor, (list, tuple)) and len(labels_tensor) > 0:
                if isinstance(labels_tensor[0], torch.Tensor):
                    label_sum_val = sum(t.item() if t.numel() == 1 else t.sum().item() for t in labels_tensor)
                else:
                    label_sum_val = sum(float(x) for x in labels_tensor)
        
        point_coords = current_prompt.get('point_coords', None) if isinstance(current_prompt, dict) else None
        point_labels = current_prompt.get('point_labels', None) if isinstance(current_prompt, dict) else None
        temporal_data = {
            'features': image_features.detach().clone(),
            'point_coords': point_coords,
            'point_labels': point_labels,
            'interaction_round': interaction_round,
            'dice': dice_val,
            'tci': tci_val,
            'category': category,
            'epoch': global_epoch,
            'label_sum': label_sum_val
        }
        
        history_list = self.temporal_memory.get_history(patient_id, category) if hasattr(self.temporal_memory, 'get_history') else []
        self.temporal_memory.store(patient_id, category, temporal_data, global_epoch)
        device = image_features.device
        
        try:
            optimized_prompt, temporal_weights = self.model_temporal_dependencies(
                patient_id,
                category,
                current_prompt
            )
            if isinstance(current_prompt, dict) and isinstance(optimized_prompt, dict):
                current_prompt.update(optimized_prompt)
        except Exception as e:
            model_logger.debug(f"[DEBUG] model_temporal_dependencies error: {str(e)}")
            optimized_prompt, temporal_weights = current_prompt, None
        
        temporal_context = torch.zeros_like(image_features, device=device)
        if isinstance(history_list, (list, tuple)) and len(history_list) > 0:
            recent_history = history_list[-15:]
            total_weight = 0.0
            weighted_features = []
            weights = []
            for i, hist in enumerate(reversed(recent_history)):
                if not isinstance(hist, dict):
                    continue
                hist_quality = hist.get('quality', hist.get('dice', 0.0))
                hist_tci = hist.get('tci', 0.0)
                quality_weight = (hist_quality * 0.7 + hist_tci * 0.3)
                quality_weight = max(quality_weight, 0.1)
                time_weight = math.exp(-i * 0.1)
                combined_weight = quality_weight * time_weight
                
                feat_key = 'features_spatial' if 'features_spatial' in hist else 'features'
                hist_features = hist.get(feat_key, None)
                if isinstance(hist_features, torch.Tensor):
                    if hist_features.device.type == 'cpu':
                        hist_features = hist_features.to(device)
                    if hist_features.shape != image_features.shape:
                        hist_features = F.interpolate(
                            hist_features.unsqueeze(0) if hist_features.dim() == 3 else hist_features,
                            size=image_features.shape[2:],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)
                    weighted_features.append(hist_features.detach())
                    weights.append(combined_weight)
                    total_weight += combined_weight
            
            if total_weight > 0 and len(weighted_features) > 0:
                for feat, weight in zip(weighted_features, weights):
                    temporal_context += (weight / total_weight) * feat
        
        trajectory_feature = None 
        if not self.ablation_no_trajectory and hasattr(self, 'trajectory_analyzer'):
            try: 
                sequence_tensor = self.get_temporal_context(patient_id, category) 
                if sequence_tensor is not None and isinstance(sequence_tensor, torch.Tensor):
                    seq_len = sequence_tensor.shape[1]
                    if isinstance(seq_len, torch.Tensor):
                        seq_len = seq_len.item()
                    if seq_len >= 1: 
                        sequence_tensor = sequence_tensor.to(device) 
                        trajectory_feature = self.trajectory_analyzer.extract_trajectory_features(sequence_tensor, patient_id=patient_id) 
                    else:
                        trajectory_feature = image_features.mean(dim=(2, 3), keepdim=True) 
                else: 
                    trajectory_feature = image_features.mean(dim=(2, 3), keepdim=True) 
            except Exception as e: 
                model_logger.error(f"[ERROR] 时序融合错误: {str(e)}") 
                trajectory_feature = image_features.mean(dim=(2, 3), keepdim=True)
        
        if isinstance(history_list, (list, tuple)) and len(history_list) > 0:
            if self.ablation_no_multi_scale and self.ablation_no_trajectory:
                combined_feature = image_features + 0.2 * temporal_context
                enhanced_features = self.output_adapter(combined_feature)
                return enhanced_features
            elif self.ablation_no_multi_scale:
                stage_factor = 0.4 if global_epoch >= self.stage_2_end else 0.01 if global_epoch >= self.stage_1_end else 0.0
                fused_features = image_features + 0.3 * temporal_context
                combined_feature = fused_features
                
                if not self.ablation_no_trajectory and trajectory_feature is not None and isinstance(trajectory_feature, torch.Tensor):
                    if trajectory_feature.dim() == 2:
                        trajectory_feature = trajectory_feature.unsqueeze(-1).unsqueeze(-1)
                    elif trajectory_feature.dim() == 3:
                        trajectory_feature = trajectory_feature.unsqueeze(-1)
                    if trajectory_feature.shape[2:] != combined_feature.shape[2:]:
                        trajectory_feature = F.interpolate(
                            trajectory_feature,
                            size=combined_feature.shape[2:],
                            mode='nearest'
                        )
                    trajectory_feature = trajectory_feature.expand_as(combined_feature)
                    trajectory_feature = trajectory_feature.to(device)
                    
                    gate_weight = torch.sigmoid(self.trajectory_fusion_weight).view(1, 1, 1, 1)
                    gate_weight = gate_weight * stage_factor
                    gate_weight = torch.clamp(gate_weight, 0.0, 0.3)
                    
                    if combined_feature.shape[1] == trajectory_feature.shape[1] and combined_feature.shape[2:] == trajectory_feature.shape[2:]:
                        temporal_fusion_input = torch.cat([combined_feature, trajectory_feature], dim=1)
                        if temporal_fusion_input.shape[1] == self.feature_dim * 2:
                            projected_fusion = self.temporal_projector(temporal_fusion_input)
                            combined_feature = combined_feature + gate_weight * projected_fusion
                        else:
                            combined_feature = combined_feature + gate_weight * trajectory_feature
                
                if combined_feature.dim() == 2:
                    combined_feature = combined_feature.unsqueeze(-1).unsqueeze(-1)
                elif combined_feature.dim() == 3:
                    combined_feature = combined_feature.unsqueeze(-1)
                if combined_feature.dim() != 4:
                    combined_feature = combined_feature.view(combined_feature.shape[0], self.feature_dim, 1, 1)
                
                enhanced_features = self.output_adapter(combined_feature)
                return enhanced_features
            else:
                if global_epoch < self.stage_1_end:
                    stage_factor = 0.0
                    if not self.ablation_no_trajectory and hasattr(self, 'trajectory_analyzer'):
                        for param in self.trajectory_analyzer.parameters():
                            param.requires_grad = False
                    for param in self.cross_scale_attention.parameters():
                        param.requires_grad = False
                elif global_epoch < self.stage_2_end:
                    stage_factor = 0.01 + (global_epoch - self.stage_1_end) * (0.19 / (self.stage_2_end - self.stage_1_end))
                    if not self.ablation_no_trajectory and hasattr(self, 'trajectory_analyzer'):
                        for param in self.trajectory_analyzer.parameters():
                            param.requires_grad = True
                    for param in self.cross_scale_attention.parameters():
                        param.requires_grad = True
                else:
                    stage_factor = 0.4
                    if not self.ablation_no_trajectory and hasattr(self, 'trajectory_analyzer'):
                        for param in self.trajectory_analyzer.parameters():
                            param.requires_grad = True
                    for param in self.cross_scale_attention.parameters():
                        param.requires_grad = True
                
                self.stage_factor = stage_factor
                
                normalized_scale_weights = F.softmax(self.fusion_weights, dim=0)
                fused_features = self.multi_scale_fusion.fuse_features(
                    image_features,
                    temporal_context,
                    scale_weights=normalized_scale_weights
                )
                if isinstance(fused_features, torch.Tensor):
                    if fused_features.dim() == 2:
                        fused_features = fused_features.unsqueeze(-1).unsqueeze(-1)
                    elif fused_features.dim() == 3:
                        fused_features = fused_features.unsqueeze(-1)
                    fused_features = self.feature_projector(fused_features)
                if isinstance(fused_features, torch.Tensor) and fused_features.dim() == 4 and fused_features.shape[2:] != (16, 16):
                    fused_features = F.interpolate(fused_features, size=(16, 16), mode='bilinear', align_corners=False)
                
                if not self.ablation_no_trajectory and trajectory_feature is not None and isinstance(trajectory_feature, torch.Tensor):
                    gate = torch.sigmoid(self.temporal_gate) * stage_factor
                    gate = torch.clamp(gate, 0.0, 0.3)
                    mem_state = trajectory_feature.mean(dim=(2, 3)) if trajectory_feature.dim() == 4 else trajectory_feature
                    attention_output = self.cross_scale_attention([fused_features], mem_state)
                    fused_features = fused_features + gate * attention_output
                else:
                    fused_features = image_features if not isinstance(fused_features, torch.Tensor) else fused_features
                
                combined_feature = fused_features
                if combined_feature.dim() != 4:
                    if combined_feature.dim() == 2:
                        combined_feature = combined_feature.unsqueeze(-1).unsqueeze(-1)
                    elif combined_feature.dim() == 3:
                        combined_feature = combined_feature.unsqueeze(-1)
                if not self.ablation_no_trajectory and trajectory_feature is not None and isinstance(trajectory_feature, torch.Tensor):
                    if trajectory_feature.dim() == 2:
                        trajectory_feature = trajectory_feature.unsqueeze(-1).unsqueeze(-1)
                    elif trajectory_feature.dim() == 3:
                        trajectory_feature = trajectory_feature.unsqueeze(-1)
                    if trajectory_feature.shape[2:] != combined_feature.shape[2:]:
                        trajectory_feature = F.interpolate(
                            trajectory_feature,
                            size=combined_feature.shape[2:],
                            mode='nearest'
                        )
                    trajectory_feature = trajectory_feature.expand_as(combined_feature)
                    trajectory_feature = trajectory_feature.to(device)
                    
                    if combined_feature.shape[2:] != trajectory_feature.shape[2:]:
                        combined_feature = F.interpolate(
                            combined_feature,
                            size=trajectory_feature.shape[2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    gate_weight = torch.sigmoid(self.trajectory_fusion_weight).view(1, 1, 1, 1)
                    gate_weight = gate_weight * stage_factor
                    gate_weight = torch.clamp(gate_weight, 0.0, 0.3)
                    if combined_feature.shape[1] == trajectory_feature.shape[1] and combined_feature.shape[2:] == trajectory_feature.shape[2:]:
                        temporal_fusion_input = torch.cat([combined_feature, trajectory_feature], dim=1)
                        if temporal_fusion_input.shape[1] == self.feature_dim * 2:
                            projected_fusion = self.temporal_projector(temporal_fusion_input)
                            combined_feature = combined_feature + gate_weight * projected_fusion
                        else:
                            combined_feature = combined_feature + gate_weight * trajectory_feature
                    else:
                        combined_feature = combined_feature + gate_weight * trajectory_feature
                    
                    if torch.isnan(combined_feature).any() or torch.isinf(combined_feature).any():
                        model_logger.warning(f"[WARNING] combined_feature包含NaN/Inf，使用原始特征替代")
                        combined_feature = torch.where(
                            torch.isnan(combined_feature) | torch.isinf(combined_feature),
                            image_features,
                            combined_feature
                        )
                if combined_feature.dim() == 2:
                    combined_feature = combined_feature.unsqueeze(-1).unsqueeze(-1)
                elif combined_feature.dim() == 3:
                    combined_feature = combined_feature.unsqueeze(-1)
                if combined_feature.dim() != 4:
                    combined_feature = combined_feature.view(combined_feature.shape[0], self.feature_dim, 1, 1)
                enhanced_features = self.output_adapter(combined_feature)
                return enhanced_features
        else:
            if image_features.dim() == 2:
                image_features = image_features.unsqueeze(-1).unsqueeze(-1)
            elif image_features.dim() == 3:
                image_features = image_features.unsqueeze(-1)
            return self.output_adapter(image_features)
    
class AdaptiveInteractionGuide:
    def __init__(self):
        self.max_points = 7
        self.min_points = 1
    def compute_uncertainty(self, pred_mask, gt_mask):
        """计算预测的不确定性"""
        try:
            return torch.logical_xor(pred_mask, gt_mask)
        except Exception as e:
            model_logger.debug(f"[DEBUG] Interaction guide uncertainty calculation error: {str(e)}")
            return torch.zeros_like(pred_mask)
    def compute_iou(self, pred_masks, labels):
        """计算预测掩码和真实标签的IoU"""
        try:
            pred_binary = (torch.sigmoid(pred_masks) > 0.5)
            labels_binary = (labels > 0)
            intersection = torch.logical_and(pred_binary, labels_binary).sum(dim=(1, 2, 3))
            union = torch.logical_or(pred_binary, labels_binary).sum(dim=(1, 2, 3))
            iou = intersection.float() / (union.float() + 1e-8)
            return iou.mean()
        except Exception as e:
            model_logger.debug(f"[DEBUG] Interaction guide IoU calculation error: {str(e)}")
            return torch.tensor(0.0, device=pred_masks.device)
    def recommend_points(self, current_mask, uncertainty_map, existing_points=None):
        """
        推荐最优交互点
        Args:
            current_mask: 当前预测的掩码
            uncertainty_map: 不确定性图
            existing_points: 已有的交互点
        Returns:
            推荐的点坐标和标签列表
        """
        if existing_points is None:
            existing_points = []
        batch_size = uncertainty_map.shape[0]
        recommended_points = []
        for b in range(batch_size):
            current_uncertainty = uncertainty_map[b].cpu().numpy()
            current_pred = current_mask[b].cpu().numpy()
            uncertain_indices = np.argwhere(current_uncertainty == 1)
            if len(uncertain_indices) > 0:
                point_indices = np.random.choice(len(uncertain_indices),
                                                 min(self.max_points, len(uncertain_indices)),
                                                 replace=False)
                selected_indices = uncertain_indices[point_indices]
                points = []
                labels = []
                for idx in selected_indices:
                    x, y = idx[0], idx[1]
                    if current_pred[x, y] == 0 and current_uncertainty[x, y] == 1:
                        labels.append(1)  
                    elif current_pred[x, y] == 1 and current_uncertainty[x, y] == 1:
                        labels.append(0)  
                    else:
                        labels.append(-1)  
                    points.append((y, x))  
                recommended_points.append((np.array(points), np.array(labels)))
            else:
                recommended_points.append((np.array([]), np.array([])))
        return recommended_points

class IMISNet(nn.Module):
    def __init__(
            self,
            sam,
            test_mode=False,
            multimask_output=True,
            category_weights=None,
            select_mask_num=None
    ):
        super().__init__()
        self.device = sam.device
        self.image_encoder = sam.image_encoder
        self.mask_decoder = sam.mask_decoder
        self.prompt_encoder = sam.prompt_encoder
        self.test_mode = test_mode
        self.multimask_output = multimask_output
        self.category_weights = category_weights
        self.select_mask_num = select_mask_num
        self.image_format = sam.image_format
        self.image_size = sam.prompt_encoder.input_image_size
        if category_weights is not None:
            self.load_category_weights(category_weights)
    def compute_uncertainty(self, pred_masks, labels):
        """计算预测掩码的不确定性图"""
        pred_binary = (torch.sigmoid(pred_masks) > 0.5)
        label_binary = (labels > 0.5)
        return torch.logical_xor(pred_binary, label_binary)

class TMFNet(IMISNet):
    def __init__(self, sam, test_mode=False, fusion_warmup_epochs=20, max_fusion_strength=1.5, num_classes=4, 
                 ablation_no_multi_scale=False, ablation_no_trajectory=False, **kwargs):
        super().__init__(sam, test_mode, **kwargs)
        self.use_auto_prompt = False  
        self.num_classes = num_classes  
        feature_dim = 768  
        self.multiscale_temporal_fusion = MultiScaleTemporalFusion(
            image_size=256,  
            feature_dim=feature_dim,
            num_scales=4,
            ablation_no_multi_scale=ablation_no_multi_scale,
            ablation_no_trajectory=ablation_no_trajectory
        )  
        self.temporal_memory = self.multiscale_temporal_fusion.temporal_memory
        self.temporal_weight = nn.Parameter(torch.tensor([0.2], dtype=torch.float32))  
        self.interaction_step = 0  
        self.interaction_history = {}  
        self.use_adaptive_threshold = True  
        self.postprocess_trigger_threshold = 0.6  
        self.difficult_dice_threshold = 0.7
        self.interaction_guide = AdaptiveInteractionGuide()
        self.best_threshold = 0.5  
        self.fusion_warmup_epochs = fusion_warmup_epochs      
        self.max_fusion_strength = max_fusion_strength      
        print(f"✅ TMFNet initialized with fusion parameters: warmup_epochs={self.fusion_warmup_epochs}, max_strength={self.max_fusion_strength}")
    def check_quality_gate(self, dice, tci, target_dice=0.9):
        return dice > target_dice and tci > 0.8
    def binary_mask_with_threshold(self, pred_masks, labels=None):
        """
        使用自适应阈值或最优阈值将预测掩码转换为二值掩码
        对于困难样本自动调整阈值
        
        按照论文描述：
        - 训练阶段：基于 IoU 动态调整阈值
        - 推理阶段：使用预测置信度作为代理指标
        """
        if self.use_adaptive_threshold and labels is not None:
            threshold = self.get_adaptive_threshold(pred_masks, labels)
        elif self.use_adaptive_threshold and labels is None:
            threshold = self.get_confidence_based_threshold(pred_masks)
        else:
            threshold = self.best_threshold  
        return (torch.sigmoid(pred_masks) > threshold).bool()
    def get_adaptive_threshold(self, pred_masks, labels):
        """
        训练阶段：根据当前预测和标签的 IoU 动态调整阈值
        优化后的阈值设置，更加平滑和合理
        """
        current_iou = self.compute_iou(pred_masks, labels)
        if current_iou < 0.55:
            return 0.38  
        elif current_iou < 0.7:
            return 0.42  
        elif current_iou < 0.8:
            return 0.48  
        return 0.52  
    
    def get_confidence_based_threshold(self, pred_masks):
        """
        推理阶段：基于预测置信度的自适应阈值
        按照论文描述，使用预测置信度作为 IoU 的代理指标
        
        核心思想：
        - 高置信度预测：使用较高阈值（0.5-0.55），确保精确边界
        - 中等置信度预测：使用标准阈值（0.45-0.5）
        - 低置信度预测：使用较低阈值（0.35-0.45），捕获更多前景区域
        
        Args:
            pred_masks: 预测掩码 [B, 1, H, W]
            
        Returns:
            float: 自适应阈值
        """
        sigmoid_pred = torch.sigmoid(pred_masks)
        
        mean_confidence = sigmoid_pred.mean().item()
        
        high_conf_ratio = (sigmoid_pred > 0.6).float().mean().item()
        
        confidence_std = sigmoid_pred.std().item()
        
        confidence_score = (
            mean_confidence * 0.4 + 
            high_conf_ratio * 0.4 + 
            (1.0 - confidence_std) * 0.2
        )
        
        if confidence_score > 0.7:
            return 0.52
        elif confidence_score > 0.55:
            return 0.48
        elif confidence_score > 0.4:
            return 0.42
        else:
            return 0.38
    def detect_difficult_samples(self, pred_mask, label_mask, test_mode=False):
        """
        检测困难样本
        基于IoU、预测不确定性和错误类型分析与动态阈值调整
        """
        try:
            pred_mask = pred_mask if isinstance(pred_mask, torch.Tensor) else torch.tensor(pred_mask)
            label_mask = label_mask if isinstance(label_mask, torch.Tensor) else torch.tensor(label_mask)
            pred_binary = (torch.sigmoid(pred_mask) > 0.5).float()
            label_binary = (label_mask > 0).float()
            intersection = (pred_binary * label_binary).sum()
            union = pred_binary.sum() + label_binary.sum() - intersection
            iou = intersection / (union + 1e-8) if union > 0 else 0
            dice = (2 * intersection) / (pred_binary.sum() + label_binary.sum() + 1e-8) if (
                                                                                                       pred_binary.sum() + label_binary.sum()) > 0 else 0
            pred_prob = torch.sigmoid(pred_mask)
            conf_std = pred_prob.std()
            total_pixels = pred_binary.shape[2] * pred_binary.shape[3] if len(pred_binary.shape) > 2 else pred_binary.shape[1] * pred_binary.shape[2]
            mask_coverage = pred_binary.sum() / total_pixels
            if len(pred_binary.shape) > 2:
                mask_2d = pred_binary.squeeze(0)
                label_2d = label_binary.squeeze(0)
            else:
                mask_2d = pred_binary
                label_2d = label_binary
            false_pos = (mask_2d > 0) & (label_2d == 0)
            false_neg = (mask_2d == 0) & (label_2d > 0)
            false_pos_area = torch.sum(false_pos).float().item()
            false_neg_area = torch.sum(false_neg).float().item()
            total_area = (torch.sum(label_binary).float() + 1e-8).item()  
            if false_pos_area > false_neg_area * 1.5 and false_pos_area > total_area * 0.2:
                error_type = 'over_segmentation'  
            elif false_neg_area > false_pos_area * 1.5 and false_neg_area > total_area * 0.2:
                error_type = 'under_segmentation'  
            elif false_pos_area > total_area * 0.1 and false_neg_area > total_area * 0.1:
                error_type = 'mixed_error'  
            else:
                error_type = 'minor_error'  
            dynamic_threshold = self.difficult_dice_threshold  
            try:
                from torch.nn.functional import max_pool2d
                mask_boundary = (mask_2d != max_pool2d(mask_2d, 3, 1, 1))
                boundary_pixels = torch.sum(mask_boundary).float()
                boundary_ratio = boundary_pixels / (pred_binary.sum() + 1e-8) if pred_binary.sum().item() > 0 else 0
            except Exception as e:
                boundary_ratio = 0
                print(f"[WARNING] Boundary detection failed: {e}")
            mask_area = torch.sum(pred_binary)
            label_area = torch.sum(label_binary)
            area_ratio = mask_area / (label_area + 1e-8) if label_area > 0 else 0
            is_over_segmented = mask_area > label_area * 1.5
            is_under_segmented = mask_area < label_area * 0.5
            wrong_prediction_ratio = false_pos_area / (mask_area + 1e-8) if mask_area > 0 else 0
            is_difficult = dice < dynamic_threshold
            if not is_difficult:
                if boundary_ratio > 0.8 and dice < 0.7:
                    is_difficult = True
                    if test_mode:
                        model_logger.debug(f"[DEBUG] Complex boundary issue - boundary ratio {boundary_ratio:.2f}")
                elif is_over_segmented and wrong_prediction_ratio > 0.3:
                    is_difficult = True
                    if test_mode:
                        model_logger.debug(f"[DEBUG] Severe over-segmentation - wrong prediction ratio {wrong_prediction_ratio:.2f}")
                elif is_under_segmented:
                    is_difficult = True
                    if test_mode:
                        model_logger.debug(f"[DEBUG] Severe under-segmentation - area ratio {area_ratio:.2f}")
            if test_mode:
                model_logger.debug(
                    f"[DEBUG] Difficult sample detection - IoU: {iou.item():.4f}, Dice: {dice.item():.4f}, confidence std: {conf_std.item():.4f}, mask coverage: {mask_coverage.item():.4f}")
                model_logger.debug(f"[DEBUG] Difficult sample detection result: {is_difficult.item()}, error type: {error_type}")
                if is_difficult:
                    model_logger.debug(
                        f"[DEBUG] Details - area ratio: {area_ratio:.2f}, boundary ratio: {boundary_ratio:.2f}, over-segmented: {is_over_segmented}, under-segmented: {is_under_segmented}")
            return is_difficult.item()
        except Exception as e:
            model_logger.debug(f"[DEBUG] Difficult sample detection error: {str(e)}")
            return False
    def compute_uncertainty(self, pred_masks, labels):
        """计算预测掩码的不确定性图
        参数:
            pred_masks: 预测的掩码张量
            labels: 真实标签张量
        返回:
            uncertainty_map: 不确定性图
        """
        try:
            if isinstance(pred_masks, torch.Tensor) and isinstance(labels, torch.Tensor):
                if pred_masks.shape != labels.shape:
                    if len(pred_masks.shape) == 4 and len(labels.shape) == 3:
                        pred_masks = pred_masks.squeeze(1)
                    elif len(pred_masks.shape) == 3 and len(labels.shape) == 4:
                        labels = labels.squeeze(1)
                return torch.logical_xor(pred_masks, labels)
            else:
                pred_tensor = torch.tensor(pred_masks) if not isinstance(pred_masks, torch.Tensor) else pred_masks
                labels_tensor = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
                return torch.logical_xor(pred_tensor, labels_tensor)
        except Exception as e:
            model_logger.debug(f"[DEBUG] Uncertainty calculation error: {str(e)}")
            return torch.zeros_like(pred_masks) if isinstance(pred_masks, torch.Tensor) else torch.tensor(0.0)
    def compute_iou(self, pred_masks, labels):
        """计算预测掩码和真实标签的IoU
        参数:
            pred_masks: 预测的掩码张量
            labels: 真实标签张量
        返回:
            mean_iou: 平均IoU值
        """
        try:
            if isinstance(pred_masks, torch.Tensor) and isinstance(labels, torch.Tensor):
                if pred_masks.shape != labels.shape:
                    if len(pred_masks.shape) == 4 and len(labels.shape) == 3:
                        pred_masks = pred_masks.squeeze(1)
                    elif len(pred_masks.shape) == 3 and len(labels.shape) == 4:
                        labels = labels.squeeze(1)
                pred_binary = (torch.sigmoid(pred_masks) > 0.5) if pred_masks.max().item() > 1 else (pred_masks > 0.5)
                labels_binary = (labels > 0.5) if labels.max().item() > 1 else (labels > 0)
                intersection = torch.logical_and(pred_binary, labels_binary).sum(dim=tuple(range(1, pred_binary.ndim)))
                union = torch.logical_or(pred_binary, labels_binary).sum(dim=tuple(range(1, pred_binary.ndim)))
                iou = intersection.float() / (union.float() + 1e-8)
                return iou.mean()
            else:
                pred_tensor = torch.tensor(pred_masks) if not isinstance(pred_masks, torch.Tensor) else pred_masks
                labels_tensor = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
                return self.compute_iou(pred_tensor, labels_tensor)
        except Exception as e:
            model_logger.debug(f"[DEBUG] IoU calculation error: {str(e)}")
            return torch.tensor(0.0, device=pred_masks.device if isinstance(pred_masks, torch.Tensor) else None)
    def image_forward(self, image):
        img_shape = image.shape
        if len(img_shape) == 3:
            image = image.unsqueeze(0)
            model_logger.debug(f"[DEBUG] 输入维度为 3，添加 batch 维度后变为: {image.shape}")
        elif len(img_shape) != 4:
            raise ValueError(f"输入维度不正确，预期 3 或 4 维，实际为 {len(img_shape)} 维: {img_shape}")
        
        if torch.isnan(image).any() or torch.isinf(image).any():
            model_logger.error("[ERROR] image_forward 输入 image 包含 NaN/Inf！")
            image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=-1.0)
            image = torch.clamp(image, -1.0, 1.0)
        
        image_embedding = self.image_encoder(image)
        
        if torch.isnan(image_embedding).any() or torch.isinf(image_embedding).any():
            model_logger.error("[ERROR] image_encoder 输出包含 NaN/Inf！")
            image_embedding = torch.nan_to_num(image_embedding, nan=0.0, posinf=1.0, neginf=-1.0)
        
        assert len(image_embedding.shape) == 4, f'required shape is (B, C, H, W), but we get {image_embedding.shape}'
        
        embed_shape = image_embedding.shape
        if self.test_mode:
            return_img_embed = image_embedding
        else:
            return_img_embed = image_embedding.view(embed_shape[0], 1, embed_shape[1], embed_shape[2], embed_shape[3])
            return_img_embed = return_img_embed.expand(embed_shape[0], self.select_mask_num, embed_shape[1], embed_shape[2], embed_shape[3])
            return_img_embed = return_img_embed.contiguous().view(-1, embed_shape[1], embed_shape[2], embed_shape[3])
        return return_img_embed
    def forward_decoder(self, image_embedding, prompt):
        
        if isinstance(prompt, dict):
            point_coords_val = prompt.get("point_coords", None)
            point_labels_val = prompt.get("point_labels", None)
        else:
            point_coords_val = None
            point_labels_val = None
        
        if point_coords_val is None:
            points = None
        else:
            points = (point_coords_val, point_labels_val)
        model_logger.debug(f"[DEBUG forward_decoder] image_embedding shape: {image_embedding.shape}, dense_pe shape: {self.prompt_encoder.get_dense_pe().shape}")
        if isinstance(prompt, dict):
            if 'category' not in prompt:
                prompt['category'] = prompt.get('current_category', 'unknown')
            if 'prompt_type' not in prompt:
                prompt['prompt_type'] = self._get_prompt_type(prompt)
            if 'mask_correction' not in prompt:
                prompt['mask_correction'] = torch.zeros(1, device=image_embedding.device)
            if 'dice_change' not in prompt:
                prompt['dice_change'] = 0.0
        is_valid_dict = isinstance(prompt, dict)
        if is_valid_dict and 'interaction_id' in prompt:
            interaction_id = prompt['interaction_id']
            if isinstance(interaction_id, torch.Tensor):
                interaction_id = str(interaction_id.item()) if interaction_id.numel() == 1 else str(id(interaction_id))
            elif not isinstance(interaction_id, str):
                interaction_id = str(interaction_id)
        else:
            interaction_id = 0
        if is_valid_dict and 'interaction_id' in prompt:
            if interaction_id not in self.interaction_history:
                self.interaction_history[interaction_id] = []
            features = image_embedding  
            history_entry = {
                'timestamp': len(self.interaction_history[interaction_id]),
                'prompt_type': self._get_prompt_type(prompt),
                'multi_scale_features': [],
                'spatial_features': features.detach().clone().cpu() if isinstance(features, torch.Tensor) else features  
            }
            if points is not None:
                history_entry['point_coords'] = points[0].detach().cpu() if isinstance(points[0], torch.Tensor) else points[0]
                history_entry['point_labels'] = points[1].detach().cpu() if isinstance(points[1], torch.Tensor) else points[1]
            if hasattr(self, 'multiscale_temporal_fusion') and hasattr(
                    self.multiscale_temporal_fusion.multi_scale_fusion, 'scale_adapters'):
                for extractor in self.multiscale_temporal_fusion.multi_scale_fusion.scale_adapters:
                    scale_feat = extractor(features)
                    history_entry['multi_scale_features'].append(scale_feat.detach().cpu() if isinstance(scale_feat, torch.Tensor) else scale_feat)
            self.interaction_history[interaction_id].append(history_entry)
            max_history = 5
            if len(self.interaction_history[interaction_id]) > max_history:
                self.interaction_history[interaction_id] = self.interaction_history[interaction_id][-max_history:]
        mask_inputs = prompt.get("mask_inputs", None) if isinstance(prompt, dict) else None
        is_valid_dict = isinstance(prompt, dict)
        if is_valid_dict and 'interaction_id' in prompt:
            interaction_id = prompt['interaction_id']
            if isinstance(interaction_id, torch.Tensor):
                interaction_id = str(interaction_id.item()) if interaction_id.numel() == 1 else str(id(interaction_id))
            elif not isinstance(interaction_id, str):
                interaction_id = str(interaction_id)
        else:
            interaction_id = None
        history = []
        if interaction_id is not None and interaction_id in self.interaction_history and len(self.interaction_history[interaction_id]) > 0:
            history = self.interaction_history[interaction_id]
        if history:
            for entry in reversed(history):
                if 'pred_mask' in entry:
                    mask_inputs = entry['pred_mask'].detach().to(image_embedding.device)
                    
                    if mask_inputs.shape[-1] != 256:
                        mask_inputs = F.interpolate(
                            mask_inputs,
                            size=(256, 256),
                            mode='bilinear',
                            align_corners=False
                        )
                    break
        
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=prompt.get("bboxes", None) if isinstance(prompt, dict) else None,
            masks=mask_inputs,
            text=prompt.get("text_inputs", None) if isinstance(prompt, dict) else None,
        )
       
        
       
        if dense_embeddings.shape[2:] != image_embedding.shape[2:]:
            dense_embeddings = F.interpolate(
                dense_embeddings,
                size=image_embedding.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        dense_pe = self.prompt_encoder.get_dense_pe()
        
        outputs = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=dense_pe,
            text_prompt_embeddings=prompt.get("text_inputs", None) if isinstance(prompt, dict) else None,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output,
        )
        if self.multimask_output:
            low_res_masks, iou_pred, semantic_pred = self.get_max_pred(outputs)
        else:
            low_res_masks, iou_pred, semantic_pred = outputs['low_res_masks'], outputs['iou_pred'], outputs[
                'semantic_pred']
        masks = F.interpolate(low_res_masks, size=self.image_size, mode='bilinear', align_corners=False)
        is_valid_dict = isinstance(prompt, dict)
        if is_valid_dict and 'interaction_id' in prompt:
            interaction_id = prompt['interaction_id']
            if isinstance(interaction_id, torch.Tensor):
                interaction_id = str(interaction_id.item()) if interaction_id.numel() == 1 else str(id(interaction_id))
            elif not isinstance(interaction_id, str):
                interaction_id = str(interaction_id)
        else:
            interaction_id = None
        if interaction_id is not None and interaction_id in self.interaction_history and len(self.interaction_history[interaction_id]) > 0:
            latest_entry = self.interaction_history[interaction_id][-1]
            latest_entry['pred_mask'] = low_res_masks.detach().cpu()
        outputs = {
            'masks': masks.float(),
            'low_res_masks': low_res_masks,
            'iou_pred': iou_pred,
            'semantic_pred': semantic_pred,
        }
        return outputs
    def forward(self, image, prompts):
        """增强的前向传播方法，包含困难样本检测和处理"""
        curr_features = self.image_forward(image)
        return self.forward_with_features(curr_features, prompts)
    
    def forward_end_to_end(self, image, interaction_seq, lengths, interaction_id, category, gt_mask=None):
        """完整的端到端前向传播方法
        
        Args:
            image: 输入图像张量
            interaction_seq: 交互序列张量
            lengths: 序列长度列表
            interaction_id: 交互ID
            category: 类别
            gt_mask: 真实掩码（可选）
        
        Returns:
            预测输出
        """
        image_features = self.image_forward(image)
        
        prompts = {
            'patient_id': interaction_id,
            'category': category,
            'interaction_id': interaction_id,
            'temporal_enabled': True,
            'lengths': lengths  
        }
        
        if interaction_seq is not None:
            if isinstance(interaction_seq, list):
                point_coords = []
                point_labels = []
                
                for i, interaction in enumerate(interaction_seq):
                    if lengths and i < lengths[0]:
                        if 'point_coords' in interaction:
                            point_coords.append(interaction['point_coords'])
                        if 'point_labels' in interaction:
                            point_labels.append(interaction['point_labels'])
                
                if point_coords:
                    prompts['point_coords'] = torch.cat(point_coords, dim=0)
                    prompts['point_labels'] = torch.cat(point_labels, dim=0)
            elif isinstance(interaction_seq, torch.Tensor):
                pass
        
        if gt_mask is not None:
            prompts['labels'] = gt_mask
        
        outputs = self.forward(image, prompts)
        
        return outputs
    
    def forward_with_features(self, curr_features, prompts):
        """使用预提取的图像特征进行前向传播（支持 Batch 版）"""
        if torch.isnan(curr_features).any() or torch.isinf(curr_features).any():
            model_logger.error("[ERROR] forward_with_features 输入 curr_features 包含 NaN/Inf！")
            curr_features = torch.nan_to_num(curr_features, nan=0.0, posinf=1.0, neginf=-1.0)
            curr_features = torch.clamp(curr_features, -1.0, 1.0)
        
        patient_ids = prompts.get('patient_ids', [])
        categories = prompts.get('categories', [])
        
        batch_size = curr_features.shape[0]
        
        if len(patient_ids) != batch_size:
            raise RuntimeError(f"patient_ids长度{len(patient_ids)}与batch_size{batch_size}不匹配！请修复data_loader.py")
        if len(categories) != batch_size:
            raise RuntimeError(f"categories长度{len(categories)}与batch_size{batch_size}不匹配！请修复data_loader.py")
        fused_features_list = []
        for i in range(curr_features.shape[0]):
            single_curr_feat = curr_features[i:i+1]  
            
            if torch.isnan(single_curr_feat).any() or torch.isinf(single_curr_feat).any():
                model_logger.error(f"[ERROR] 样本 {i} 的特征包含 NaN/Inf！")
                single_curr_feat = torch.nan_to_num(single_curr_feat, nan=0.0, posinf=1.0, neginf=-1.0)
                single_curr_feat = torch.clamp(single_curr_feat, -1.0, 1.0)
            
            single_patient_id = patient_ids[i]
            single_category = categories[i]
            history = self.temporal_memory.get_history(single_patient_id, single_category) if hasattr(self, 'temporal_memory') else []
            
            single_prompt = {}
            for key, value in prompts.items():
                if key not in ['point_coords', 'point_labels']:
                    if isinstance(value, torch.Tensor):
                        if value.numel() == 1:
                            single_prompt[key] = value.item()
                        elif i < value.numel():
                            single_prompt[key] = value[i].item() if value[i].numel() == 1 else value[i].float().mean().item()
                        else:
                            single_prompt[key] = value.mean().item()
                    elif isinstance(value, (list, tuple)) and i < len(value):
                        single_prompt[key] = value[i]
                    else:
                        single_prompt[key] = value
            single_prompt['patient_id'] = single_patient_id
            single_prompt['category'] = single_category
            
            if isinstance(prompts, dict) and 'point_coords' in prompts and 'point_labels' in prompts:
                single_prompt['point_coords'] = prompts['point_coords'][i:i+1]
                single_prompt['point_labels'] = prompts['point_labels'][i:i+1]
            
            if isinstance(prompts, dict):
                if 'dice' in prompts:
                    dice_val = prompts['dice']
                    if isinstance(dice_val, (list, tuple)) and len(dice_val) > i:
                        single_prompt['dice'] = dice_val[i]
                    elif isinstance(dice_val, torch.Tensor) and dice_val.numel() > i:
                        single_prompt['dice'] = dice_val[i].item() if dice_val[i].numel() == 1 else dice_val[i].float().mean().item()
                if 'tci' in prompts:
                    tci_val = prompts['tci']
                    if isinstance(tci_val, (list, tuple)) and len(tci_val) > i:
                        single_prompt['tci'] = tci_val[i]
                    elif isinstance(tci_val, torch.Tensor) and tci_val.numel() > i:
                        single_prompt['tci'] = tci_val[i].item() if tci_val[i].numel() == 1 else tci_val[i].float().mean().item()
            
            single_fused_feat = single_curr_feat
            if hasattr(self, 'multiscale_temporal_fusion'):
                global_epoch = single_prompt.get('global_epoch', 0)
                
                fusion_history = []
                for entry in history:
                    if isinstance(entry, dict) and 'prompt' in entry:
                        fusion_history.append(entry['prompt'])
                    else:
                        fusion_history.append(entry)
                
                fusion_history = fusion_history[-3:]  
                
                fusion_enabled = (global_epoch >= 10)
                
                warmup_epochs = getattr(self, 'fusion_warmup_epochs', 20)  
                max_fusion_weight = getattr(self, 'max_fusion_strength', 1.5)  
                
                if not fusion_enabled:
                    fusion_strength = 0.0
                elif global_epoch < 10 + warmup_epochs:
                    progress = (global_epoch - 10) / warmup_epochs
                    fusion_strength = progress * max_fusion_weight   
                else:
                    fusion_strength = max_fusion_weight
                
                temporal_enhancement = self.multiscale_temporal_fusion(
                    current_prompt=single_prompt,
                    history=fusion_history,
                    image_features=single_curr_feat
                )
                
                gate_weight = torch.sigmoid(self.multiscale_temporal_fusion.trajectory_fusion_weight)
                single_fused_feat = single_curr_feat + fusion_strength * gate_weight * temporal_enhancement
                
                model_logger.debug(f"[DEBUG] Epoch {global_epoch}, fusion_strength: {fusion_strength}, fusion_enabled: {fusion_enabled}")
            history_current = self.temporal_memory.get_history(single_patient_id, single_category) if hasattr(self, 'temporal_memory') else []
            model_logger.debug(f"[DEBUG] 交互轮次: {single_prompt.get('interaction_round', 0)}, 当前类别库长度: {len(history_current)}, 总库长度: {len(history)}")
            
            fused_features_list.append(single_fused_feat)
        
        fused_features = torch.cat(fused_features_list, dim=0)
        
        if isinstance(prompts, dict) and 'patient_id' not in prompts and patient_ids:
            prompts['patient_id'] = patient_ids[0]
            
        outputs = self.forward_decoder(fused_features, prompts)
        
        total_difficult = 0
        successfully_improved = 0
        total_improvement = 0.0
        binary_masks = None
        
        if isinstance(prompts, dict) and 'labels' in prompts and prompts['labels'] is not None and self.test_mode and not self.training:
            model_logger.debug(f"[DEBUG] Forward method call: test_mode={self.test_mode}, labels={prompts['labels'] is not None}")
            if not isinstance(outputs, dict):
                outputs = {'masks': outputs}
            pred_masks = outputs['masks']
            batch_size = pred_masks.shape[0]
            binary_masks = torch.zeros_like(pred_masks)
            for i in range(batch_size):
                single_pred = pred_masks[i:i + 1]
                single_label = prompts['labels'][i:i + 1]
                pred_binary = (torch.sigmoid(single_pred) > 0.5).float()
                label_binary = (single_label > 0).float()
                intersection = (pred_binary.squeeze() * label_binary.squeeze()).sum()
                orig_dice = (2 * intersection) / (pred_binary.squeeze().sum() + label_binary.squeeze().sum() + 1e-8)
                orig_iou = intersection / (
                            pred_binary.squeeze().sum() + label_binary.squeeze().sum() - intersection + 1e-8)
                if orig_dice < self.postprocess_trigger_threshold:
                    total_difficult += 1
                    model_logger.debug(f"[DEBUG] Difficult sample #{i}, orig_dice={orig_dice:.4f}, orig_iou={orig_iou:.4f}") 
                    processed_mask = self.postprocess_difficult_samples(single_pred, single_label)
                    processed_binary = processed_mask
                    post_intersection = (processed_binary.squeeze() * label_binary.squeeze()).sum()
                    post_dice = (2 * post_intersection) / (
                                processed_binary.squeeze().sum() + label_binary.squeeze().sum() + 1e-8)
                    post_iou = post_intersection / (
                                processed_binary.squeeze().sum() + label_binary.squeeze().sum() - post_intersection + 1e-8)
                    improvement = (post_dice - orig_dice) / orig_dice * 100 if orig_dice > 0 else 0
                    if post_dice > orig_dice:
                        successfully_improved += 1
                        total_improvement += improvement.item()
                        print(
                            f"[SUCCESS] 困难样本 #{i} 后处理有效: dice从{orig_dice:.4f}提升到{post_dice:.4f}, 改善{improvement:.2f}%"
                        )
                    else:
                        print(
                            f"[INFO] 困难样本 #{i} 后处理无效: dice从{orig_dice:.4f}变为{post_dice:.4f}"
                        ) 
                binary_masks[i] = processed_mask
            else:
                binary_masks[i] = self.binary_mask_with_threshold(single_pred, 0.5)
        avg_improvement = total_improvement / successfully_improved if successfully_improved > 0 else 0
        model_logger.debug(f"[DEBUG] Total {total_difficult} difficult samples detected in forward")
        model_logger.debug(f"[DEBUG] Successfully improved {successfully_improved} difficult samples, average improvement {avg_improvement:.2f}%")
        if binary_masks is not None:
            outputs['binary_masks'] = binary_masks
        assert not (self.training and 'binary_masks' in outputs), "训练时不能生成二值掩码！"
        return outputs
    def postprocess_difficult_samples(self, pred_mask, label_mask=None, quality_score=None):
        """Difficult-sample post-processing without using GT label info.
        Pipeline:
          1) sigmoid -> prob map
          2) threshold 0.5 (or quality-based: 0.6 if quality_score > 0.8 else 0.5)
          3) morphological open+close (kernel=3) + remove connected components < k=20 pixels
        Args:
            pred_mask:     predicted logits, shape [1, C, H, W] or [B, C, H, W]
            label_mask:    kept for signature compatibility; NOT used internally
            quality_score: optional model-estimated quality in [0,1] for thresholding
        Returns:
            binary mask {0, 1} float tensor with the same shape as input
        """
        device = pred_mask.device
        original_shape = pred_mask.shape
        model_logger.debug(f"[DEBUG postprocess_difficult_samples_v2] input shape: {original_shape}")
        pred_prob = torch.sigmoid(pred_mask)
        if quality_score is not None:
            if isinstance(quality_score, torch.Tensor):
                q = quality_score.mean().item() if quality_score.numel() > 1 else quality_score.item()
            else:
                try:
                    q = float(quality_score)
                except Exception:
                    q = 0.5
            threshold = 0.6 if q > 0.8 else 0.5
        else:
            threshold = 0.5
        binary_mask = (pred_prob > threshold).float()
        orig_area = torch.sum(binary_mask).item()
        try:
            k3 = torch.ones((1, 1, 3, 3), device=device)
            work = binary_mask
            while work.ndim < 4:
                work = work.unsqueeze(0)
            erode_result = F.conv2d(work, k3, padding=1, groups=work.shape[1])
            eroded = (erode_result == 9.0).float()
            dilate_result = F.conv2d(eroded, k3, padding=1, groups=work.shape[1])
            opened = (dilate_result > 0).float()
            d2 = F.conv2d(opened, k3, padding=1, groups=work.shape[1])
            d2 = (d2 > 0).float()
            e2 = F.conv2d(d2, k3, padding=1, groups=work.shape[1])
            closed = (e2 == 9.0).float()
            binary_mask = closed
            while binary_mask.ndim > original_shape.ndim:
                binary_mask = binary_mask.squeeze(0)
            if binary_mask.shape != original_shape:
                binary_mask = F.interpolate(
                    binary_mask, size=original_shape[2:], mode='nearest'
                )
        except Exception:
            binary_mask = (pred_prob > threshold).float()
        try:
            from scipy import ndimage as _ndi
            import numpy as _np
            b_np = binary_mask.squeeze().detach().cpu().numpy()
            if b_np.ndim == 2:
                lab, nf = _ndi.label(b_np > 0.5)
                if nf > 0:
                    sizes = _ndi.sum(b_np, lab, range(1, nf + 1))
                    keep = _np.isin(lab, _np.where(sizes > 20)[0] + 1)
                    b_np = keep.astype(_np.float32)
                else:
                    b_np = b_np.astype(_np.float32)
                cleaned = torch.tensor(b_np, device=device, dtype=torch.float32)
                while cleaned.ndim < 4:
                    cleaned = cleaned.unsqueeze(0)
                if cleaned.shape != original_shape:
                    cleaned = cleaned.reshape(
                        min(cleaned.shape[0], original_shape[0]),
                        min(cleaned.shape[1], original_shape[1]),
                        cleaned.shape[2], cleaned.shape[3]
                    )
                binary_mask = cleaned
        except ImportError:
            pass
        binary_mask = (binary_mask > 0).float()
        if binary_mask.shape != original_shape:
            binary_mask = F.interpolate(binary_mask, size=original_shape[2:], mode='nearest')
        new_area = torch.sum(binary_mask).item()
        area_change_pct = (new_area - orig_area) / max(orig_area, 1) * 100.0
        model_logger.debug(
            f"[DEBUG postprocess_difficult_samples_v2] threshold={threshold}, "
            f"pixels {orig_area:.0f}->{new_area:.0f} ({area_change_pct:+.1f}%)"
            f" (NO GT label used)"
        )
        return binary_mask.float()
    def _apply_morphological_op(self, mask, kernel, operation):
        """应用形态学操作"""
        if operation == 'erode':
            result = F.conv2d(mask, kernel, padding=kernel.size(2) // 2, groups=1)
            return (result == kernel.size(2) * kernel.size(3)).float()
        elif operation == 'dilate':
            result = F.conv2d(mask, kernel, padding=kernel.size(2) // 2, groups=1)
            return (result > 0).float()
        elif operation == 'close':
            dilated = F.conv2d(mask, kernel, padding=kernel.size(2) // 2, groups=1)
            dilated = (dilated > 0).float()
            eroded = F.conv2d(dilated, kernel, padding=kernel.size(2) // 2, groups=1)
            return (eroded == kernel.size(2) * kernel.size(3)).float()
        elif operation == 'open':
            eroded = F.conv2d(mask, kernel, padding=kernel.size(2) // 2, groups=1)
            eroded = (eroded == kernel.size(2) * kernel.size(3)).float()
            dilated = F.conv2d(eroded, kernel, padding=kernel.size(2) // 2, groups=1)
            return (dilated > 0).float()
        return mask
    def _keep_major_connected_regions(self, mask, top_k=3):
        """保留最大的几个连通区域"""
        device = mask.device
        batch_size, channels, height, width = mask.shape
        result = torch.zeros_like(mask)
        for b in range(batch_size):
            for c in range(channels):
                current_mask = mask[b, c]
                if current_mask.sum().item() == 0:
                    continue
                kernel = torch.ones((1, 1, 3, 3), device=device)
                dilated = F.conv2d(current_mask.unsqueeze(0).unsqueeze(0),
                                 kernel, padding=1, groups=1)
                dilated = (dilated > 0).float()
                result[b, c] = dilated.squeeze()
        return result
    def _fine_tune_mask_area(self, mask, target_area, pred_prob):
        """优化的掩码面积微调方法 - 更精确地匹配目标面积"""
        if len(mask.shape) > 2:
            mask = mask.squeeze(0).squeeze(0)
        if len(pred_prob.shape) > 2:
            pred_prob = pred_prob.squeeze(0).squeeze(0)
        device = mask.device
        current_area = torch.sum(mask)
        area_diff = target_area - current_area
        area_ratio = current_area / target_area if target_area > 0 else 0
        if target_area > 0 and abs(area_diff) / target_area < 0.1:  
            model_logger.debug(f"[DEBUG] Area fine-tuning: Current area is already close to target area ({current_area:.0f} vs {target_area:.0f})")
            return mask
        model_logger.debug(
            f"[DEBUG] Area fine-tuning: Current area {current_area:.0f}, target area {target_area:.0f}, difference {area_diff:+.0f} ({area_ratio:.2f}x)")
        tuned_mask = mask.clone()
        if area_diff > 0:  
            available_prob = pred_prob.clone()
            available_prob[mask > 0] = 0  
            threshold_steps = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
            added_pixels = 0
            target_add = int(area_diff)
            for thresh in threshold_steps:
                if added_pixels >= target_add:  
                    break
                candidate_pixels = (available_prob > thresh).nonzero(as_tuple=False)
                if len(candidate_pixels) > 0:
                    add_count = min(len(candidate_pixels), target_add - added_pixels)
                    prob_values = available_prob[tuple(candidate_pixels.T)]
                    sorted_indices = torch.argsort(prob_values, descending=True)
                    selected_pixels = candidate_pixels[sorted_indices[:add_count]]
                    for pixel in selected_pixels:
                        tuned_mask[tuple(pixel)] = 1.0
                    added_pixels += add_count
                    model_logger.debug(f"[DEBUG] Area fine-tuning: Threshold {thresh} added {add_count} pixels")
            if added_pixels < target_add and (available_prob > 0).any():
                remaining_add = target_add - added_pixels
                flat_probs = available_prob.flatten()
                top_values, top_indices = torch.topk(flat_probs, min(remaining_add, len(flat_probs)))
                valid_indices = top_indices[top_values > 0.1]
                if len(valid_indices) > 0:
                    unraveled = torch.unravel_index(valid_indices, available_prob.shape)
                    tuned_mask[unraveled] = 1.0
                    added_pixels += len(valid_indices)
                    model_logger.debug(f"[DEBUG] Area fine-tuning: Added {len(valid_indices)} additional pixels")
        else:  
            selected_prob = pred_prob.clone()
            selected_prob[mask == 0] = 1.0  
            threshold_steps = [0.1, 0.2, 0.3, 0.4, 0.5]
            removed_pixels = 0
            target_remove = int(-area_diff)
            for thresh in threshold_steps:
                if removed_pixels >= target_remove:  
                    break
                candidate_pixels = (selected_prob < thresh).nonzero(as_tuple=False)
                if len(candidate_pixels) > 0:
                    remove_count = min(len(candidate_pixels), target_remove - removed_pixels)
                    prob_values = selected_prob[tuple(candidate_pixels.T)]
                    sorted_indices = torch.argsort(prob_values, descending=False)
                    selected_pixels = candidate_pixels[sorted_indices[:remove_count]]
                    for pixel in selected_pixels:
                        tuned_mask[tuple(pixel)] = 0.0
                    removed_pixels += remove_count
                    model_logger.debug(f"[DEBUG] Area fine-tuning: Threshold {thresh} removed {remove_count} pixels")
            if removed_pixels < target_remove and (selected_prob < 1.0).any():
                remaining_remove = target_remove - removed_pixels
                flat_probs = selected_prob.flatten()
                bottom_values, bottom_indices = torch.topk(flat_probs, min(remaining_remove, len(flat_probs)),
                                                           largest=False)
                valid_indices = bottom_indices[bottom_values < 1.0]
                if len(valid_indices) > 0:
                    unraveled = torch.unravel_index(valid_indices, selected_prob.shape)
                    tuned_mask[unraveled] = 0.0
                    removed_pixels += len(valid_indices)
                    model_logger.debug(f"[DEBUG] Area fine-tuning: Removed {len(valid_indices)} additional pixels")
        final_area = torch.sum(tuned_mask)
        model_logger.debug(f"[DEBUG] Area fine-tuning completed: Original {current_area:.0f}, target {target_area:.0f}, result {final_area:.0f}")
        return tuned_mask.float()
    def _compute_mask_quality(self, mask, labels=None):
        """优化的掩码质量评估方法 - 专为低Dice样本后处理优化"""
        if mask.max().item() > 1:
            mask = (mask > 0.5).float()
        if len(mask.shape) > 2:
            mask = mask.squeeze(0).squeeze(0)  
        device = mask.device
        if labels is not None and not (labels == 0).all().item():
            if labels.max().item() > 1:
                labels = (labels > 0.5).float()
            if len(labels.shape) > 2:
                labels = labels.squeeze(0).squeeze(0)
            intersection = (mask * labels).sum()
            mask_sum = mask.sum()
            label_sum = labels.sum()
            dice_score = (2 * intersection) / (mask_sum + label_sum + 1e-8) if (mask_sum + label_sum) > 0 else 0.0
            union = mask_sum + label_sum - intersection
            iou_score = intersection / (union + 1e-8) if union > 0 else 0.0
            precision = intersection / (mask_sum + 1e-8) if mask_sum > 0 else 0.0
            recall = intersection / (label_sum + 1e-8) if label_sum > 0 else 0.0
            true_positives = intersection
            false_positives = mask_sum - intersection
            false_negatives = label_sum - intersection
            total_pixels = torch.prod(torch.tensor(mask.shape, device=device))
            true_negatives = total_pixels - (true_positives + false_positives + false_negatives)
            if dice_score < 0.5:
                balance_factor = 0.6  
            elif dice_score < 0.7:
                balance_factor = 0.5
            else:
                balance_factor = 0.4
            balanced_f1 = (1 + 0.5) * precision * recall / (0.5 * precision + recall + 1e-8) if (
                                                                                                            precision + recall) > 0 else 0.0
            area_similarity = 1.0
            if label_sum > 0 and mask_sum > 0:
                area_ratio = mask_sum / label_sum
                area_similarity = 1.0 - min(0.5, abs(torch.log(area_ratio)) * 0.5)
            position_score = 0.0
            if true_positives > 0:
                correct_ratio = true_positives / mask_sum if mask_sum > 0 else 0.0
                detected_ratio = true_positives / label_sum if label_sum > 0 else 0.0
                position_score = (correct_ratio + detected_ratio) / 2.0
            connectivity_score = 0.0
            area = mask_sum
            if area > 0:
                from torch.nn.functional import conv2d
                try:
                    kernel = torch.ones((1, 1, 3, 3), device=device)
                    conv_result = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding=1)
                    connected_ratio = (conv_result > 4).sum() / area
                    connectivity_score = connected_ratio.item() if not torch.isnan(connected_ratio).any().item() else 0.7
                except:
                    connectivity_score = 0.7
            if dice_score < 0.4:
                final_score = (0.35 * dice_score + 0.25 * iou_score +
                               0.2 * balanced_f1 + 0.1 * area_similarity +
                               0.05 * position_score + 0.05 * connectivity_score)
            else:
                final_score = (0.3 * dice_score + 0.25 * iou_score +
                               0.2 * balanced_f1 + 0.1 * area_similarity +
                               0.05 * position_score + 0.05 * connectivity_score)
            return final_score.item() if not torch.isnan(final_score) else 0.0
        else:
            try:
                area = mask.sum()
                total_pixels = torch.prod(torch.tensor(mask.shape, device=device))
                coverage_ratio = area / total_pixels
                if 0.01 < coverage_ratio < 0.95:
                    coverage_score = 1.0 - min(0.5, abs(coverage_ratio - 0.15) * 3)  
                else:
                    coverage_score = 0.5
                connectivity_score = 0.7  
                if area > 0:
                    try:
                        kernel = torch.ones((1, 1, 3, 3), device=device)
                        conv_result = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding=1)
                        connected_ratio = (conv_result > 4).sum() / area
                        connectivity_score = connected_ratio.item() if not torch.isnan(connected_ratio).any().item() else 0.7
                    except:
                        pass
                shape_score = 0.7  
                if area > 0:
                    try:
                        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                              dtype=torch.float32, device=device).view(1, 1, 3, 3)
                        boundary = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding=1)
                        perimeter = torch.sum(torch.abs(boundary) > 0)
                        if perimeter > 0:
                            compactness = (perimeter ** 2) / (4 * torch.pi * area)
                            shape_score = max(0.0, 1.0 - min(1.0, compactness / 15))
                    except:
                        pass
                combined_score = 0.3 * coverage_score + 0.4 * connectivity_score + 0.3 * shape_score
                return combined_score.item() if not torch.isnan(combined_score) else 0.0
            except Exception as e:
                model_logger.debug(f"[DEBUG] Label-free mask quality evaluation error: {str(e)}")
                try:
                    total_pixels = mask.shape[0] * mask.shape[1] if len(mask.shape) >= 2 else 1
                    return float(torch.sum(mask)) / total_pixels
                except:
                    return 0.0
    def _clean_small_regions(self, mask, min_size_ratio=0.01, min_size=None):
        """
        清理掩码中的小区域
        使用卷积操作近似连通区域分析
        参数:
            mask: 输入掩码
            min_size_ratio: 相对于图像总像素的最小区域比例
            min_size: 直接指定的最小区域大小（优先使用此参数）
        """
        try:
            device = mask.device
            total_pixels = mask.shape[2] * mask.shape[3]
            if min_size is not None:
                min_size_value = min_size
            else:
                min_size_value = max(10, int(total_pixels * min_size_ratio))
            kernel_size = max(3, int(min_size_value ** 0.5))
            kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device)
            neighbor_count = F.conv2d(mask, kernel, padding=kernel_size // 2, groups=1)
            cleaned_mask = (neighbor_count >= min_size_value // 2).float() * mask
            return cleaned_mask
        except Exception as e:
            model_logger.debug(f"[DEBUG] Region cleaning error: {str(e)}")
            return mask
    
    def reset_interaction(self, interaction_id=None):
        """重置交互状态，清除特定或全部历史记录"""
        if hasattr(self, 'interaction_history'):
            if interaction_id is None:
                self.interaction_history.clear()
            else:
                if isinstance(interaction_id, torch.Tensor):
                    interaction_id = str(interaction_id.item()) if interaction_id.numel() == 1 else str(id(interaction_id))
                elif not isinstance(interaction_id, str):
                    interaction_id = str(interaction_id)
                
                if interaction_id in self.interaction_history:
                    del self.interaction_history[interaction_id]
        if hasattr(self, 'multiscale_temporal_fusion'):
            self.multiscale_temporal_fusion.temporal_memory.clear_history(interaction_id)
        if hasattr(self, 'interaction_step'):
            self.interaction_step = 0
    def _adaptive_history_length(self, current_length):
        """自适应历史长度"""
        if current_length < 3:
            return 3  
        elif current_length < 6:
            return 5  
        else:
            return 8  
    def _get_prompt_type(self, prompt):
        """获取提示类型，用于分析交互模式
        参数:
            prompt: 包含各种提示类型的字典
        返回:
            prompt_type: 提示类型字符串
        """
        if isinstance(prompt, dict):
            if 'point_coords' in prompt:
                return 'points'
            elif 'bboxes' in prompt:
                return 'bboxes'
            elif 'text_inputs' in prompt:
                return 'text'
            elif 'mask_inputs' in prompt:
                return 'mask'
        return 'unknown'
    def _select_informative_history(self, history_entries, max_history):
        """选择信息量最大的历史记录"""
        if len(history_entries) <= max_history:
            return history_entries
        scores = []
        for i, entry in enumerate(history_entries):
            score = 1.0
            time_factor = 1.0 - (i / len(history_entries)) * 0.5
            point_factor = 1.5 if 'point_coords' in entry and entry['point_coords'] is not None else 1.0
            feature_factor = 1.2 if len(entry.get('multi_scale_features', [])) > 0 else 1.0
            final_score = score * time_factor * point_factor * feature_factor
            scores.append(final_score)
        indices = torch.topk(torch.tensor(scores), max_history).indices.tolist()
        indices.sort()
        selected_entries = [history_entries[i] for i in indices]
        return selected_entries
    def get_multiscale_features(self, image_embedding):
        """获取图像的多尺度特征 - 完整版本：使用多尺度融合网络提取特征"""
        if hasattr(self, 'multiscale_temporal_fusion'):
            return self.multiscale_temporal_fusion.multi_scale_fusion.extract_multi_scale_features(image_embedding)
        return [image_embedding]
    def _base_prompts(self, masks, pred_masks, low_res_masks, specify_prompt, classes=None, is_supervised=True):
        """
        基础提示生成方法，包含supervised_prompts和unsupervised_prompts的共同逻辑
        参数:
            masks: 标签掩码（监督学习）或伪标签掩码（无监督学习）
            pred_masks: 预测掩码
            low_res_masks: 低分辨率掩码
            specify_prompt: 提示类型
            classes: 类别信息（仅监督学习需要）
            is_supervised: 是否为监督学习模式
        返回:
            bs_prompts: 提示字典
        """
        bs_prompts = {}
        if low_res_masks is not None:
            bs_prompts.update(self.process_mask_prompt(low_res_masks))
        if specify_prompt == 'points':
            bs = masks.shape[0]
            bs_prompts.update(self.process_points_prompt(bs, masks, pred_masks))
        elif specify_prompt == 'text' and is_supervised and classes is not None:
            bs_prompts.update(self.process_text_prompt(classes))
        elif specify_prompt == 'bboxes':
            bs = masks.shape[0]
            bs_prompts.update(self.process_bboxes_prompt(bs, masks))
        assert len(bs_prompts) > 0, f'prompt error: {bs_prompts}'
        return bs_prompts
    def supervised_prompts(self, classes, labels, pred_masks, low_res_masks, specify_prompt):
        """生成监督学习的提示"""
        return self._base_prompts(labels, pred_masks, low_res_masks, specify_prompt, classes, is_supervised=True)
    def unsupervised_prompts(self, pseudo_labels, pred_masks, low_res_masks, specify_prompt):
        """生成无监督学习的提示"""
        return self._base_prompts(pseudo_labels, pred_masks, low_res_masks, specify_prompt, is_supervised=False)
    def process_bboxes_prompt(self, bs, labels):
        bs_bboxes = [get_bboxes_from_mask(labels[idx]) for idx in range(bs)]
        return {'bboxes': torch.stack(bs_bboxes, dim=0).to(self.device)}
    def process_points_prompt(self, bs, labels, pred_masks=None, interaction_id=None):
        if self.test_mode:
            point_num = 3  
        else:
            point_num = random.choice([1, 3, 4, 7])
        if pred_masks is not None:
            pred_masks = torch.sigmoid(pred_masks)
            pred_masks = (pred_masks > 0.5).bool().squeeze(1)
        labels = labels.bool().squeeze(1)
        
        error_area = None
        if pred_masks is not None:
            error_area = pred_masks ^ labels
        guided_points = None
        existing_points = []  
        if interaction_id is not None and hasattr(self, 'interaction_history'):
            safe_interaction_id = interaction_id
            if isinstance(interaction_id, torch.Tensor):
                safe_interaction_id = str(interaction_id.item()) if interaction_id.numel() == 1 else str(id(interaction_id))
            elif not isinstance(interaction_id, str):
                safe_interaction_id = str(interaction_id)
            
            if safe_interaction_id in self.interaction_history:
                for hist_item in self.interaction_history[safe_interaction_id]:
                    if 'point_coords' in hist_item and 'point_labels' in hist_item:
                        coords = hist_item['point_coords']
                        labels = hist_item['point_labels']
                        for i in range(coords.shape[1]):  
                            if i < coords.shape[1] and i < labels.shape[1]:  
                                existing_points.append((coords[0, i].cpu().numpy(), labels[0, i].cpu().numpy()))
        if pred_masks is not None and hasattr(self, 'interaction_guide'):
            guided_points = self.interaction_guide.recommend_points(
                current_mask=pred_masks,
                uncertainty_map=self.compute_uncertainty(pred_masks, labels),
                existing_points=existing_points  
            )
            if guided_points and guided_points[0][0].shape[0] > 0:
                point_num = max(1, guided_points[0][0].shape[0])
        bs_point_coords = torch.empty((bs, point_num, 2), dtype=torch.float32, device=labels.device)
        bs_point_labels = torch.empty((bs, point_num), dtype=torch.long, device=labels.device)
        for idx in range(bs):
            if pred_masks is None:
                point_coords, point_labels = get_points_from_mask(labels[idx], get_point=point_num)
            else:
                if guided_points and idx < len(guided_points) and guided_points[idx][0].shape[0] > 0:
                    point_coords, point_labels = guided_points[idx]
                else:
                    point_coords, point_labels = self.get_points_from_interaction(
                        error_area[idx], pred_masks[idx], labels[idx], get_point=point_num
                    )
            if len(point_coords) < point_num:
                if len(point_coords) > 0:
                    repeats = point_num - len(point_coords)
                    indices = np.random.choice(len(point_coords), repeats, replace=True)
                    point_coords = np.vstack([point_coords, point_coords[indices]])
                    point_labels = np.concatenate([point_labels, point_labels[indices]])
                else:
                    point_coords, point_labels = get_points_from_mask(labels[idx], get_point=point_num)
            elif len(point_coords) > point_num:
                point_coords = point_coords[:point_num]
                point_labels = point_labels[:point_num]
            if not self.test_mode and self.training:
                noise = np.random.normal(0, 0.5, point_coords.shape)
                point_coords = point_coords + noise
                point_coords = np.clip(point_coords, 0, labels.shape[0]-1)
            point_coords = point_coords.astype(np.float32)
            point_coords = np.clip(point_coords, 0, labels.shape[0]-1)
            bs_point_coords[idx, :] = torch.as_tensor(point_coords, device=labels.device)
            bs_point_labels[idx, :] = torch.as_tensor(point_labels, device=labels.device)
        return {
            'point_coords': bs_point_coords,
            'point_labels': bs_point_labels
        }
    
    
    def process_mask_prompt(self, low_res_masks):
        low_res_masks_logist = low_res_masks.detach().clone()
        return {'mask_inputs': low_res_masks_logist.to(self.device)}
    def load_category_weights(self, src_weights=None):
        if src_weights is not None:
            with open(src_weights, "rb") as f:
                self.src_weights, self.categories_map, self.category_to_index, self.index_to_category = pickle.load(f)
                self.src_weights = torch.tensor(self.src_weights).to(self.device)
    def category_labels(self, classes):
        norm_target = []
        for clas in classes:
            if clas in self.categories_map:
                clas = self.categories_map[clas][1]
            else:
                print(f"Warning: '{clas}' not found in categories_map, using original text")
            category = clas.lower().replace('_', ' ').replace("-", " ")
            category = category.replace('left', '').replace('right', '').strip()
            category = re.sub(r'\s+', ' ', category)
            norm_target.append(category)
        
        indices = []
        for clas in norm_target:
            if clas in self.category_to_index:
                indices.append(self.category_to_index[clas])
            else:
                print(f"Warning: '{clas}' not found in category_to_index, using default index 0")
                indices.append(0)
        
        return torch.tensor(indices).unsqueeze(-1).to(self.device)
    def category_loss(self, semantic_preds, classes, ce_loss):
        labels = self.category_labels(classes)
        logits = nn.functional.normalize(semantic_preds, dim=-1) @ self.src_weights
        probs = nn.functional.softmax(logits, dim=-1)
        loss = ce_loss(probs.squeeze(1), labels.squeeze(1))
        return loss, probs
    def get_max_pred(self, outputs):
        low_res_masks, iou_pred, semantic_pred = outputs['low_res_masks'], outputs['iou_pred'], outputs['semantic_pred']
        max_values, max_indices = torch.max(iou_pred, dim=1, keepdim=True)
        low_mask_indices = max_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, low_res_masks.shape[2],
                                                                          low_res_masks.shape[3])
        semantic_idices = max_indices.unsqueeze(-1).expand(-1, -1, 512)
        low_res_masks_selected = torch.gather(low_res_masks, 1, low_mask_indices)
        semantic_selected = torch.gather(semantic_pred, 1, semantic_idices)
        return low_res_masks_selected, max_values, semantic_selected
    def get_points_from_interaction(self, error, pr, gt, get_point=3):
        """
        大师级点选择策略：形态学引导 + 连通域优先
        """
        import numpy as np
        from scipy.ndimage import distance_transform_edt, label
        
        pred, gt = pr.data.cpu().numpy(), gt.data.cpu().numpy()
        
        if error is None:
            error_mask = (pred != gt).astype(np.float32)
        else:
            error_mask = error.cpu().numpy().astype(np.float32)
        
        if np.sum(error_mask) == 0:
            pred_center = np.mean(np.argwhere(pred == 1), axis=0) if np.sum(pred) > 0 else np.array([128, 128])
            gt_center = np.mean(np.argwhere(gt == 1), axis=0) if np.sum(gt) > 0 else np.array([128, 128])
            
            center = np.round((pred_center + gt_center) / 2).astype(int)
            x = np.clip(center[1], 0, pred.shape[1] - 1)  
            y = np.clip(center[0], 0, pred.shape[0] - 1)  
            selected_indices = np.array([[x, y]])  
            
            if get_point > 1:
                additional_points = []
                center_mask = np.zeros_like(error_mask)
                center_mask[center[0], center[1]] = 1
                dist_map = cv2.distanceTransform(error_mask.astype(np.uint8), cv2.DIST_L2, 5)
                if error_mask.sum().item() > 0:
                    _, max_val, _, _ = cv2.minMaxLoc(dist_map)
                    dynamic_radius = min(20, max_val * 0.5)
                else:
                    dynamic_radius = 10  
                
                for i in range(1, get_point):
                    angle = 2 * np.pi * i / get_point
                    radius = dynamic_radius  
                    new_y = int(center[0] + radius * np.sin(angle))
                    new_x = int(center[1] + radius * np.cos(angle))
                    new_x = np.clip(new_x, 0, pred.shape[1]-1)  
                    new_y = np.clip(new_y, 0, pred.shape[0]-1)  
                    additional_points.append([new_x, new_y])  
                selected_indices = np.vstack([selected_indices, np.array(additional_points)])
        else:
            labeled_array, num_features = label(error_mask)
            if num_features == 0:
                indices = np.argwhere(error_mask == 1)
                center_y = int(np.mean(indices[:, 0]))  
                center_x = int(np.mean(indices[:, 1]))  
                x = np.clip(center_x, 0, pred.shape[1] - 1)  
                y = np.clip(center_y, 0, pred.shape[0] - 1)  
                selected_indices = np.array([[x, y]])  
            else:
                is_training = getattr(self, 'training', False)
                if is_training and np.random.rand() < 0.2:
                    all_error_points = np.argwhere(error_mask == 1)
                    if len(all_error_points) > 0:
                        random_idx = np.random.randint(0, len(all_error_points))
                        cy, cx = all_error_points[random_idx]
                        x = np.clip(cx, 0, pred.shape[1] - 1)  
                        y = np.clip(cy, 0, pred.shape[0] - 1)  
                        selected_indices = np.array([[x, y]])  
                    else:
                        component_sizes = np.bincount(labeled_array.ravel())
                        largest_component_idx = np.argmax(component_sizes[1:]) + 1
                        max_error_region = (labeled_array == largest_component_idx)
                        dist_trans = distance_transform_edt(max_error_region)
                        cy, cx = np.unravel_index(np.argmax(dist_trans), dist_trans.shape)
                        x = np.clip(cx, 0, pred.shape[1] - 1)  
                        y = np.clip(cy, 0, pred.shape[0] - 1)  
                        selected_indices = np.array([[x, y]])  
                else:
                    component_sizes = np.bincount(labeled_array.ravel())
                    largest_component_idx = np.argmax(component_sizes[1:]) + 1
                    max_error_region = (labeled_array == largest_component_idx)
                    dist_trans = distance_transform_edt(max_error_region)
                    
                    cy, cx = np.unravel_index(np.argmax(dist_trans), dist_trans.shape)
                    
                    x = np.clip(cx, 0, pred.shape[1] - 1)  
                    y = np.clip(cy, 0, pred.shape[0] - 1)  
                    selected_indices = np.array([[x, y]])  
                
                if get_point > 1:
                    remaining_error = error_mask.copy()
                    remaining_error[max_error_region] = 0
                    
                    for i in range(1, get_point):
                        if np.sum(remaining_error) > 0:
                            labeled_remaining, num_remaining = label(remaining_error)
                            if num_remaining > 0:
                                comp_sizes = np.bincount(labeled_remaining.ravel())
                                if len(comp_sizes) > 1:
                                    next_largest = np.argmax(comp_sizes[1:]) + 1
                                    next_region = (labeled_remaining == next_largest)
                                    dist_trans_next = distance_transform_edt(next_region)
                                    next_cy, next_cx = np.unravel_index(np.argmax(dist_trans_next), dist_trans_next.shape)
                                    next_x = np.clip(next_cx, 0, pred.shape[1] - 1)  
                                    next_y = np.clip(next_cy, 0, pred.shape[0] - 1)  
                                    selected_indices = np.vstack([selected_indices, [next_x, next_y]])  
                                    remaining_error[next_region] = 0
                        else:
                            center_mask = np.zeros_like(error_mask)
                            center_mask[cy, cx] = 1
                            dist_map = cv2.distanceTransform(error_mask.astype(np.uint8), cv2.DIST_L2, 5)
                            if error_mask.sum().item() > 0:
                                _, max_val, _, _ = cv2.minMaxLoc(dist_map)
                                dynamic_radius = min(20, max_val * 0.5)
                            else:
                                dynamic_radius = 10  
                            
                            angle = 2 * np.pi * i / get_point
                            radius = dynamic_radius  
                            new_y = int(cy + radius * np.sin(angle))
                            new_x = int(cx + radius * np.cos(angle))
                            new_x = np.clip(new_x, 0, pred.shape[1]-1)  
                            new_y = np.clip(new_y, 0, pred.shape[0]-1)  
                            selected_indices = np.vstack([selected_indices, [new_x, new_y]])  
        is_training = getattr(self, 'training', False)
        if is_training:
            random_offset = np.random.randint(-2, 3, size=selected_indices.shape)
            selected_indices = selected_indices + random_offset
            selected_indices[:, 0] = np.clip(selected_indices[:, 0], 0, pred.shape[1]-1)  
            selected_indices[:, 1] = np.clip(selected_indices[:, 1], 0, pred.shape[0]-1)  
        selected_indices = selected_indices.reshape(-1, 2)
        points, labels = [], []
        for i in selected_indices:
            x, y = i[0], i[1]  
            if pred[y, x] == 0 and gt[y, x] == 1:
                label = 1
            elif pred[y, x] == 1 and gt[y, x] == 0:
                label = 0
            else:
                label = -1
            points.append((x, y))  
            labels.append(label)
        return np.array(points), np.array(labels)
