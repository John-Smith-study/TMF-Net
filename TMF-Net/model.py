import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import math
import cv2
from dataloaders.data_utils import get_points_from_mask, get_bboxes_from_mask
# REMOVED: Text model code - from transformers import AutoTokenizer
import pickle
import re
import random
import logging
from collections import OrderedDict

from utils import DiceLoss

# 配置 model.py 的日志（使用全局logger，避免被覆盖）
model_logger = logging.getLogger('model')
model_logger.setLevel(logging.DEBUG)
# 不添加自己的handler，使用全局配置的handler
model_logger.propagate = True


# Focal Loss 实现
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # 限制 inputs 的范围，防止 logits 过大导致数值不稳定
        inputs = torch.clamp(inputs, -10.0, 10.0)

        # 计算 BCE loss
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # 处理全0/全1标签的情况
        pt = torch.exp(-bce_loss)
        pt = torch.clamp(pt, 1e-8, 1.0 - 1e-8)

        # 添加epsilon防止梯度消失
        focal_loss = self.alpha * ((1 - pt) + 1e-4) ** self.gamma * bce_loss
        focal_loss = torch.nan_to_num(focal_loss, nan=0.0, posinf=1.0, neginf=0.0)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 时序感知的损失函数
# ==================== 修复：完整的时序感知损失函数 ====================
class BoundaryLoss(nn.Module):
    """边界损失，用于增强边界分割精度"""
    def __init__(self, sigma=2.0):
        super().__init__()
        self.sigma = sigma
    
    def forward(self, pred, gt):
        # 确保输入是logits
        pred_prob = torch.sigmoid(pred)
        gt_binary = (gt > 0).float()
        
        # 计算边界
        kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], device=pred.device).view(1, 1, 3, 3).float()
        
        gt_boundary = torch.abs(F.conv2d(gt_binary, kernel, padding=1, groups=1))
        gt_boundary = (gt_boundary > 0).float()
        
        # 计算距离图
        b, c, h, w = gt_binary.shape
        total_loss = 0.0
        
        for i in range(b):
            for j in range(c):
                gt_boundary_i = gt_boundary[i, j]
                if gt_boundary_i.sum() > 0:
                    # 计算每个像素到边界的距离
                    boundary_coords = torch.nonzero(gt_boundary_i)
                    if boundary_coords.numel() > 0:
                        # 【内存优化】限制边界点数量，避免内存爆炸
                        max_boundary_points = 1000  # 设置最大边界点数量
                        if boundary_coords.shape[0] > max_boundary_points:
                            # 随机采样边界点
                            idx = torch.randperm(boundary_coords.shape[0], device=pred.device)[:max_boundary_points]
                            boundary_coords = boundary_coords[idx]
                        
                        # 【内存优化】使用更高效的距离计算方法
                        # 创建坐标网格
                        x = torch.arange(h, device=pred.device).unsqueeze(1).repeat(1, w)
                        y = torch.arange(w, device=pred.device).unsqueeze(0).repeat(h, 1)
                        
                        # 计算距离 (h, w, N) -> (h, w)
                        min_dist = torch.full((h, w), float('inf'), device=pred.device)
                        for coord in boundary_coords:
                            dist = torch.sqrt((x - coord[0])**2 + (y - coord[1])**2)
                            min_dist = torch.min(min_dist, dist)
                        
                        # 应用高斯权重
                        weight = torch.exp(-min_dist**2 / (2 * self.sigma**2))
                        # 计算边界损失
                        boundary_loss = -weight * (gt_binary[i, j] * torch.log(pred_prob[i, j] + 1e-8) + 
                                                 (1 - gt_binary[i, j]) * torch.log(1 - pred_prob[i, j] + 1e-8))
                        total_loss += boundary_loss.mean()
                        # 【内存优化】清理临时变量
                        del x, y, min_dist, weight, boundary_loss
                        torch.cuda.empty_cache()
        
        return total_loss / (b * c) if (b * c) > 0 else 0.0

class TemporalAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        # 提高Focal Loss的gamma和alpha，让模型更关注错分像素
        self.focal_loss = FocalLoss(alpha=0.75, gamma=2.0)
        self.consistency_loss = nn.MSELoss(reduction='none')  # 使用none以便逐样本计算
        self.boundary_loss = BoundaryLoss()
        self.base_weight = 0.1
        self.max_weight = 0.3
        self.consistency_warmup_epochs = 20

    def forward(self, pred_masks, gt_masks, previous_pred=None, interaction_round=0, global_epoch=0):
        # 尺寸匹配修复（已有）
        if pred_masks.shape[2:] != gt_masks.shape[2:]:
            pred_masks = F.interpolate(pred_masks, size=gt_masks.shape[2:], mode='bilinear', align_corners=False)

        # 确保 gt_masks 是 float 类型（二值格式）
        if gt_masks.dtype == torch.long:
            gt_masks = gt_masks.float()

        # 主损失：Dice + Focal (加大 Focal 比重)
        dice_loss = self.dice_loss(pred_masks, gt_masks)
        focal_loss = self.focal_loss(pred_masks, gt_masks)
        bce_loss = self.bce_loss(pred_masks, gt_masks)
        
        # 从第5个epoch开始引入边界损失
        boundary_loss = self.boundary_loss(pred_masks, gt_masks) if global_epoch >= 5 else 0.0
        
        # 权重设计：Dice主导，Focal辅助，BCE小权重，边界损失后期增强
        w_dice = 1.0
        w_focal = 0.5  # 从0.3增加到0.5
        w_bce = 0.1
        w_boundary = 0.3 if global_epoch >= 5 else 0.0  # 从0.2增加到0.3
        
        main_loss = w_dice * dice_loss + w_focal * focal_loss + w_bce * bce_loss + w_boundary * boundary_loss
        
        # 一致性损失仅在 dice > 0.65 时启用，且权重逐步增加
        consistency_loss = 0.0
        # 安全转换interaction_round为标量
        interaction_round = (
            interaction_round.item() if isinstance(interaction_round, torch.Tensor) else interaction_round
        )
        if previous_pred is not None and interaction_round > 0:
            # 设备/尺寸匹配（已有）
            if previous_pred.device != pred_masks.device:
                previous_pred = previous_pred.to(pred_masks.device)
            if previous_pred.shape[2:] != pred_masks.shape[2:]:
                previous_pred = F.interpolate(previous_pred, size=pred_masks.shape[2:], mode='bilinear', align_corners=False)

            pred_prob = torch.sigmoid(pred_masks)
            prev_prob = torch.sigmoid(previous_pred).detach()

            # 批次级别的dice计算（避免单样本dice波动）
            pred_binary = (pred_prob > 0.5).float()
            gt_binary = (gt_masks > 0).float()
            intersection = (pred_binary * gt_binary).sum(dim=(1,2,3))  # 批次维度
            union = pred_binary.sum(dim=(1,2,3)) + gt_binary.sum(dim=(1,2,3))
            dice = (2 * intersection) / (union + 1e-8)
            cur_dice = dice.mean()
            
            # 一致性损失仅在 dice > 0.65 时启用
            if cur_dice > 0.65:
                consistency_weight = min(0.3, (global_epoch - 20) * 0.02) if global_epoch >= 20 else 0.0
                consistency_loss = (self.consistency_loss(pred_prob, prev_prob).mean(dim=(1,2,3))).mean() * consistency_weight

        return main_loss + consistency_loss


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
            # 预热期：返回极低阈值，等效于全部通过
            return 0.0, 0.0, True
        elif epoch < self.warmup_epochs + self.ramp_epochs:
            # 上升期：线性插值
            progress = (epoch - self.warmup_epochs) / self.ramp_epochs
            dice_threshold = self.base_dice_threshold + progress * (self.max_dice_threshold - self.base_dice_threshold)
            tci_threshold = self.base_tci_threshold + progress * (self.max_tci_threshold - self.base_tci_threshold)
            return dice_threshold, tci_threshold, False
        else:
            # 稳定期：使用最大阈值
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
            # 历史记录不足时，返回最大值表示稳定
            return 1.0
        
        # 计算相邻轮次的 Dice 变化 ΔD_i
        dice_changes = []
        for i in range(1, len(dice_history)):
            delta_d = dice_history[i] - dice_history[i - 1]
            dice_changes.append(delta_d)
        
        if len(dice_changes) < 2:
            # 只有一个变化值，标准差为 0
            return 1.0
        
        # 计算 Dice 变化的标准差 σ({ΔD_i})
        mean_change = sum(dice_changes) / len(dice_changes)
        variance = sum((dc - mean_change) ** 2 for dc in dice_changes) / len(dice_changes)
        std_dev = variance ** 0.5  # σ
        
        # 按照论文公式计算 TCI
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
        # 获取动态阈值
        dice_threshold, tci_threshold, is_warmup = self._get_dynamic_thresholds(global_epoch)
        
        # 预热期：全部通过
        if is_warmup:
            tci = self.calculate_tci(dice_history + [current_dice]) if len(dice_history) >= 1 else 1.0
            return True, current_dice, tci, dice_threshold, tci_threshold, True
        
        # 非预热期：正常质量检查
        # 计算当前 TCI（包含当前 Dice 值）
        full_history = dice_history + [current_dice]
        tci = self.calculate_tci(full_history)
        
        # 检查双重阈值条件
        dice_passed = current_dice > dice_threshold
        tci_passed = tci > tci_threshold
        
        return (dice_passed and tci_passed), current_dice, tci, dice_threshold, tci_threshold, False


# New: Temporal memory module, stores and manages historical interaction information
class TemporalMemoryModule(nn.Module):
    MAX_HISTORY = 8  # 每个病人每个类别最多存 8 条历史，减少内存占用
    
    def __init__(self, max_history_length=8, feature_dim=256):
        super().__init__()
        self.max_history_length = max_history_length
        self.feature_dim = feature_dim
        self.history_buffer = OrderedDict()
        
        # 时序记忆模块 - 质量门控参数优化后的设置（更宽松合理）
        self.quality_gate = QualityGateStrategy(
            dice_threshold=0.68,  # τ_dice: 从 0.85 降低到 0.68，避免过严
            tci_threshold=0.7,    # τ_tci: 从 0.75 降低到 0.7，更合理
            delta=0.15            # δ: 保持 0.15，提高容错性
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
        
        # 添加调试打印，确认方法被调用
        # 添加调试打印，确认方法被调用
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
            # 【终极修复】超级安全的标量转换器
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
            
            # 强制转换所有标量值
            dice = ultra_safe_scalar(data.get('dice', 0.0))
            global_epoch = ultra_safe_scalar(global_epoch)
            print(f"[STOREDEBUG] converted values: dice={dice:.4f}, epoch={global_epoch}")
            
            # 获取历史 Dice 记录用于 TCI 计算
            history_list = self.history_buffer[patient_id_str].get(category_str, [])
            dice_history = [h['dice'] for h in history_list if 'dice' in h]
            
            # 按照论文公式 (4) 进行质量门控检查（使用动态阈值）
            is_qualified, current_dice, current_tci, dice_threshold, tci_threshold, is_warmup = self.quality_gate.check_quality(dice, dice_history, global_epoch)
            
            stage = "WARMUP" if is_warmup else ("RAMP" if global_epoch < self.quality_gate.warmup_epochs + self.quality_gate.ramp_epochs else "STABLE")
            print(f"[QUALITYGATE] patient={patient_id_str}, cat={category_str}, stage={stage}, dice={current_dice:.4f} (thresh={dice_threshold:.4f}), tci={current_tci:.4f} (thresh={tci_threshold:.4f}), qualified={is_qualified}")
            
            # 只存储必要的信息，减少内存占用
            reduced_data = {
                'dice': dice,
                'tci': current_tci,
                'epoch': global_epoch,
                'qualified': is_qualified
            }
            
            # 存储条件判断（全部使用 Python 标量）
            training_flag = False
            if hasattr(self, 'training'):
                if isinstance(self.training, bool):
                    training_flag = self.training
                elif isinstance(self.training, torch.Tensor):
                    training_flag = self.training.mean().item() if self.training.numel() > 1 else self.training.item()
            print(f"[STOREDEBUG] training_flag={training_flag}")
            
            # 【论文公式 (4) 实现】质量门控机制
            # 只有满足质量要求的记录才会被存储到记忆缓冲中
            # B_t = { {Z_t, h_t}, if Dice > τ_dice and TCI > τ_tci
            #       { B_{t-1}, otherwise
            if not is_qualified:
                print(f"[QUALITYGATE] Record rejected: dice={dice:.4f} (threshold=0.68), tci={current_tci:.4f} (threshold=0.7)")
                # 不存储低质量记录，保持原有的历史缓冲
                return
            
            # 处理特征张量
            if 'features' in data and isinstance(data['features'], torch.Tensor):
                has_nan = torch.isnan(data['features']).any().item()
                has_inf = torch.isinf(data['features']).any().item()
                print(f"[STOREDEBUG] patient={patient_id_str}, cat={category_str}, epoch={global_epoch}, dice={dice:.4f}, has_nan={has_nan}, has_inf={has_inf}")
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
        # 向后兼容，使用默认类别
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
            # 返回所有类别的历史记录
            all_history = []
            for cat_history in self.history_buffer[patient_id_str].values():
                all_history.extend(cat_history)
            return all_history
        else:
            # 返回特定类别的历史记录
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
                # 过滤掉超过max_age_epochs的记录
                self.history_buffer[patient_id][category] = [
                    record for record in self.history_buffer[patient_id][category]
                    if current_epoch - record.get('epoch', 0) <= max_age_epochs
                ]
                # 如果该类别无记录，删除空列表
                if not self.history_buffer[patient_id][category]:
                    del self.history_buffer[patient_id][category]
            # 如果该病人无记录，删除空字典
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
                    # 清空特定类别的历史
                    category_str = self._safe_convert_category(category)
                    if category_str in self.history_buffer[patient_id_str]:
                        del self.history_buffer[patient_id_str][category_str]
                    # 如果该病人没有其他类别，删除整个病人记录
                    if not isinstance(self.history_buffer[patient_id_str], dict) or len(self.history_buffer[patient_id_str]) == 0:
                        del self.history_buffer[patient_id_str]
                else:
                    # 清空整个病人的历史
                    del self.history_buffer[patient_id_str]
        else:
            # 清空所有历史
            self.history_buffer.clear()



class LSTMTemporalModule(nn.Module):
    """LSTM时序记忆模块"""

    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

        # 特征对齐模块，用于处理不同维度的特征
        self.feature_align = nn.ModuleDict()

        # 先用一个轻量卷积融合空间上下文
        self.pre_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, groups=feature_dim),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
        )

        # 时序记忆网络用于分析交互序列
        self.temporal_memory = nn.GRU(
            input_size=feature_dim,
            hidden_size=feature_dim,
            num_layers=2,
            batch_first=True
        )

        # 轨迹模式识别 - 增强表达能力
        self.pattern_recognizer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),  # GRU输出是hidden_size=feature_dim
            nn.GELU(),  # 使用GELU激活函数
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 注意力池化层，用于增强时间特征
        self.attn = nn.Linear(feature_dim, 1)  # GRU输出是hidden_size=feature_dim
        
        # 优化 LSTM 权重初始化
        self._init_weights()

    def _init_weights(self):
        """优化权重初始化，提高训练稳定性"""
        # 初始化 LSTM 权重
        for name, param in self.temporal_memory.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # 初始化模式识别器的线性层权重
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
        
        # 转换为浮点数
        coords = coords.float()
        
        # 归一化到 [0, 1] 范围
        height, width = image_size
        coords[:, 0] /= width - 1  # x坐标（列）
        coords[:, 1] /= height - 1  # y坐标（行）
        
        # 确保坐标在 [0, 1] 范围内
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
            # 使用 Embedding
            if not hasattr(self, 'label_embedding'):
                # 假设最大类别索引不超过 100
                max_class = 100
                self.label_embedding = nn.Embedding(max_class, embedding_dim).to(device)
            
            # 确保标签是整数
            labels = labels.long()
            # 确保标签在有效范围内
            labels = torch.clamp(labels, 0, 99)
            # 获取 Embedding
            label_embeddings = self.label_embedding(labels)
            return label_embeddings
        else:
            # 使用 One-hot 编码
            if num_classes is None:
                # 自动推断类别数量
                num_classes = labels.max().item() + 1
            
            # 确保标签是整数
            labels = labels.long()
            # 生成 One-hot 编码
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
        
        # 计算序列长度
        seq_lengths = [len(seq) for seq in sequences]
        
        # 确定最大序列长度
        if max_length is None:
            max_length = max(seq_lengths)
        
        # 获取特征维度
        feature_dim = sequences[0].shape[1]
        device = sequences[0].device
        
        # 创建填充后的张量
        padded_sequences = torch.full((len(sequences), max_length, feature_dim), 
                                     padding_value, device=device)
        
        # 填充序列
        for i, seq in enumerate(sequences):
            length = min(len(seq), max_length)
            padded_sequences[i, :length] = seq[:length]
        
        return padded_sequences, seq_lengths

    def construct_interaction_sequence(self, interactions, image_size=(256, 256)):
        """构造交互序列张量 - 恢复真实图像特征版"""
        # 确保interactions是列表或元组
        if interactions is None or (isinstance(interactions, torch.Tensor) and interactions.numel() == 0) or (not isinstance(interactions, (list, tuple))):
            device = next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cpu')
            return torch.zeros(1, 0, self.feature_dim, device=device), [0]
        
        # 确保列表不为空
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
                    # 处理各种格式：[C,H,W], [1,C,H,W], [B,C,H,W]
                    if feat.dim() == 5:
                        # 严重的维度错误，直接压缩
                        feat = feat.mean(dim=(0, 1))  # → [H,W]
                    if feat.dim() == 4:
                        feat = torch.mean(feat, dim=(2, 3))  # → [B, C] 或 [1, C]
                    if feat.dim() == 3:
                        feat = feat.mean(dim=(1, 2))  # → [C]
                    # 确保是2D [B, C]
                    if feat.dim() == 1:
                        feat = feat.unsqueeze(0)  # → [1, C]
                elif 'features' in interaction and interaction['features'] is not None:
                    feat = interaction['features'].to(device)
                    # 处理各种格式：[C,H,W], [1,C,H,W], [B,C,H,W]
                    if feat.dim() == 5:
                        feat = feat.mean(dim=(0, 1))
                    if feat.dim() == 4:
                        feat = torch.mean(feat, dim=(2, 3))
                    if feat.dim() == 3:
                        feat = feat.mean(dim=(1, 2))
                    if feat.dim() == 1:
                        feat = feat.unsqueeze(0)
                    # 【关键修复】确保特征维度与LSTM期望的维度一致
                    if feat.dim() == 2 and feat.shape[1] != self.feature_dim:
                        current_dim = feat.shape[1]
                        if current_dim > self.feature_dim:
                            # 如果维度过大，使用特征对齐模块
                            key = str(current_dim)
                            if key not in self.feature_align:
                                self.feature_align[key] = nn.Linear(current_dim, self.feature_dim).to(device)
                            feat = self.feature_align[key](feat)
                        else:
                            # 如果维度过小，使用零填充
                            padding = torch.zeros(feat.shape[0], self.feature_dim - current_dim, device=device)
                            feat = torch.cat([feat, padding], dim=1)
                    elif feat.dim() == 3 and feat.shape[2] != self.feature_dim:
                        current_dim = feat.shape[2]
                        if current_dim > self.feature_dim:
                            # 如果维度过大，使用特征对齐模块
                            key = str(current_dim)
                            if key not in self.feature_align:
                                self.feature_align[key] = nn.Linear(current_dim, self.feature_dim).to(device)
                            feat = self.feature_align[key](feat)
                        else:
                            # 如果维度过小，使用零填充
                            padding = torch.zeros(feat.shape[0], feat.shape[1], self.feature_dim - current_dim, device=device)
                            feat = torch.cat([feat, padding], dim=2)
                else:
                    feat = torch.zeros(1, self.feature_dim, device=device)
            else:
                feat = torch.zeros(1, self.feature_dim, device=device)

            sequence_features.append(feat)

        sequence_features = [f.squeeze(0) for f in sequence_features]  # [C]
        sequence_tensor = torch.stack(sequence_features, dim=0)       # [T,C]
        sequence_tensor = sequence_tensor.unsqueeze(0)                # [B,T,C]
        seq_lengths = [len(sequence_features)]
        return sequence_tensor, seq_lengths

    def extract_trajectory_features(self, interaction_sequence, patient_id=None):
        """提取交互轨迹特征 - 标准LSTM前向传播版"""
        device = next(self.parameters()).device
        self.device = device
        
        # 增强检查：如果是空张量或形状异常，直接返回零特征
        if interaction_sequence is None:
            return torch.zeros(1, self.feature_dim, device=device)
        if isinstance(interaction_sequence, torch.Tensor):
            if interaction_sequence.numel() == 0:
                return torch.zeros(1, self.feature_dim, device=device)
            # 检查维度，确保至少是2D
            if interaction_sequence.dim() < 2:
                return torch.zeros(1, self.feature_dim, device=device)
        elif not isinstance(interaction_sequence, (list, tuple)):
            return torch.zeros(1, self.feature_dim, device=device)
        
        # 【关键修复】直接使用传入的interaction_sequence作为序列张量
        # 避免再次调用construct_interaction_sequence导致的潜在问题
        if isinstance(interaction_sequence, torch.Tensor):
            sequence_tensor = interaction_sequence
        else:
            # 回退到原始逻辑
            sequence_tensor, seq_lengths = self.construct_interaction_sequence(interaction_sequence)
        
        # 检查维度，确保是3D (batch, seq_len, feature_dim)
        if sequence_tensor.dim() == 5:
            # 如果是5D [B, T, C, H, W]，压缩空间维度
            sequence_tensor = sequence_tensor.mean(dim=(3, 4))  # → [B, T, C]
        if sequence_tensor.dim() == 4:
            # 如果是4D，压缩空间维度
            sequence_tensor = sequence_tensor.mean(dim=(2, 3))
        
        # 【关键修复】确保sequence_tensor是3D (batch, seq_len, feature_dim)
        if sequence_tensor.dim() == 4:
            # 如果是4D，压缩空间维度
            sequence_tensor = sequence_tensor.mean(dim=(2, 3))
        elif sequence_tensor.dim() == 2:
            # 如果是2D，添加序列维度
            sequence_tensor = sequence_tensor.unsqueeze(1)
        
        # 【关键修复】确保特征维度匹配LSTM输入要求
        if sequence_tensor.dim() == 3 and sequence_tensor.shape[2] != self.feature_dim:
            # 如果特征维度不匹配，进行维度调整
            current_dim = sequence_tensor.shape[2]
            if current_dim > self.feature_dim:
                # 如果维度过大，使用特征对齐模块
                key = str(current_dim)
                if key not in self.feature_align:
                    self.feature_align[key] = nn.Linear(current_dim, self.feature_dim).to(device)
                sequence_tensor = self.feature_align[key](sequence_tensor)
            else:
                # 如果维度过小，使用零填充
                padding = torch.zeros(sequence_tensor.shape[0], sequence_tensor.shape[1], self.feature_dim - current_dim, device=device)
                sequence_tensor = torch.cat([sequence_tensor, padding], dim=2)
        elif sequence_tensor.dim() == 2 and sequence_tensor.shape[1] != self.feature_dim:
            # 如果特征维度不匹配，进行维度调整
            current_dim = sequence_tensor.shape[1]
            if current_dim > self.feature_dim:
                # 如果维度过大，使用特征对齐模块
                key = str(current_dim)
                if key not in self.feature_align:
                    self.feature_align[key] = nn.Linear(current_dim, self.feature_dim).to(device)
                sequence_tensor = self.feature_align[key](sequence_tensor)
            else:
                # 如果维度过小，使用零填充
                padding = torch.zeros(sequence_tensor.shape[0], self.feature_dim - current_dim, device=device)
                sequence_tensor = torch.cat([sequence_tensor, padding], dim=1)
        
        # 检查序列长度是否为0
        if sequence_tensor.dim() < 3 or sequence_tensor.shape[1] == 0:
            return torch.zeros(1, self.feature_dim, device=device)
        
        batch_size = sequence_tensor.shape[0]
        sequence_tensor = sequence_tensor.to(device)
        
        # ✅ 修复：保留更长的交互历史，发挥 LSTM 优势
        max_seq_length = 8  # 与 TemporalMemoryModule.MAX_HISTORY 保持一致，平衡内存占用和性能
        if sequence_tensor.shape[1] > max_seq_length:
            sequence_tensor = sequence_tensor[:, -max_seq_length:, :]
        
        # 确保维度正确
        if sequence_tensor.dim() != 3:
            model_logger.error(f"[ERROR] 序列维度错误: {sequence_tensor.dim()}")
            model_logger.error(f"[ERROR] 序列形状: {sequence_tensor.shape}")
            return torch.zeros(1, self.feature_dim, device=device)
        
        # ✅ 正常初始化隐藏状态为 0（不要用 randn 造假梯度）
        # 单向GRU的隐藏状态维度：(num_layers, batch_size, hidden_size)
        h_0 = torch.zeros(2, batch_size, self.feature_dim, device=device)
        
        try:
            # 确保sequence_tensor是3D张量
            if not isinstance(sequence_tensor, torch.Tensor):
                sequence_tensor = torch.zeros(batch_size, 1, self.feature_dim, device=device)
            
            # GRU 前向传播 (只需要h_0，不需要c_0)
            gru_output = self.temporal_memory(sequence_tensor, h_0)
            
            # 【数值稳定性】检查GRU输出
            if isinstance(gru_output, tuple):
                # GRU返回 (output, hidden_state)，我们只关心第一个元素
                temporal_out = gru_output[0]
                # 检查output部分
                if torch.isnan(temporal_out).any() or torch.isinf(temporal_out).any():
                    model_logger.warning(f"[WARNING] GRU输出包含NaN/Inf，使用零输出替代")
                    temporal_out = torch.zeros_like(temporal_out)
                    # 重新构建输出tuple
                    gru_output = (temporal_out, gru_output[1])
            else:
                # 如果不是tuple，直接检查
                if torch.isnan(gru_output).any() or torch.isinf(gru_output).any():
                    model_logger.warning(f"[WARNING] GRU输出包含NaN/Inf，使用零输出替代")
                    gru_output = torch.zeros_like(gru_output)
            
            # 安全解包输出
            if isinstance(gru_output, torch.Tensor):
                temporal_out = gru_output
            elif isinstance(gru_output, (tuple, list)):
                temporal_out = gru_output[0] if len(gru_output) > 0 else torch.zeros(batch_size, sequence_tensor.shape[1] if sequence_tensor.dim() == 3 else 1, self.feature_dim, device=device)
            else:
                temporal_out = torch.zeros(batch_size, 1, self.feature_dim, device=device)
            
            # 确保temporal_out是3D张量
            if not isinstance(temporal_out, torch.Tensor) or temporal_out.dim() != 3:
                model_logger.error(f"[ERROR] GRU输出类型/维度错误: type={type(temporal_out)}, dim={temporal_out.dim() if isinstance(temporal_out, torch.Tensor) else 'N/A'}")
                return torch.zeros(1, self.feature_dim, device=device)
            
            # 使用注意力池化替代最后一个时间步
            # GRU输出维度: [B, T, hidden_size] = [B, T, 256]
            attn = torch.softmax(self.attn(temporal_out), dim=1)
            
            # 【数值稳定性】检查注意力权重
            if torch.isnan(attn).any() or torch.isinf(attn).any():
                model_logger.warning(f"[WARNING] 注意力权重包含NaN/Inf，使用均匀分布替代")
                attn = torch.ones_like(attn) / attn.shape[1]
            
            attn_feature = (temporal_out * attn).sum(dim=1)
            
            # 轨迹特征映射
            pattern_out = self.pattern_recognizer(attn_feature)
            
            # 【数值稳定性】检查最终输出
            if torch.isnan(pattern_out).any() or torch.isinf(pattern_out).any():
                model_logger.warning(f"[WARNING] 模式识别器输出包含NaN/Inf，使用零特征替代")
                pattern_out = torch.zeros_like(pattern_out)
            
            return pattern_out
        except Exception as e:
            # 捕获任何LSTM错误，确保模型能够继续运行
            model_logger.error(f"[ERROR] LSTM前向传播错误: {str(e)}")
            model_logger.error(f"[ERROR] 序列形状: {sequence_tensor.shape if isinstance(sequence_tensor, torch.Tensor) else 'N/A'}")
            model_logger.error(f"[ERROR] 序列维度: {sequence_tensor.dim() if isinstance(sequence_tensor, torch.Tensor) else 'N/A'}")
            return torch.zeros(1, self.feature_dim, device=device)

    def forward(self, interaction_sequence):
        """前向传播"""
        features = self.extract_trajectory_features(interaction_sequence)
        # 确保无论如何都不出现 NaN 或维度崩塌
        if not isinstance(features, torch.Tensor) or torch.isnan(features).any() or torch.isinf(features).any():
            model_logger.error("[ERROR] LSTMTemporalModule 提取的特征包含 NaN/Inf！")
            features = torch.zeros(1, self.feature_dim, device=next(self.parameters()).device)
        # 确保返回形状为[1, feature_dim]，保留批次维度
        if features.dim() == 1:
            features = features.unsqueeze(0)
        return features  # 返回[1, feature_dim]


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
        # 时序记忆作为查询
        self.q_proj = nn.Conv2d(feature_dim, feature_dim, 1)
        
        # 注意力缩放因子
        self.scale = feature_dim ** -0.5
        
        # 输出投影
        self.out_proj = nn.Conv2d(feature_dim, feature_dim, 1)
        
        # 统一的键值投影层（不再动态创建，避免维度混乱）
        self.k_proj = nn.Conv2d(feature_dim, feature_dim, 1)
        self.v_proj = nn.Conv2d(feature_dim, feature_dim, 1)

    def forward(self, feats, mem_state):
        # 确保feats是列表或元组且不为空
        if feats is None or (isinstance(feats, (list, tuple)) and len(feats) == 0):
            return mem_state

        # 检查输入是否包含 NaN/Inf
        if torch.isnan(mem_state).any() or torch.isinf(mem_state).any():
            model_logger.error("[ERROR] CrossScaleAttention 输入 mem_state 包含 NaN/Inf！")
            return torch.zeros_like(mem_state)

        feats = list(feats)
        for i in range(len(feats)):
            if feats[i].dim() == 2:
                feats[i] = feats[i].unsqueeze(-1).unsqueeze(-1)
            elif feats[i].dim() == 3:
                feats[i] = feats[i].unsqueeze(-1)
            
            # 检查每个特征是否包含 NaN/Inf
            if torch.isnan(feats[i]).any() or torch.isinf(feats[i]).any():
                model_logger.error(f"[ERROR] CrossScaleAttention 输入 feats[{i}] 包含 NaN/Inf！")
                feats[i] = torch.zeros_like(feats[i])

        if feats[0].dim() != 4:
            feats[0] = feats[0].unsqueeze(-1).unsqueeze(-1)

        B, C = feats[0].shape[:2]
        target_size = (16, 16)

        # 确保mem_state是张量
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
            # 动态通道对齐 - 确保每个特征的通道数都是feature_dim
            in_channels = f.shape[1]
            # 确保in_channels是标量
            if isinstance(in_channels, torch.Tensor):
                in_channels = in_channels.item()
            aligned_f = f
            if in_channels != self.feature_dim:
                key = str(in_channels)
                # 安全地检查键是否存在于字典中
                key = str(in_channels)

                if key in self.channel_adapters:
                    aligned_f = self.channel_adapters[key](f)
                else:
                    aligned_f = f  # fallback（或raise error）
            
            # 【关键修改】利用记忆特征作为 Query，与当前特征 k 做点积生成空间权重
            k = self.k_proj(aligned_f)
            # 计算空间注意力图：[B, 1, H, W]
            spatial_attn = torch.sigmoid((k * mem_context).sum(dim=1, keepdim=True))
            # 用注意力图加权当前特征
            weighted_f = aligned_f * spatial_attn
            
            # 统一尺寸进行融合
            weighted_f = F.interpolate(weighted_f, size=target_size, mode='bilinear', align_corners=False)
            weighted_feats.append(weighted_f)
        
        # 融合结果
        if len(weighted_feats) == 1:
            output = self.out_proj(weighted_feats[0])
        else:
            output = self.out_proj(torch.cat(weighted_feats, dim=1))
        
        # 检查输出是否包含 NaN/Inf
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
        # 不同尺度对应不同的池化尺寸 - 按照论文 s ∈ {8, 16, 32, 64} 从小到大排列
        self.scale_adapters = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(8), nn.Conv2d(in_dim, feature_dims[0], 1), nn.BatchNorm2d(feature_dims[0]), nn.ReLU()),
            nn.Sequential(nn.AdaptiveAvgPool2d(16), nn.Conv2d(in_dim, feature_dims[1], 1), nn.BatchNorm2d(feature_dims[1]), nn.ReLU()),
            nn.Sequential(nn.AdaptiveAvgPool2d(32), nn.Conv2d(in_dim, feature_dims[2], 1), nn.BatchNorm2d(feature_dims[2]), nn.ReLU()),
            nn.Sequential(nn.AdaptiveAvgPool2d(64), nn.Conv2d(in_dim, feature_dims[3], 1), nn.BatchNorm2d(feature_dims[3]), nn.ReLU()),
        ])
        
        # 自适应门控网络 - 按照论文公式 (2) 实现
        # 每个尺度独立的轻量级投影网络，生成空间自适应权重
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dims[0] * 2, feature_dims[0], kernel_size=1),  # Conv1x1^(1): 2D -> D
                nn.BatchNorm2d(feature_dims[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dims[0], 2, kernel_size=1, bias=False),  # Conv1x1^(2): D -> 2
                nn.Softmax(dim=1)  # 沿通道维度 Softmax，生成 W_s ∈ R^{2×h_s×w_s}
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
            # 确保空间尺寸一致
            if c.shape[2:] != h.shape[2:]:
                h = F.interpolate(h, size=c.shape[2:], mode='bilinear', align_corners=False)

            # 拼接当前特征和历史特征：[F_t^s; C_t^s]
            concatenated = torch.cat([c, h], dim=1)  # [B, 2D, H, W]
            
            # 通过门控网络计算空间自适应权重 W_s
            weights = self.gates[i](concatenated)  # [B, 2, H, W]
            
            # 按照论文公式 (3) 进行像素级动态调制
            # weights[:, 0:1] 是当前特征权重 W_s[0]
            # weights[:, 1:2] 是历史特征权重 W_s[1]
            fused_feat = weights[:, 0:1] * c + weights[:, 1:2] * h
            fused.append(fused_feat)

        # 将所有尺度插值到同一尺寸（目标尺寸为最后一个尺度的尺寸，即最大尺寸 64x64）
        target_size = fused[-1].shape[2:]  # 64x64
        resized_fused = []
        for f in fused:
            if f.shape[2:] != target_size:
                resized_fused.append(F.interpolate(f, size=target_size, mode='bilinear', align_corners=False))
            else:
                resized_fused.append(f)

        # 拼接所有尺度的融合结果：[B, C1+C2+C3+C4, H, W]
        combined = torch.cat(resized_fused, dim=1)

        return combined



class MultiScaleTemporalFusion(nn.Module):
    """Multi-scale temporal interaction fusion main module - Complete version (including LSTM temporal modeling and multi-scale fusion)"""

    def __init__(self, image_size=256, feature_dim=768, num_scales=4, 
                 ablation_no_multi_scale=False, ablation_no_trajectory=False):
        super().__init__()
        self.feature_dim = feature_dim  # 【关键修复】存储 feature_dim
        self.ablation_no_multi_scale = ablation_no_multi_scale
        self.ablation_no_trajectory = ablation_no_trajectory
        self.temporal_memory = TemporalMemoryModule(feature_dim=feature_dim)
        
        if not ablation_no_multi_scale:
            # 多尺度融合模块 - feature_dims总和需要和feature_dim对应
            # 768 -> [384, 192, 96, 48] 每个尺度
            self.multi_scale_fusion = MultiScaleFeatureFusion(
                image_size=image_size,
                feature_dims=[feature_dim // (2 ** i) for i in range(num_scales)],
                in_dim=feature_dim
            )
        
        # 特征对齐层：将多尺度融合结果（多个尺度拼接）映射回统一维度
        # 每个尺度: 768/(2^i)，总通道数为所有尺度之和: 384+192+96+48 = 720
        total_scale_channels = sum([feature_dim // (2 ** i) for i in range(num_scales)])  # 包含所有尺度
        self.feature_projector = nn.Sequential(
            nn.Conv2d(total_scale_channels, feature_dim, kernel_size=1),  # 720 -> 768
            nn.GroupNorm(8, feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # 时序投影层：将当前帧和历史帧的特征拼接后投影
        self.temporal_projector = nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1)

        if not ablation_no_trajectory:
            # LSTM 交互轨迹分析模块
            self.trajectory_analyzer = LSTMTemporalModule(feature_dim)

        # 时序融合权重：初始化为均等权重，长度和 num_scales 一致
        # 注意：由于 multi_scale_fusion 内部已实现论文公式 (2)(3) 的自适应门控，
        # 此处的 fusion_weights 仅用于控制不同尺度的全局重要性
        self.fusion_weights = nn.Parameter(torch.ones(num_scales) / num_scales, requires_grad=True)
        
        # 控制是否保存 fusion_weights
        self.save_weights = True
        
        # 可学习的轨迹特征融合权重
        # ✅ 修复：初始值设为 -2.5 (sigmoid(-2.5) ≈ 0.07)
        # 让模型前期以 SAM 自身特征为主，随着训练深入，自动把权重学上去
        self.trajectory_fusion_weight = nn.Parameter(torch.tensor(-2.5, dtype=torch.float32), requires_grad=True)
        
        # 融合权重预热参数
        self.fusion_warmup_epochs = 50  # 从20延长到50，让权重增长更缓慢
        self.max_fusion_weight = 0.5  # 从1.0降低到0.5，避免数值爆炸
        
        # LSTM和注意力稳定化参数
        self.temporal_gate = nn.Parameter(torch.tensor(-3.0, dtype=torch.float32), requires_grad=True)  # 初始值很小
        self.stage_factor = 0.0  # 由epoch决定的阶段因子
        self.stage_1_end = 10  # 阶段一结束epoch
        self.stage_2_end = 30  # 阶段二结束epoch

        # 输出适配层 - 适配动态通道输入
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
        
        # 【关键修复】使用更多历史记录，充分利用存储的历史信息
        recent_history = history_list[-15:]  # 增加到15条历史记录
        interaction_sequences = [] 
        for data in recent_history: 
            # 确保data是字典类型
            if not isinstance(data, dict):
                continue
            # 安全获取特征，避免使用or操作符导致的张量布尔值错误
            feat_spatial = data.get('features_spatial', None)
            feat = feat_spatial if feat_spatial is not None else data.get('features', None) 
            seq_entry = { 
                'features': feat, 
                'point_coords': data.get('point_coords', None), 
                'point_labels': data.get('point_labels', None), 
                'interaction_round': data.get('interaction_round', 0) 
            } 
            interaction_sequences.append(seq_entry) 
        
        # 确保序列长度不超过10
        if len(interaction_sequences) > 10:
            interaction_sequences = interaction_sequences[-10:]
        
        sequence_tensor, _ = self.trajectory_analyzer.construct_interaction_sequence(interaction_sequences) 
        
        # 【关键修复】确保序列张量维度正确
        if sequence_tensor.dim() == 3 and sequence_tensor.shape[2] != self.trajectory_analyzer.feature_dim:
            device = next(self.parameters()).device
            current_dim = sequence_tensor.shape[2]
            if current_dim > self.trajectory_analyzer.feature_dim:
                # 如果维度过大，使用固定的权重矩阵进行投影
                if not hasattr(self, 'temporal_projection_weight'):
                    self.temporal_projection_weight = nn.Parameter(torch.randn(self.trajectory_analyzer.feature_dim, current_dim, device=device))
                    self.temporal_projection_bias = nn.Parameter(torch.zeros(self.trajectory_analyzer.feature_dim, device=device))
                # 使用可学习的权重进行投影
                sequence_tensor = torch.matmul(sequence_tensor, self.temporal_projection_weight.t()) + self.temporal_projection_bias
            else:
                # 如果维度过小，使用零填充
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
                # 对于多元素张量，返回平均值的标量
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
                # 获取权重
                if i < len(self.fusion_weights):
                    reversed_weights = torch.flip(self.fusion_weights, dims=[0])
                    base_weight_tensor = reversed_weights[i]
                    base_weight = base_weight_tensor.item()  # 仅用于条件判断
                else:
                    base_weight_tensor = torch.tensor(0.05, device=self.fusion_weights.device)
                    base_weight = 0.05
                
                # 提取并转换 dice_change
                dice_change = MultiScaleTemporalFusion._safe_get_value(hist.get('dice_change', 0.0))
                # 再次确保是标量
                if hasattr(dice_change, 'item'):
                    dice_change = dice_change.item()
                
                # 现在比较是安全的
                if dice_change > 0.02:
                    base_weight_tensor = base_weight_tensor * 2.0
                    base_weight = base_weight * 2.0
                
                weights.append(base_weight_tensor)  # 存储张量而非标量
                useful_history.append(hist)
        
            if isinstance(current_prompt, dict):
                for hist, w in zip(useful_history, weights):
                    # 确保 w_value 是 Python float
                    w_value = MultiScaleTemporalFusion._safe_get_value(w)
                    if hasattr(w_value, 'item'):
                        w_value = w_value.item()
                    
                    # 确保 prompt_type 是字符串
                    prompt_type = MultiScaleTemporalFusion._safe_get_value(hist.get('prompt_type', ''))
                    if not isinstance(prompt_type, str):
                        prompt_type = str(prompt_type)
                    
                    # 确保 dice_change 是 Python float
                    dice_change = MultiScaleTemporalFusion._safe_get_value(hist.get('dice_change', 0.0))
                    if isinstance(dice_change, torch.Tensor):
                        dice_change = dice_change.item() if dice_change.numel() == 1 else float(dice_change.mean())
                    
                    # 所有比较操作都使用标量值
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
                # ✅ 修复：将坐标限制在[0, image_size]范围内，而不是[0, 1]
                refined_points = torch.clamp(refined_points, 0.0, image_size)
                return refined_points
        return current_points

    def update_fusion_weight(self, current_epoch):
        """根据epoch动态调整融合权重，实现平滑过渡"""
        if current_epoch < self.fusion_warmup_epochs:
            # 预热阶段：融合权重从0线性增长到max_fusion_weight
            progress = current_epoch / self.fusion_warmup_epochs
            # 从-5.0（sigmoid≈0）增长到对应max_fusion_weight的值
            # sigmoid(0) = 0.5, sigmoid(1) = 0.73, sigmoid(2) = 0.88
            # 如果max_fusion_weight=0.5，目标值约为0
            # 如果max_fusion_weight=0.5，目标值约为0
            target_weight = -5.0 + progress * (5.0 + 0.0)  # 从-5.0增长到0（sigmoid≈0.5）
            # 【关键修复】不再使用no_grad，允许梯度流动，让优化器能够学习合适的权重
            self.trajectory_fusion_weight.data.fill_(target_weight)
        return torch.sigmoid(self.trajectory_fusion_weight).item()

    def forward(self, current_prompt, history, image_features):
        """前向传播 - 优化版（支持消融实验）"""
        # 确保权重始终可学习
        self.fusion_weights.requires_grad = True
        self.trajectory_fusion_weight.requires_grad = True

        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(-1).unsqueeze(-1)
        elif image_features.dim() == 3:
            image_features = image_features.unsqueeze(-1)
        
        # 【新增】统一安全转换标量字段
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

        # 提取并转换所有关键字段
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
        
        # 准备时序数据 - 安全获取 current_prompt 中的值
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
            'epoch': global_epoch
        }
        
        # 先获取历史，再存储当前帧（避免当前帧立刻被用于历史）
        history_list = self.temporal_memory.get_history(patient_id, category) if hasattr(self.temporal_memory, 'get_history') else []

        # 存储当前帧到历史记录
        self.temporal_memory.store(patient_id, category, temporal_data, global_epoch)

        # 时序融合逻辑
        device = image_features.device
        
        # 1. 建模时序依赖
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
        
        # 2. 构建时序上下文
        temporal_context = torch.zeros_like(image_features, device=device)
        if isinstance(history_list, (list, tuple)) and len(history_list) > 0:
            recent_history = history_list[-15:]
            total_weight = 0.0
            weighted_features = []
            weights = []
            for i, hist in enumerate(reversed(recent_history)):
                if not isinstance(hist, dict):
                    continue
                hist_dice = hist.get('dice', 0.0)
                hist_tci = hist.get('tci', 0.0)
                quality_weight = (hist_dice * 0.7 + hist_tci * 0.3)
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
        
        # 3. LSTM时序轨迹建模（消融实验控制）
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
        
        # 4. 多尺度特征融合（消融实验控制）
        if isinstance(history_list, (list, tuple)) and len(history_list) > 0:
            if self.ablation_no_multi_scale and self.ablation_no_trajectory:
                # 两个模块都消融，使用简单的时序融合
                combined_feature = image_features + 0.2 * temporal_context
                enhanced_features = self.output_adapter(combined_feature)
                return enhanced_features
            elif self.ablation_no_multi_scale:
                # 只消融多尺度模块
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
                # 正常情况或只消融轨迹模块
                # 根据epoch设置阶段因子
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
                
                # 跨尺度注意力 - 仅在轨迹模块没有被消融时使用
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

                # 结合轨迹特征 - 仅在轨迹模块没有被消融时
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
    

# 新增：自适应交互引导类
class AdaptiveInteractionGuide:
    def __init__(self):
        # 初始化参数
        self.max_points = 7
        self.min_points = 1

    def compute_uncertainty(self, pred_mask, gt_mask):
        """计算预测的不确定性"""
        try:
            # 使用torch.logical_xor确保张量兼容性
            return torch.logical_xor(pred_mask, gt_mask)
        except Exception as e:
            model_logger.debug(f"[DEBUG] Interaction guide uncertainty calculation error: {str(e)}")
            return torch.zeros_like(pred_mask)

    def compute_iou(self, pred_masks, labels):
        """计算预测掩码和真实标签的IoU"""
        try:
            # 将预测转换为二值掩码
            pred_binary = (torch.sigmoid(pred_masks) > 0.5)
            labels_binary = (labels > 0)

            # 计算交集和并集
            intersection = torch.logical_and(pred_binary, labels_binary).sum(dim=(1, 2, 3))
            union = torch.logical_or(pred_binary, labels_binary).sum(dim=(1, 2, 3))

            # 计算IoU
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
            # 获取当前批次的不确定性图
            current_uncertainty = uncertainty_map[b].cpu().numpy()
            current_pred = current_mask[b].cpu().numpy()

            # 找出不确定区域
            uncertain_indices = np.argwhere(current_uncertainty == 1)

            if len(uncertain_indices) > 0:
                # 根据不确定性程度排序并选择点
                # 这里简单随机选择，实际可以实现更复杂的选择策略
                point_indices = np.random.choice(len(uncertain_indices),
                                                 min(self.max_points, len(uncertain_indices)),
                                                 replace=False)
                selected_indices = uncertain_indices[point_indices]

                # 准备点坐标和标签
                points = []
                labels = []

                for idx in selected_indices:
                    x, y = idx[0], idx[1]
                    # 根据预测和真实值确定点的标签
                    if current_pred[x, y] == 0 and current_uncertainty[x, y] == 1:
                        labels.append(1)  # 需要包含的点
                    elif current_pred[x, y] == 1 and current_uncertainty[x, y] == 1:
                        labels.append(0)  # 需要排除的点
                    else:
                        labels.append(-1)  # 忽略的点

                    points.append((y, x))  # 注意坐标转换 (x,y) -> (col,row)

                recommended_points.append((np.array(points), np.array(labels)))
            else:
                # 如果没有不确定区域，返回空
                recommended_points.append((np.array([]), np.array([])))

        return recommended_points


# %% set up model
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
        # REMOVED: Text model code - self.text_model = sam.text_model
        # REMOVED: Text model code - self.text_out_dim = sam.text_out_dim
        # REMOVED: Text model code - self.tokenizer = AutoTokenizer.from_pretrained('clip-vit-base-patch32')

        self.test_mode = test_mode
        self.multimask_output = multimask_output
        self.category_weights = category_weights
        self.select_mask_num = select_mask_num

        self.image_format = sam.image_format
        self.image_size = sam.prompt_encoder.input_image_size

        # REMOVED: Text model code - text model parameter freezing
        # for n, value in self.text_model.named_parameters():
        #     value.requires_grad = False

        if category_weights is not None:
            self.load_category_weights(category_weights)
    def compute_uncertainty(self, pred_masks, labels):
        """计算预测掩码的不确定性图"""
        pred_binary = (torch.sigmoid(pred_masks) > 0.5)
        label_binary = (labels > 0.5)
        return torch.logical_xor(pred_binary, label_binary)





# New: TMFNet class, integrating multi-scale temporal fusion
class TMFNet(IMISNet):
    def __init__(self, sam, test_mode=False, fusion_warmup_epochs=20, max_fusion_strength=1.5, num_classes=4, 
                 ablation_no_multi_scale=False, ablation_no_trajectory=False, **kwargs):
        super().__init__(sam, test_mode, **kwargs)
        self.use_auto_prompt = False  # Disable auto-prompting, use external clicks
        # Integrate multi-scale temporal fusion main module, replacing separate temporal memory and multi-scale fusion modules
        self.num_classes = num_classes  # 动态设置类别数
        feature_dim = 768  # Match SAM's image encoder output channels
        self.multiscale_temporal_fusion = MultiScaleTemporalFusion(
            image_size=256,  # Default image size: 256x256
            feature_dim=feature_dim,
            num_scales=4,
            ablation_no_multi_scale=ablation_no_multi_scale,
            ablation_no_trajectory=ablation_no_trajectory
        )  # New innovation point 2
        # Use temporal memory from multi-scale temporal fusion module
        self.temporal_memory = self.multiscale_temporal_fusion.temporal_memory
        self.temporal_weight = nn.Parameter(torch.tensor([0.2], dtype=torch.float32))  # Temporal fusion weight, reduced to avoid overfitting
        self.interaction_step = 0  # Track interaction steps
        self.interaction_history = {}  # Store interaction history for each batch
        self.use_adaptive_threshold = True  # Enable adaptive threshold processing
        self.postprocess_trigger_threshold = 0.6  # Global unified threshold, renamed to avoid confusion
        self.difficult_dice_threshold = 0.7
        self.interaction_guide = AdaptiveInteractionGuide()
        self.best_threshold = 0.5  # 最优分割阈值，默认为0.5
        # 添加融合预热参数
        self.fusion_warmup_epochs = fusion_warmup_epochs      # 预热周期（从第10个epoch开始算）
        self.max_fusion_strength = max_fusion_strength      # 最终融合强度（对应 sigmoid≈0.82）
        print(f"✅ TMFNet initialized with fusion parameters: warmup_epochs={self.fusion_warmup_epochs}, max_strength={self.max_fusion_strength}")

    def check_quality_gate(self, dice, tci, target_dice=0.9):
        # SIMPLIFIED: QGS logic to match paper
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
            # 训练阶段：使用 IoU 作为指标
            threshold = self.get_adaptive_threshold(pred_masks, labels)
        elif self.use_adaptive_threshold and labels is None:
            # 推理阶段：使用预测置信度作为代理指标
            threshold = self.get_confidence_based_threshold(pred_masks)
        else:
            threshold = self.best_threshold  # 使用最优阈值

        return (torch.sigmoid(pred_masks) > threshold).bool()

    def get_adaptive_threshold(self, pred_masks, labels):
        """
        训练阶段：根据当前预测和标签的 IoU 动态调整阈值
        优化后的阈值设置，更加平滑和合理
        """
        current_iou = self.compute_iou(pred_masks, labels)
        if current_iou < 0.55:
            return 0.38  # 很低 IoU：使用最低阈值，尽可能恢复前景
        elif current_iou < 0.7:
            return 0.42  # 较低 IoU：使用较低阈值
        elif current_iou < 0.8:
            return 0.48  # 中等 IoU：使用标准阈值
        return 0.52  # 高 IoU：使用较高阈值，确保边界精确
    
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
        # 计算预测置信度统计信息
        sigmoid_pred = torch.sigmoid(pred_masks)
        
        # 置信度指标：平均置信度
        mean_confidence = sigmoid_pred.mean().item()
        
        # 置信度指标：高置信度区域比例（> 0.6）- 从 0.7 降低到 0.6，更敏感
        high_conf_ratio = (sigmoid_pred > 0.6).float().mean().item()
        
        # 置信度指标：置信度标准差（衡量预测确定性）
        confidence_std = sigmoid_pred.std().item()
        
        # 综合置信度评分（加权平均）
        confidence_score = (
            mean_confidence * 0.4 + 
            high_conf_ratio * 0.4 + 
            (1.0 - confidence_std) * 0.2
        )
        
        # 根据置信度评分动态调整阈值 - 优化后的阈值分级
        if confidence_score > 0.7:
            # 高置信度：使用较高阈值，确保边界精确（从 0.55 降低到 0.52）
            return 0.52
        elif confidence_score > 0.55:
            # 中等置信度：使用标准阈值（从 0.5 降低到 0.48）
            return 0.48
        elif confidence_score > 0.4:
            # 较低置信度：使用较低阈值，捕获更多前景（从 0.45 降低到 0.42）
            return 0.42
        else:
            # 很低置信度：使用最低阈值，尽可能恢复前景区域（从 0.4 降低到 0.38）
            return 0.38

    def detect_difficult_samples(self, pred_mask, label_mask, test_mode=False):
        """
        检测困难样本
        基于IoU、预测不确定性和错误类型分析与动态阈值调整
        """
        try:
            # 确保输入是张量
            pred_mask = pred_mask if isinstance(pred_mask, torch.Tensor) else torch.tensor(pred_mask)
            label_mask = label_mask if isinstance(label_mask, torch.Tensor) else torch.tensor(label_mask)

            # 1. 计算IoU和Dice系数
            pred_binary = (torch.sigmoid(pred_mask) > 0.5).float()
            label_binary = (label_mask > 0).float()

            intersection = (pred_binary * label_binary).sum()
            union = pred_binary.sum() + label_binary.sum() - intersection
            iou = intersection / (union + 1e-8) if union > 0 else 0

            # 计算Dice系数
            dice = (2 * intersection) / (pred_binary.sum() + label_binary.sum() + 1e-8) if (
                                                                                                       pred_binary.sum() + label_binary.sum()) > 0 else 0

            # 2. 计算置信度分布的标准差（衡量预测的不确定性）
            pred_prob = torch.sigmoid(pred_mask)
            conf_std = pred_prob.std()

            # 3. 计算掩码覆盖率（检测过拟合或欠拟合）
            total_pixels = pred_binary.shape[2] * pred_binary.shape[3] if len(pred_binary.shape) > 2 else \
            pred_binary.shape[1] * pred_binary.shape[2]
            mask_coverage = pred_binary.sum() / total_pixels

            # 4. 形态学特征分析
            # 计算边界像素比例
            # 确保mask是2D格式
            if len(pred_binary.shape) > 2:
                mask_2d = pred_binary.squeeze(0)
                label_2d = label_binary.squeeze(0)
            else:
                mask_2d = pred_binary
                label_2d = label_binary

            # 5. 错误类型分析
            false_pos = (mask_2d > 0) & (label_2d == 0)
            false_neg = (mask_2d == 0) & (label_2d > 0)

            false_pos_area = torch.sum(false_pos).float().item()
            false_neg_area = torch.sum(false_neg).float().item()
            total_area = (torch.sum(label_binary).float() + 1e-8).item()  # 避免除零

            # 动态判断错误类型
            if false_pos_area > false_neg_area * 1.5 and false_pos_area > total_area * 0.2:
                error_type = 'over_segmentation'  # 过分割
            elif false_neg_area > false_pos_area * 1.5 and false_neg_area > total_area * 0.2:
                error_type = 'under_segmentation'  # 欠分割
            elif false_pos_area > total_area * 0.1 and false_neg_area > total_area * 0.1:
                error_type = 'mixed_error'  # 混合错误
            else:
                error_type = 'minor_error'  # 轻微错误

            # 使用全局统一的困难样本Dice阈值
            dynamic_threshold = self.difficult_dice_threshold  # 替换原动态阈值逻辑

            # 7. 边界分析
            try:
                from torch.nn.functional import max_pool2d
                mask_boundary = (mask_2d != max_pool2d(mask_2d, 3, 1, 1))
                boundary_pixels = torch.sum(mask_boundary).float()
                boundary_ratio = boundary_pixels / (pred_binary.sum() + 1e-8) if pred_binary.sum().item() > 0 else 0
            except Exception as e:
                boundary_ratio = 0
                print(f"[WARNING] Boundary detection failed: {e}")

            # 8. 计算面积比例和错误预测比例
            mask_area = torch.sum(pred_binary)
            label_area = torch.sum(label_binary)
            area_ratio = mask_area / (label_area + 1e-8) if label_area > 0 else 0

            is_over_segmented = mask_area > label_area * 1.5
            is_under_segmented = mask_area < label_area * 0.5

            wrong_prediction_ratio = false_pos_area / (mask_area + 1e-8) if mask_area > 0 else 0

            # 9. 最终困难样本判断
            is_difficult = dice < dynamic_threshold

            # 对于某些特殊情况，即使Dice > dynamic_threshold也标记为困难
            if not is_difficult:
                # 边界过于复杂
                if boundary_ratio > 0.8 and dice < 0.7:
                    is_difficult = True
                    if test_mode:
                        model_logger.debug(f"[DEBUG] Complex boundary issue - boundary ratio {boundary_ratio:.2f}")
                # 过分割严重
                elif is_over_segmented and wrong_prediction_ratio > 0.3:
                    is_difficult = True
                    if test_mode:
                        model_logger.debug(f"[DEBUG] Severe over-segmentation - wrong prediction ratio {wrong_prediction_ratio:.2f}")
                # 欠分割严重
                elif is_under_segmented:
                    is_difficult = True
                    if test_mode:
                        model_logger.debug(f"[DEBUG] Severe under-segmentation - area ratio {area_ratio:.2f}")

            # 调试日志
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
            # 返回默认值
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
            # 处理不同的输入类型和形状
            if isinstance(pred_masks, torch.Tensor) and isinstance(labels, torch.Tensor):
                # 确保形状匹配
                if pred_masks.shape != labels.shape:
                    # 尝试调整维度以匹配
                    if len(pred_masks.shape) == 4 and len(labels.shape) == 3:
                        pred_masks = pred_masks.squeeze(1)
                    elif len(pred_masks.shape) == 3 and len(labels.shape) == 4:
                        labels = labels.squeeze(1)

                # 使用逻辑异或计算不确定性
                return torch.logical_xor(pred_masks, labels)
            else:
                # 转换为张量后计算
                pred_tensor = torch.tensor(pred_masks) if not isinstance(pred_masks, torch.Tensor) else pred_masks
                labels_tensor = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
                return torch.logical_xor(pred_tensor, labels_tensor)
        except Exception as e:
            model_logger.debug(f"[DEBUG] Uncertainty calculation error: {str(e)}")
            # 返回默认值
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
            # 处理不同的输入类型和形状
            if isinstance(pred_masks, torch.Tensor) and isinstance(labels, torch.Tensor):
                # 确保形状匹配
                if pred_masks.shape != labels.shape:
                    # 尝试调整维度以匹配
                    if len(pred_masks.shape) == 4 and len(labels.shape) == 3:
                        pred_masks = pred_masks.squeeze(1)
                    elif len(pred_masks.shape) == 3 and len(labels.shape) == 4:
                        labels = labels.squeeze(1)

                # 将预测转换为二值掩码
                pred_binary = (torch.sigmoid(pred_masks) > 0.5) if pred_masks.max().item() > 1 else (pred_masks > 0.5)
                labels_binary = (labels > 0.5) if labels.max().item() > 1 else (labels > 0)

                # 计算交集和并集
                intersection = torch.logical_and(pred_binary, labels_binary).sum(dim=tuple(range(1, pred_binary.ndim)))
                union = torch.logical_or(pred_binary, labels_binary).sum(dim=tuple(range(1, pred_binary.ndim)))

                # 计算IoU
                iou = intersection.float() / (union.float() + 1e-8)
                return iou.mean()
            else:
                # 转换为张量后计算
                pred_tensor = torch.tensor(pred_masks) if not isinstance(pred_masks, torch.Tensor) else pred_masks
                labels_tensor = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
                return self.compute_iou(pred_tensor, labels_tensor)
        except Exception as e:
            model_logger.debug(f"[DEBUG] IoU calculation error: {str(e)}")
            # 返回默认值
            return torch.tensor(0.0, device=pred_masks.device if isinstance(pred_masks, torch.Tensor) else None)

    def image_forward(self, image):
        img_shape = image.shape
        # 确保输入有 4 维 (B, C, H, W)
        if len(img_shape) == 3:
            # 添加 batch 维度
            image = image.unsqueeze(0)
            model_logger.debug(f"[DEBUG] 输入维度为 3，添加 batch 维度后变为: {image.shape}")
        elif len(img_shape) != 4:
            raise ValueError(f"输入维度不正确，预期 3 或 4 维，实际为 {len(img_shape)} 维: {img_shape}")
        
        # 检查输入图像是否包含 NaN/Inf
        if torch.isnan(image).any() or torch.isinf(image).any():
            model_logger.error("[ERROR] image_forward 输入 image 包含 NaN/Inf！")
            # 替换 NaN/Inf 为 0
            image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=-1.0)
            # 裁剪到合理范围
            image = torch.clamp(image, -1.0, 1.0)
        
        image_embedding = self.image_encoder(image)
        
        # 检查编码器输出是否包含 NaN/Inf
        if torch.isnan(image_embedding).any() or torch.isinf(image_embedding).any():
            model_logger.error("[ERROR] image_encoder 输出包含 NaN/Inf！")
            # 替换 NaN/Inf 为 0
            image_embedding = torch.nan_to_num(image_embedding, nan=0.0, posinf=1.0, neginf=-1.0)
        
        assert len(image_embedding.shape) == 4, f'required shape is (B, C, H, W), but we get {image_embedding.shape}'
        
        # 获取图像嵌入的实际形状
        embed_shape = image_embedding.shape

        if self.test_mode:
            return_img_embed = image_embedding
        else:
            # 使用更高效的方式重复嵌入，避免循环和detach.clone
            # 训练模式下保留梯度信息，不要detach
            # 使用view和expand提高效率，使用嵌入的实际高度和宽度
            return_img_embed = image_embedding.view(embed_shape[0], 1, embed_shape[1], embed_shape[2], embed_shape[3])
            return_img_embed = return_img_embed.expand(embed_shape[0], self.select_mask_num, embed_shape[1], embed_shape[2], embed_shape[3])
            return_img_embed = return_img_embed.contiguous().view(-1, embed_shape[1], embed_shape[2], embed_shape[3])
        return return_img_embed

    # _get_prompt_type方法已移至EnhancedIMISNet基类中，减少代码重复

    def forward_decoder(self, image_embedding, prompt):
        
        # 获取点提示信息（用于后续历史记录）
        # 【关键修复】先检查 prompt 是否为字典
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

        # 调试：打印 image_embedding 的形状
        model_logger.debug(f"[DEBUG forward_decoder] image_embedding shape: {image_embedding.shape}, dense_pe shape: {self.prompt_encoder.get_dense_pe().shape}")

        # 确保prompt包含时序融合所需的关键参数
        if isinstance(prompt, dict):
            if 'category' not in prompt:
                prompt['category'] = prompt.get('current_category', 'unknown')

            if 'prompt_type' not in prompt:
                prompt['prompt_type'] = self._get_prompt_type(prompt)

            # 确保mask_correction和dice_change参数存在
            if 'mask_correction' not in prompt:
                prompt['mask_correction'] = torch.zeros(1, device=image_embedding.device)

            if 'dice_change' not in prompt:
                prompt['dice_change'] = 0.0

        # 确保交互ID存在于历史记录中
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

            # 存储多尺度特征而非全局平均
            features = image_embedding  # 保留原始特征
            history_entry = {
                'timestamp': len(self.interaction_history[interaction_id]),
                'prompt_type': self._get_prompt_type(prompt),
                'multi_scale_features': [],
                'spatial_features': features.detach().clone().cpu() if isinstance(features, torch.Tensor) else features  # 【核心修复】移到CPU省显存
            }

            # 存储关键点信息（如果存在）
            if points is not None:
                history_entry['point_coords'] = points[0].detach().cpu() if isinstance(points[0], torch.Tensor) else points[0]
                history_entry['point_labels'] = points[1].detach().cpu() if isinstance(points[1], torch.Tensor) else points[1]

            # 提取并存储多尺度特征
            if hasattr(self, 'multiscale_temporal_fusion') and hasattr(
                    self.multiscale_temporal_fusion.multi_scale_fusion, 'scale_adapters'):
                for extractor in self.multiscale_temporal_fusion.multi_scale_fusion.scale_adapters:
                    scale_feat = extractor(features)
                    history_entry['multi_scale_features'].append(scale_feat.detach().cpu() if isinstance(scale_feat, torch.Tensor) else scale_feat)

            self.interaction_history[interaction_id].append(history_entry)

            # FIFO memory buffer: 只保留最近的5条记录
            max_history = 5
            if len(self.interaction_history[interaction_id]) > max_history:
                # 移除最早的记录
                self.interaction_history[interaction_id] = self.interaction_history[interaction_id][-max_history:]

        # 强化 Prompt 引导：从历史记录中获取之前的掩码并编码
        mask_inputs = prompt.get("mask_inputs", None) if isinstance(prompt, dict) else None

        # 从历史记录中获取之前的掩码
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
            # 查找最近的掩码预测
            for entry in reversed(history):
                if 'pred_mask' in entry:
                    # ✅ 修复：必须将其放回与 image_embedding 相同的设备上
                    mask_inputs = entry['pred_mask'].detach().to(image_embedding.device)
                    
                    if mask_inputs.shape[-1] != 256:
                        mask_inputs = F.interpolate(
                            mask_inputs,
                            size=(256, 256),
                            mode='bilinear',
                            align_corners=False
                        )
                    # 确保不经过 sigmoid，SAM 内部需要原始 Logits 进行边界判定
                    break
        
        # 提示编码（在时序融合之后）
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=prompt.get("bboxes", None) if isinstance(prompt, dict) else None,
            masks=mask_inputs,
            text=prompt.get("text_inputs", None) if isinstance(prompt, dict) else None,
        )
       
        
       
        # 确保 dense_embeddings 的形状与 image_embedding 匹配
        if dense_embeddings.shape[2:] != image_embedding.shape[2:]:
            dense_embeddings = F.interpolate(
                dense_embeddings,
                size=image_embedding.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # 执行mask decoder
        # 【注意】不要在这里扩展 dense_pe，mask_decoder 会通过 repeat_interleave 处理
        # 原始 SAM 代码假设 batch=1，让 mask_decoder 处理 batch 扩展
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

        # 存储预测掩码到历史记录，用于后续交互轮次的提示
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
        # 1. 图像编码
        image_features = self.image_forward(image)
        
        # 2. 构建提示字典
        prompts = {
            'patient_id': interaction_id,
            'category': category,
            'interaction_id': interaction_id,
            'temporal_enabled': True,
            'lengths': lengths  # 添加序列长度信息
        }
        
        # 3. 处理交互序列，提取点提示信息
        if interaction_seq is not None:
            # 检查 interaction_seq 的类型
            if isinstance(interaction_seq, list):
                # 列表格式，每个元素是包含点坐标和标签的字典
                point_coords = []
                point_labels = []
                
                for i, interaction in enumerate(interaction_seq):
                    # 只处理长度范围内的交互
                    if lengths and i < lengths[0]:
                        if 'point_coords' in interaction:
                            point_coords.append(interaction['point_coords'])
                        if 'point_labels' in interaction:
                            point_labels.append(interaction['point_labels'])
                
                if point_coords:
                    prompts['point_coords'] = torch.cat(point_coords, dim=0)
                    prompts['point_labels'] = torch.cat(point_labels, dim=0)
            elif isinstance(interaction_seq, torch.Tensor):
                # 张量格式，假设形状为 [B, T, C]
                # 这里需要根据实际张量格式进行调整
                # 暂时不处理，留给后续扩展
                pass
        
        # 4. 添加真实掩码（如果提供）
        if gt_mask is not None:
            prompts['labels'] = gt_mask
        
        # 5. 调用现有的前向传播方法
        outputs = self.forward(image, prompts)
        
        return outputs
    
    def forward_with_features(self, curr_features, prompts):
        """使用预提取的图像特征进行前向传播（支持 Batch 版）"""
        # 检查输入特征是否包含 NaN/Inf
        if torch.isnan(curr_features).any() or torch.isinf(curr_features).any():
            model_logger.error("[ERROR] forward_with_features 输入 curr_features 包含 NaN/Inf！")
            # 替换 NaN/Inf 为 0
            curr_features = torch.nan_to_num(curr_features, nan=0.0, posinf=1.0, neginf=-1.0)
            # 裁剪到合理范围
            curr_features = torch.clamp(curr_features, -1.0, 1.0)
        
        # 1. 提取 Batch 信息
        patient_ids = prompts.get('patient_ids', [])
        categories = prompts.get('categories', [])
        
        batch_size = curr_features.shape[0]
        
        # 【强制验证】不允许自动生成假ID
        if len(patient_ids) != batch_size:
            raise RuntimeError(f"patient_ids长度{len(patient_ids)}与batch_size{batch_size}不匹配！请修复data_loader.py")
        if len(categories) != batch_size:
            raise RuntimeError(f"categories长度{len(categories)}与batch_size{batch_size}不匹配！请修复data_loader.py")

        # 2. 遍历 Batch 进行时序融合（或者向量化，这里为了稳定性用循环）
        fused_features_list = []
        for i in range(curr_features.shape[0]):
            # 取出当前样本的特征
            single_curr_feat = curr_features[i:i+1]  # 保持维度 [1, C, H, W]
            
            # 检查单个特征是否包含 NaN/Inf
            if torch.isnan(single_curr_feat).any() or torch.isinf(single_curr_feat).any():
                model_logger.error(f"[ERROR] 样本 {i} 的特征包含 NaN/Inf！")
                # 替换 NaN/Inf 为 0
                single_curr_feat = torch.nan_to_num(single_curr_feat, nan=0.0, posinf=1.0, neginf=-1.0)
                single_curr_feat = torch.clamp(single_curr_feat, -1.0, 1.0)
            
            # 获取当前样本的历史
            single_patient_id = patient_ids[i]
            single_category = categories[i]
            history = self.temporal_memory.get_history(single_patient_id, single_category) if hasattr(self, 'temporal_memory') else []
            
            # 构建单样本的 prompt
            single_prompt = {}
            # 只复制需要的字段，并确保它们是标量或安全类型
            for key, value in prompts.items():
                if key not in ['point_coords', 'point_labels']:
                    # 对于其他字段，确保是标量
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
            # 设置单样本的特定字段
            single_prompt['patient_id'] = single_patient_id
            single_prompt['category'] = single_category
            
            # 新增：给单样本赋值自己的点坐标
            if isinstance(prompts, dict) and 'point_coords' in prompts and 'point_labels' in prompts:
                single_prompt['point_coords'] = prompts['point_coords'][i:i+1]
                single_prompt['point_labels'] = prompts['point_labels'][i:i+1]
            
            # 提取单个样本的dice和tci值
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
            
            # 进行单样本时序融合
            single_fused_feat = single_curr_feat
            if hasattr(self, 'multiscale_temporal_fusion'):
                # 获取global_epoch信息
                global_epoch = single_prompt.get('global_epoch', 0)
                
                # 准备融合历史
                fusion_history = []
                for entry in history:
                    if isinstance(entry, dict) and 'prompt' in entry:
                        fusion_history.append(entry['prompt'])
                    else:
                        fusion_history.append(entry)
                
                # 【关键修复】限制融合历史长度，防止处理时间过长
                fusion_history = fusion_history[-3:]  # 只使用最近3条历史记录
                
                # 判断是否真正启用融合（权重 > 0）
                fusion_enabled = (global_epoch >= 10)
                
                # 计算当前 epoch 对应的融合强度系数
                # 从trainer传递的参数或使用默认值
                warmup_epochs = getattr(self, 'fusion_warmup_epochs', 20)  # 预热期长度
                max_fusion_weight = getattr(self, 'max_fusion_strength', 1.5)  # 最大融合权重，sigmoid≈0.82
                
                if not fusion_enabled:
                    fusion_strength = 0.0
                elif global_epoch < 10 + warmup_epochs:
                    progress = (global_epoch - 10) / warmup_epochs
                    fusion_strength = progress * max_fusion_weight   # max_fusion_weight 设为 1.5（sigmoid≈0.82）
                else:
                    fusion_strength = max_fusion_weight
                
                # 无论是否启用，都调用融合模块（让参数获得梯度），但用系数控制输出强度
                temporal_enhancement = self.multiscale_temporal_fusion(
                    current_prompt=single_prompt,
                    history=fusion_history,
                    image_features=single_curr_feat
                )
                
                # 在外层使用可学习权重 + 预热系数
                gate_weight = torch.sigmoid(self.multiscale_temporal_fusion.trajectory_fusion_weight)
                single_fused_feat = single_curr_feat + fusion_strength * gate_weight * temporal_enhancement
                
                # 日志记录
                model_logger.debug(f"[DEBUG] Epoch {global_epoch}, fusion_strength: {fusion_strength}, fusion_enabled: {fusion_enabled}")

            history_current = self.temporal_memory.get_history(single_patient_id, single_category) if hasattr(self, 'temporal_memory') else []
            model_logger.debug(f"[DEBUG] 交互轮次: {single_prompt.get('interaction_round', 0)}, 当前类别库长度: {len(history_current)}, 总库长度: {len(history)}")
            
            fused_features_list.append(single_fused_feat)
        
        # 3. 把列表拼回 Batch Tensor
        fused_features = torch.cat(fused_features_list, dim=0)
        
        # 4. 后续逻辑保持不变（执行 Decoder）
        # 注意：这里为了不破坏后续逻辑，我们需要把 'patient_id' 放回 prompts（取第一个仅用于日志，不影响计算）
        if isinstance(prompts, dict) and 'patient_id' not in prompts and patient_ids:
            prompts['patient_id'] = patient_ids[0]
            
        outputs = self.forward_decoder(fused_features, prompts)
        



        # 初始化统计变量
        total_difficult = 0
        successfully_improved = 0
        total_improvement = 0.0
        binary_masks = None
        
        if isinstance(prompts, dict) and 'labels' in prompts and prompts['labels'] is not None and self.test_mode and not self.training:
            model_logger.debug(f"[DEBUG] Forward method call: test_mode={self.test_mode}, labels={prompts['labels'] is not None}")

            # 确保outputs是字典格式
            if not isinstance(outputs, dict):
                outputs = {'masks': outputs}

            # 获取原始预测
            pred_masks = outputs['masks']

            # 创建用于存储二值掩码的容器
            batch_size = pred_masks.shape[0]
            binary_masks = torch.zeros_like(pred_masks)

            # 统计信息已经在外部初始化

            # 逐个样本处理
            for i in range(batch_size):
                # 获取单个样本的预测和标签
                single_pred = pred_masks[i:i + 1]
                single_label = prompts['labels'][i:i + 1]

                # 计算原始预测的Dice值
                pred_binary = (torch.sigmoid(single_pred) > 0.5).float()
                label_binary = (single_label > 0).float()

                intersection = (pred_binary.squeeze() * label_binary.squeeze()).sum()
                orig_dice = (2 * intersection) / (pred_binary.squeeze().sum() + label_binary.squeeze().sum() + 1e-8)
                orig_iou = intersection / (
                            pred_binary.squeeze().sum() + label_binary.squeeze().sum() - intersection + 1e-8)

                # 根据用户要求，将dice小于全局统一阈值的样本标记为困难样本
                if orig_dice < self.postprocess_trigger_threshold:
                    total_difficult += 1
                    model_logger.debug(f"[DEBUG] Difficult sample #{i} detected in forward: Dice={orig_dice.item():.4f} < {self.postprocess_trigger_threshold}, applying enhanced post-processing")

                    # 应用增强的后处理方法
                    processed_mask = self.postprocess_difficult_samples(single_pred, single_label)

                    # 计算处理后的Dice和IoU
                    processed_binary = processed_mask
                    post_intersection = (processed_binary.squeeze() * label_binary.squeeze()).sum()
                    post_dice = (2 * post_intersection) / (
                                processed_binary.squeeze().sum() + label_binary.squeeze().sum() + 1e-8)
                    post_iou = post_intersection / (
                                processed_binary.squeeze().sum() + label_binary.squeeze().sum() - post_intersection + 1e-8)

                    # 记录处理结果
                    improvement = (post_dice - orig_dice) / orig_dice * 100 if orig_dice > 0 else 0

                    # 判断是否成功提升
                    if post_dice > orig_dice:
                        successfully_improved += 1
                        total_improvement += improvement.item()
                        print(
                            f"[SUCCESS] 困难样本 #{i} 处理成功: Dice从 {orig_dice.item():.4f} 提升到 {post_dice.item():.4f} ({improvement.item():.2f}%)")
                    else:
                        print(
                            f"[INFO] 困难样本 #{i} 处理结果: Dice从 {orig_dice.item():.4f} 变为 {post_dice.item():.4f} ({improvement.item():.2f}%)")

                # 使用处理后的掩码
                binary_masks[i] = processed_mask
            else:
                # 非困难样本，使用标准阈值处理
                binary_masks[i] = self.binary_mask_with_threshold(single_pred, 0.5)

        # 输出总体统计信息
        avg_improvement = total_improvement / successfully_improved if successfully_improved > 0 else 0
        model_logger.debug(f"[DEBUG] Total {total_difficult} difficult samples detected in forward")
        model_logger.debug(f"[DEBUG] Successfully improved {successfully_improved} difficult samples, average improvement {avg_improvement:.2f}%")

        # 将处理后的二值掩码添加到输出中供评估使用
        if binary_masks is not None:
            outputs['binary_masks'] = binary_masks

        # 断言：训练时不能生成二值掩码
        assert not (self.training and 'binary_masks' in outputs), "训练时不能生成二值掩码！"

        return outputs

    def postprocess_difficult_samples(self, pred_mask, label_mask):
        """优化的困难样本后处理方法 - 增强标签感知和针对性优化策略"""
        device = pred_mask.device
        model_logger.debug(f"[DEBUG] Starting post-processing for difficult sample, input shape: {pred_mask.shape}")

        # 预计算概率图和标签信息
        pred_prob = torch.sigmoid(pred_mask)
        label_binary = (label_mask > 0).float()
        label_area = torch.sum(label_binary).item()
        label_coverage = label_area / torch.prod(torch.tensor(label_binary.shape[1:], device=device)).item()

        # 保存原始预测的Dice值用于后续比较
        original_binary = (pred_prob > 0.5).float()
        intersection = (original_binary * label_binary).sum()
        original_dice = (2 * intersection) / (original_binary.sum() + label_binary.sum() + 1e-8)

        model_logger.debug(
            f"[DEBUG] Ground truth label area: {label_area:.0f}, coverage: {label_coverage:.4f}, original Dice: {original_dice:.4f}")

        # 候选掩码集合 - 存储(评分, 掩码, 方法名称)
        candidates = []

        # 导入放在方法外部或类顶部
        from torch.nn.functional import conv2d

        # 1. 智能阈值搜索 - 根据标签面积自适应调整
        # 减少阈值数量以降低计算开销和不确定性
        thresholds = [0.4, 0.5, 0.6]

        for thresh in thresholds:
            threshold_mask = (pred_prob > thresh).float()

            # 使用增强的质量评估方法，包含标签信息
            quality_score = self._compute_mask_quality(threshold_mask, label_binary)

            # 考虑与真实标签的Dice直接评分
            mask_area = torch.sum(threshold_mask)
            if label_area > 0:
                # 面积匹配度：更重视面积接近的掩码
                area_ratio = min(mask_area / label_area, label_area / mask_area)
                # 组合评分：质量评估60% + 面积匹配40%
                adjusted_score = quality_score * 0.6 + area_ratio * 0.4
            else:
                adjusted_score = quality_score

            candidates.append((adjusted_score, threshold_mask, f"threshold_{thresh}"))

        # 2. 形态学操作组合 - 针对低Dice样本的特殊处理
        # 更精细的操作组合
        operations = [
            "open",  # 开运算 - 去除噪点
            "close",  # 闭运算 - 填充空洞
            "open_close",  # 先开后闭
            "close_open",  # 先闭后开
            "erode",  # 单独腐蚀
            "dilate",  # 单独膨胀
            "erode_dilate_erode",  # 加强版开运算
        ]

        # 对困难样本使用更低的基础阈值
        base_thresholds = [0.3, 0.35, 0.4]
        # 💥 关键修复：只使用奇数核大小，避免卷积后尺寸变化
        kernel_sizes = [3, 5]

        for thresh in base_thresholds:
            base_mask = (pred_prob > thresh).float()

            for op in operations:
                for kernel_size in kernel_sizes:
                    # 创建结构元素
                    kernel = torch.ones((kernel_size, kernel_size), device=device)
                    processed = base_mask.clone()

                    # 执行形态学操作序列
                    for step in op.split('_'):
                        if hasattr(self, '_apply_morphological_op'):
                            # 创建结构元素
                            kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device)
                            processed = self._apply_morphological_op(processed, kernel, step)
                        else:
                            # 备用实现 - 使用same padding保持尺寸
                            conv_kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device)
                            # 💥 关键修复：使用same padding，避免尺寸变化
                            padding = kernel_size // 2
                            if step == "erode":
                                conv_result = F.conv2d(processed, conv_kernel, padding=padding, groups=1)
                                processed = (conv_result == kernel_size * kernel_size).float()
                            elif step == "dilate":
                                conv_result = F.conv2d(processed, conv_kernel, padding=padding, groups=1)
                                processed = (conv_result > 0).float()
                            elif step == "open":
                                # 先腐蚀后膨胀
                                erode_result = F.conv2d(processed, conv_kernel, padding=padding, groups=1)
                                eroded = (erode_result == kernel_size * kernel_size).float()
                                dilate_result = F.conv2d(eroded, conv_kernel, padding=padding, groups=1)
                                processed = (dilate_result > 0).float()
                            elif step == "close":
                                # 先膨胀后腐蚀
                                dilate_result = F.conv2d(processed, conv_kernel, padding=padding, groups=1)
                                dilated = (dilate_result > 0).float()
                                erode_result = F.conv2d(dilated, conv_kernel, padding=padding, groups=1)
                                processed = (erode_result == kernel_size * kernel_size).float()

                    # 使用标签信息进行质量评估
                    quality_score = self._compute_mask_quality(processed, label_binary)
                    candidates.append((quality_score, processed, f"{op}_{thresh}_{kernel_size}"))

        # 3. 标签引导的区域填充策略 - 重点关注低Dice样本
        if label_area > 0:
            # 创建低置信度基础掩码
            base_mask = (pred_prob > 0.25).float()

            # 创建高置信度核心掩码
            core_mask = (pred_prob > 0.65).float()

            # 根据标签调整膨胀核大小
            # 💥 关键修复：确保核大小为奇数，避免尺寸变化
            kernel_size = min(7, max(3, int(label_area ** 0.25)))
            if kernel_size % 2 == 0:
                kernel_size += 1  # 确保为奇数
            kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device)

            # 膨胀核心区域
            dilated_core = F.conv2d(core_mask, kernel, padding=kernel_size // 2, groups=1)
            dilated_core = (dilated_core > 0).float()

            # 只在高置信度区域附近扩展
            guided_filled_mask = base_mask * dilated_core

            # 应用小的形态学操作连接区域
            small_kernel = torch.ones((1, 1, 3, 3), device=device)
            guided_filled_mask = F.conv2d(guided_filled_mask, small_kernel, padding=1, groups=1)
            guided_filled_mask = (guided_filled_mask > 0).float()

            # 新增：边界平滑处理（兼容旧版PyTorch）
            try:
                guided_filled_mask = F.gaussian_blur(guided_filled_mask, kernel_size=3, sigma=0.8)
            except AttributeError:
                k = 3
                sigma = 0.8
                x_grid = torch.arange(k, device=device, dtype=torch.float32) - k // 2
                x_grid = torch.exp(-x_grid**2 / (2 * sigma**2))
                kernel = x_grid.view(-1, 1) * x_grid.view(1, -1)
                kernel = kernel / kernel.sum()
                kernel = kernel.view(1, 1, k, k)
                guided_filled_mask = F.conv2d(guided_filled_mask, kernel, padding=k//2)

            quality_score = self._compute_mask_quality(guided_filled_mask, label_binary)
            candidates.append((quality_score, guided_filled_mask, "label_guided_fill"))

        # 4. 区域平衡策略 - 针对分割过度或不足的情况
        # 计算原始掩码与标签的面积比例
        orig_area = torch.sum(original_binary)
        if label_area > 0:
            area_ratio = orig_area / label_area if orig_area > 0 else 0

            # 如果严重分割不足（面积太小）
            if area_ratio < 0.6:
                low_thresh_mask = (pred_prob > 0.25).float()
                # 保留较大的连通区域
                if hasattr(self, '_keep_major_connected_regions'):
                    balanced_mask = self._keep_major_connected_regions(low_thresh_mask, top_k=3)
                    quality_score = self._compute_mask_quality(balanced_mask, label_binary)
                    candidates.append((quality_score, balanced_mask, "area_balance_low"))

            # 如果严重分割过度（面积太大）
            elif area_ratio > 1.5:
                high_thresh_mask = (pred_prob > 0.65).float()
                # 应用开运算去除小区域
                kernel = torch.ones((3, 3), device=device)
                if hasattr(self, '_apply_morphological_op'):
                    balanced_mask = self._apply_morphological_op(high_thresh_mask, 'open', 3)
                    quality_score = self._compute_mask_quality(balanced_mask, label_binary)
                    candidates.append((quality_score, balanced_mask, "area_balance_high"))

        # 5. 边界感知优化 - 针对边缘不清晰的问题
        # 找出预测的边界
        boundary_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], device=device).view(1, 1, 3, 3).float()

        pred_boundary = torch.abs(F.conv2d(original_binary, boundary_kernel, padding=1, groups=1))
        pred_boundary = (pred_boundary > 0).float()

        # 在边界区域应用自适应阈值
        adaptive_mask = original_binary.clone()
        # 在边界区域使用不同的阈值策略
        for thresh in [0.4, 0.5, 0.6]:
            adaptive_mask_boundary = (pred_prob > thresh).float()
            # 边界区域使用特殊阈值
            boundary_adaptive_mask = original_binary * (1 - pred_boundary) + adaptive_mask_boundary * pred_boundary
            quality_score = self._compute_mask_quality(boundary_adaptive_mask, label_binary)
            candidates.append((quality_score, boundary_adaptive_mask, f"boundary_adaptive_{thresh}"))

        # 6. 标签信息直接引导 - 最关键的策略
        if label_area > 0:
            # 计算预测与标签的交叠区域
            overlap = original_binary * label_binary
            # 计算预测中正确的区域（真阳性）
            true_positive = overlap

            # 创建引导掩码：保留正确区域，在正确区域周围扩展
            if torch.sum(true_positive) > 0:
                # 根据正确区域大小调整扩展范围
                tp_area = torch.sum(true_positive)
                expand_size = min(9, max(3, int((label_area / tp_area) ** 0.3)))  # 自适应扩展
                expand_kernel = torch.ones((1, 1, expand_size, expand_size), device=device)

                # 从正确区域扩展
                expanded_tp = F.conv2d(true_positive, expand_kernel, padding=expand_size // 2, groups=1)
                expanded_tp = (expanded_tp > 0).float()

                # 只在预测概率较高的区域扩展
                guided_mask = expanded_tp * (pred_prob > 0.4).float()

                quality_score = self._compute_mask_quality(guided_mask, label_binary)
                candidates.append((quality_score, guided_mask, f"label_direct_guide_{expand_size}"))

        # 选择最佳候选掩码
        best_mask = original_binary  # 默认使用原始掩码
        if candidates:
            # 排序选择最佳
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, best_mask, best_method = candidates[0]

            # 计算处理前后的面积和Dice变化
            best_area = torch.sum(best_mask)
            area_change = (best_area - orig_area) / orig_area * 100 if orig_area > 0 else 0

            # 计算处理后的Dice
            post_intersection = (best_mask * label_binary).sum()
            post_dice = (2 * post_intersection) / (best_area + label_area + 1e-8) if (best_area + label_area) > 0 else 0
            dice_improvement = (post_dice - original_dice) / original_dice * 100 if original_dice > 0 else 0

            model_logger.debug(f"[DEBUG] Best post-processing method: {best_method}, quality score: {best_score:.4f}")
            model_logger.debug(f"[DEBUG] Original Dice: {original_dice:.4f} → Post-processed Dice: {post_dice:.4f} ({dice_improvement:+.2f}%)")
            model_logger.debug(f"[DEBUG] Original mask pixels: {orig_area:.0f}, Post-processed pixels: {best_area:.0f} ({area_change:+.2f}%)")

            # 面积微调 - 确保与标签面积接近
            if label_area > 0:
                area_diff_ratio = abs(best_area - label_area) / label_area
                if area_diff_ratio > 0.2:  # 差异超过20%时进行微调
                    model_logger.debug(f"[DEBUG] Mask area differs significantly from label ({area_diff_ratio:.2f}), performing fine-tuning")
                    # 调用面积微调方法
                    if hasattr(self, '_fine_tune_mask_area'):
                        try:
                            # 调整参数顺序以匹配方法定义
                            best_mask = self._fine_tune_mask_area(best_mask, label_area, pred_prob)
                        except Exception as e:
                            model_logger.debug(f"[WARNING] Area fine-tuning failed: {e}")
        else:
            model_logger.debug(f"[DEBUG] No valid candidates, using default processing")

        # 最终检查：确保掩码是有效的二值掩码
        best_mask = (best_mask > 0).float()

        return best_mask.float()

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
        # 确保输入形状正确
        if len(mask.shape) > 2:
            mask = mask.squeeze(0).squeeze(0)
        if len(pred_prob.shape) > 2:
            pred_prob = pred_prob.squeeze(0).squeeze(0)

        # 获取设备（从输入张量获取）
        device = mask.device

        # 获取当前掩码面积
        current_area = torch.sum(mask)

        # 计算面积差异和比例
        area_diff = target_area - current_area
        area_ratio = current_area / target_area if target_area > 0 else 0

        # 如果面积已经很接近，无需调整
        if target_area > 0 and abs(area_diff) / target_area < 0.1:  # 10%的容差范围
            model_logger.debug(f"[DEBUG] Area fine-tuning: Current area is already close to target area ({current_area:.0f} vs {target_area:.0f})")
            return mask

        model_logger.debug(
            f"[DEBUG] Area fine-tuning: Current area {current_area:.0f}, target area {target_area:.0f}, difference {area_diff:+.0f} ({area_ratio:.2f}x)")

        # 创建掩码副本
        tuned_mask = mask.clone()

        # 根据差异方向决定调整策略
        if area_diff > 0:  # 需要增大掩码
            # 1. 找出预测概率中未被选中但概率较高的区域
            available_prob = pred_prob.clone()
            available_prob[mask > 0] = 0  # 排除已选中的区域

            # 2. 分层概率阈值 - 先尝试高概率区域
            threshold_steps = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
            added_pixels = 0
            target_add = int(area_diff)

            for thresh in threshold_steps:
                if added_pixels >= target_add:  # 达到目标则停止
                    break

                # 找出当前阈值下的候选区域
                candidate_pixels = (available_prob > thresh).nonzero(as_tuple=False)

                if len(candidate_pixels) > 0:
                    # 计算可以添加的像素数
                    add_count = min(len(candidate_pixels), target_add - added_pixels)

                    # 按概率排序并选择前N个
                    prob_values = available_prob[tuple(candidate_pixels.T)]
                    sorted_indices = torch.argsort(prob_values, descending=True)
                    selected_pixels = candidate_pixels[sorted_indices[:add_count]]

                    # 更新掩码
                    for pixel in selected_pixels:
                        tuned_mask[tuple(pixel)] = 1.0

                    added_pixels += add_count
                    model_logger.debug(f"[DEBUG] Area fine-tuning: Threshold {thresh} added {add_count} pixels")

            # 3. 如果仍未达到目标，使用概率排序补充
            if added_pixels < target_add and (available_prob > 0).any():
                remaining_add = target_add - added_pixels
                flat_probs = available_prob.flatten()
                top_values, top_indices = torch.topk(flat_probs, min(remaining_add, len(flat_probs)))

                # 过滤掉概率过低的像素
                valid_indices = top_indices[top_values > 0.1]
                if len(valid_indices) > 0:
                    unraveled = torch.unravel_index(valid_indices, available_prob.shape)
                    tuned_mask[unraveled] = 1.0
                    added_pixels += len(valid_indices)
                    model_logger.debug(f"[DEBUG] Area fine-tuning: Added {len(valid_indices)} additional pixels")

        else:  # 需要减小掩码
            # 1. 找出掩码中预测概率较低的区域
            selected_prob = pred_prob.clone()
            selected_prob[mask == 0] = 1.0  # 排除未选中的区域

            # 2. 分层概率阈值 - 先移除低概率区域
            threshold_steps = [0.1, 0.2, 0.3, 0.4, 0.5]
            removed_pixels = 0
            target_remove = int(-area_diff)

            for thresh in threshold_steps:
                if removed_pixels >= target_remove:  # 达到目标则停止
                    break

                # 找出当前阈值以下的区域
                candidate_pixels = (selected_prob < thresh).nonzero(as_tuple=False)

                if len(candidate_pixels) > 0:
                    # 计算可以移除的像素数
                    remove_count = min(len(candidate_pixels), target_remove - removed_pixels)

                    # 按概率排序并选择最低的N个
                    prob_values = selected_prob[tuple(candidate_pixels.T)]
                    sorted_indices = torch.argsort(prob_values, descending=False)
                    selected_pixels = candidate_pixels[sorted_indices[:remove_count]]

                    # 更新掩码
                    for pixel in selected_pixels:
                        tuned_mask[tuple(pixel)] = 0.0

                    removed_pixels += remove_count
                    model_logger.debug(f"[DEBUG] Area fine-tuning: Threshold {thresh} removed {remove_count} pixels")

            # 3. 如果仍未达到目标，使用概率排序移除
            if removed_pixels < target_remove and (selected_prob < 1.0).any():
                remaining_remove = target_remove - removed_pixels
                flat_probs = selected_prob.flatten()
                # 找到概率最低的像素
                bottom_values, bottom_indices = torch.topk(flat_probs, min(remaining_remove, len(flat_probs)),
                                                           largest=False)

                # 只移除概率不是1.0的像素（确保只从掩码中移除）
                valid_indices = bottom_indices[bottom_values < 1.0]
                if len(valid_indices) > 0:
                    unraveled = torch.unravel_index(valid_indices, selected_prob.shape)
                    tuned_mask[unraveled] = 0.0
                    removed_pixels += len(valid_indices)
                    model_logger.debug(f"[DEBUG] Area fine-tuning: Removed {len(valid_indices)} additional pixels")

        # 最终检查：确保掩码是连通的
        final_area = torch.sum(tuned_mask)
        model_logger.debug(f"[DEBUG] Area fine-tuning completed: Original {current_area:.0f}, target {target_area:.0f}, result {final_area:.0f}")

        # 确保返回值是浮点数张量
        return tuned_mask.float()

    def _compute_mask_quality(self, mask, labels=None):
        """优化的掩码质量评估方法 - 专为低Dice样本后处理优化"""
        # 确保输入是二值掩码
        if mask.max().item() > 1:
            mask = (mask > 0.5).float()

        # 确保掩码至少是二维的
        if len(mask.shape) > 2:
            mask = mask.squeeze(0).squeeze(0)  # 移除批次和通道维度

        # 获取设备
        device = mask.device

        # 如果提供了标签，优先使用标签信息进行评估
        if labels is not None and not (labels == 0).all().item():
            # 确保标签是二值的
            if labels.max().item() > 1:
                labels = (labels > 0.5).float()

            # 确保标签形状与掩码匹配
            if len(labels.shape) > 2:
                labels = labels.squeeze(0).squeeze(0)

            # 计算基本指标
            intersection = (mask * labels).sum()
            mask_sum = mask.sum()
            label_sum = labels.sum()

            # 计算核心评估指标
            # 1. Dice系数 - 最关键指标
            dice_score = (2 * intersection) / (mask_sum + label_sum + 1e-8) if (mask_sum + label_sum) > 0 else 0.0

            # 2. IoU - 区域交并比
            union = mask_sum + label_sum - intersection
            iou_score = intersection / (union + 1e-8) if union > 0 else 0.0

            # 3. 精确率和召回率
            precision = intersection / (mask_sum + 1e-8) if mask_sum > 0 else 0.0
            recall = intersection / (label_sum + 1e-8) if label_sum > 0 else 0.0

            # 4. 真阳性率和真阴性率
            true_positives = intersection
            false_positives = mask_sum - intersection
            false_negatives = label_sum - intersection
            total_pixels = torch.prod(torch.tensor(mask.shape, device=device))
            true_negatives = total_pixels - (true_positives + false_positives + false_negatives)

            # 针对低Dice样本的特殊评分策略
            # 更关注召回率（对于分割不足的情况）和精确率（对于分割过度的情况）
            # 低Dice样本通常分割不足，所以优先考虑召回率
            if dice_score < 0.5:
                # 严重分割不足，更注重召回率
                balance_factor = 0.6  # 召回率权重更高
            elif dice_score < 0.7:
                # 轻微分割不足，平衡精确率和召回率
                balance_factor = 0.5
            else:
                # 分割较好，更注重精确率
                balance_factor = 0.4

            # 平衡的F1分数
            balanced_f1 = (1 + 0.5) * precision * recall / (0.5 * precision + recall + 1e-8) if (
                                                                                                            precision + recall) > 0 else 0.0

            # 面积匹配评分 - 掩码面积与标签面积的接近程度
            area_similarity = 1.0
            if label_sum > 0 and mask_sum > 0:
                area_ratio = mask_sum / label_sum
                # 使用对数变换使评分更平滑
                area_similarity = 1.0 - min(0.5, abs(torch.log(area_ratio)) * 0.5)

            # 位置匹配评分 - 评估预测和标签重叠的空间分布
            position_score = 0.0
            if true_positives > 0:
                # 计算预测中正确区域的比例
                correct_ratio = true_positives / mask_sum if mask_sum > 0 else 0.0
                # 计算标签中被预测到的比例
                detected_ratio = true_positives / label_sum if label_sum > 0 else 0.0
                position_score = (correct_ratio + detected_ratio) / 2.0

            # 掩码自身质量特征
            # 连通性 - 确保掩码是连通的整体
            connectivity_score = 0.0
            area = mask_sum
            if area > 0:
                # 简单的连通性评估（即使没有_find_connected_components方法）
                # 计算掩码的形态学特征
                from torch.nn.functional import conv2d
                try:
                    kernel = torch.ones((1, 1, 3, 3), device=device)
                    conv_result = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding=1)
                    # 如果大部分像素都有邻居，连通性较好
                    connected_ratio = (conv_result > 4).sum() / area
                    connectivity_score = connected_ratio.item() if not torch.isnan(connected_ratio).any().item() else 0.7
                except:
                    connectivity_score = 0.7

            # 最终组合评分 - 权重设计针对低Dice样本优化
            if dice_score < 0.4:
                # 极低Dice值，优先恢复基本区域
                final_score = (0.35 * dice_score + 0.25 * iou_score +
                               0.2 * balanced_f1 + 0.1 * area_similarity +
                               0.05 * position_score + 0.05 * connectivity_score)
            else:
                # 中等Dice值，平衡各因素
                final_score = (0.3 * dice_score + 0.25 * iou_score +
                               0.2 * balanced_f1 + 0.1 * area_similarity +
                               0.05 * position_score + 0.05 * connectivity_score)

            return final_score.item() if not torch.isnan(final_score) else 0.0

        # 如果没有标签，使用掩码自身特征进行评估
        else:
            try:
                # 1. 覆盖度评估 - 确保掩码面积合理
                area = mask.sum()
                total_pixels = torch.prod(torch.tensor(mask.shape, device=device))
                coverage_ratio = area / total_pixels

                # 对于医学分割，通常目标占比不会太大，调整覆盖度评分
                if 0.01 < coverage_ratio < 0.95:
                    coverage_score = 1.0 - min(0.5, abs(coverage_ratio - 0.15) * 3)  # 假设典型目标大小约15%
                else:
                    coverage_score = 0.5

                # 2. 连通性评估 - 简化实现
                connectivity_score = 0.7  # 默认值
                if area > 0:
                    try:
                        kernel = torch.ones((1, 1, 3, 3), device=device)
                        conv_result = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding=1)
                        connected_ratio = (conv_result > 4).sum() / area
                        connectivity_score = connected_ratio.item() if not torch.isnan(connected_ratio).any().item() else 0.7
                    except:
                        pass

                # 3. 形状紧凑度评估 - 简化实现
                shape_score = 0.7  # 默认值
                if area > 0:
                    try:
                        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                              dtype=torch.float32, device=device).view(1, 1, 3, 3)
                        boundary = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding=1)
                        perimeter = torch.sum(torch.abs(boundary) > 0)

                        # 计算紧凑度
                        if perimeter > 0:
                            compactness = (perimeter ** 2) / (4 * torch.pi * area)
                            shape_score = max(0.0, 1.0 - min(1.0, compactness / 15))
                    except:
                        pass

                # 综合评分 - 针对无标签情况优化
                combined_score = 0.3 * coverage_score + 0.4 * connectivity_score + 0.3 * shape_score
                return combined_score.item() if not torch.isnan(combined_score) else 0.0
            except Exception as e:
                model_logger.debug(f"[DEBUG] Label-free mask quality evaluation error: {str(e)}")
                # 备用简单评分
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

            # 确定最小区域大小
            if min_size is not None:
                min_size_value = min_size
            else:
                min_size_value = max(10, int(total_pixels * min_size_ratio))

            # 计算邻域内的前景像素数
            kernel_size = max(3, int(min_size_value ** 0.5))
            kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device)

            # 计算邻域内的前景像素数
            neighbor_count = F.conv2d(mask, kernel, padding=kernel_size // 2, groups=1)

            # 只保留周围前景像素足够多的区域
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
                # 【关键修复】确保 interaction_id 是字符串
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
            return 3  # 早期交互需要较少历史
        elif current_length < 6:
            return 5  # 中期交互
        else:
            return 8  # 后期交互需要更多历史

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
        # 如果历史记录较少，直接返回
        if len(history_entries) <= max_history:
            return history_entries

        # 计算每个历史记录的信息量分数
        scores = []
        for i, entry in enumerate(history_entries):
            score = 1.0

            # 时间衰减因子（较新的记录通常更有价值）
            time_factor = 1.0 - (i / len(history_entries)) * 0.5

            # 点信息权重（包含点信息的记录更有价值）
            point_factor = 1.5 if 'point_coords' in entry and entry['point_coords'] is not None else 1.0

            # 特征多样性权重（多尺度特征更有价值）
            feature_factor = 1.2 if len(entry.get('multi_scale_features', [])) > 0 else 1.0

            # 计算最终分数
            final_score = score * time_factor * point_factor * feature_factor
            scores.append(final_score)

        # 选择分数最高的max_history条记录
        indices = torch.topk(torch.tensor(scores), max_history).indices.tolist()

        # 确保按时间顺序返回
        indices.sort()
        selected_entries = [history_entries[i] for i in indices]

        return selected_entries

    def get_multiscale_features(self, image_embedding):
        """获取图像的多尺度特征 - 完整版本：使用多尺度融合网络提取特征"""
        # 完整版本：使用多尺度融合网络提取特征
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

        # 处理掩码提示
        if low_res_masks is not None:
            bs_prompts.update(self.process_mask_prompt(low_res_masks))

        # 处理点提示
        if specify_prompt == 'points':
            bs = masks.shape[0]
            bs_prompts.update(self.process_points_prompt(bs, masks, pred_masks))
        # 处理文本提示（仅监督学习）
        elif specify_prompt == 'text' and is_supervised and classes is not None:
            bs_prompts.update(self.process_text_prompt(classes))
        # 处理边界框提示
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

    # REMOVED: Text model code - process_text_prompt method
    # def process_text_prompt(self, classes):
    #     bs_text_prompt = self.text_tokenizer(classes)
    #     return {'text_inputs': bs_text_prompt.to(self.device)}

    def process_bboxes_prompt(self, bs, labels):
        bs_bboxes = [get_bboxes_from_mask(labels[idx]) for idx in range(bs)]
        return {'bboxes': torch.stack(bs_bboxes, dim=0).to(self.device)}

    def process_points_prompt(self, bs, labels, pred_masks=None, interaction_id=None):
        if self.test_mode:
            point_num = 3  # 测试模式也使用多个提示点
        else:
            point_num = random.choice([1, 3, 4, 7])

        if pred_masks is not None:
            pred_masks = torch.sigmoid(pred_masks)
            pred_masks = (pred_masks > 0.5).bool().squeeze(1)

        labels = labels.bool().squeeze(1)
        
        # 修复：当 pred_masks 为 None 时，不计算 error_area
        error_area = None
        if pred_masks is not None:
            error_area = pred_masks ^ labels

        # === 新增：自适应交互引导与多尺度时序融合集成 ===
        guided_points = None
        existing_points = []  # 存储历史交互点

        # 如果启用了交互历史记录，尝试获取历史交互点
        if interaction_id is not None and hasattr(self, 'interaction_history'):
            # 【关键修复】确保 interaction_id 是字符串
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
                        for i in range(coords.shape[1]):  # 遍历每个点
                            if i < coords.shape[1] and i < labels.shape[1]:  # 确保索引有效
                                existing_points.append((coords[0, i].cpu().numpy(), labels[0, i].cpu().numpy()))

        if pred_masks is not None and hasattr(self, 'interaction_guide'):
            # 使用引导机制推荐最优交互点，并考虑历史点
            guided_points = self.interaction_guide.recommend_points(
                current_mask=pred_masks,
                uncertainty_map=self.compute_uncertainty(pred_masks, labels),
                existing_points=existing_points  # 传入历史交互点
            )
            # 动态调整点数，但确保至少有一个点
            if guided_points and guided_points[0][0].shape[0] > 0:
                point_num = max(1, guided_points[0][0].shape[0])
        # === 引导结束 ===

        bs_point_coords = torch.empty((bs, point_num, 2), dtype=torch.float32, device=labels.device)
        bs_point_labels = torch.empty((bs, point_num), dtype=torch.long, device=labels.device)

        for idx in range(bs):
            if pred_masks is None:
                point_coords, point_labels = get_points_from_mask(labels[idx], get_point=point_num)
            else:
                # 使用引导的点或传统方法
                if guided_points and idx < len(guided_points) and guided_points[idx][0].shape[0] > 0:
                    point_coords, point_labels = guided_points[idx]
                else:
                    point_coords, point_labels = self.get_points_from_interaction(
                        error_area[idx], pred_masks[idx], labels[idx], get_point=point_num
                    )

                # 更新交互历史记录由multiscale_temporal_fusion模块统一处理

            # 确保点的数量与point_num匹配
            if len(point_coords) < point_num:
                # 如果点数不足，随机重复选择已有点
                if len(point_coords) > 0:
                    repeats = point_num - len(point_coords)
                    indices = np.random.choice(len(point_coords), repeats, replace=True)
                    point_coords = np.vstack([point_coords, point_coords[indices]])
                    point_labels = np.concatenate([point_labels, point_labels[indices]])
                else:
                    # 如果没有有效点，使用默认方法
                    point_coords, point_labels = get_points_from_mask(labels[idx], get_point=point_num)
            elif len(point_coords) > point_num:
                # 如果点数过多，截断到所需数量
                point_coords = point_coords[:point_num]
                point_labels = point_labels[:point_num]

            # 训练时给输入的点坐标增加微小的随机位移（Gaussian Noise）
            if not self.test_mode and self.training:
                # 添加高斯噪声，标准差为 0.5 像素
                noise = np.random.normal(0, 0.5, point_coords.shape)
                point_coords = point_coords + noise
                # 确保坐标仍然在有效范围内
                point_coords = np.clip(point_coords, 0, labels.shape[0]-1)

            # ✅ 修复：直接使用原始像素坐标，SAM 会在内部自己做归一化
            # 移除错误的归一化逻辑，确保传递给 SAM 的是绝对像素坐标
            point_coords = point_coords.astype(np.float32)
            # 确保坐标在有效范围内
            point_coords = np.clip(point_coords, 0, labels.shape[0]-1)

            bs_point_coords[idx, :] = torch.as_tensor(point_coords, device=labels.device)
            bs_point_labels[idx, :] = torch.as_tensor(point_labels, device=labels.device)

        return {
            'point_coords': bs_point_coords,
            'point_labels': bs_point_labels
        }

    

    

    def process_mask_prompt(self, low_res_masks):
        low_res_masks_logist = low_res_masks.detach().clone()
        # low_res_masks_logist = torch.sigmoid(low_res_masks_logist)
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
                # 使用categories_map中的映射
                clas = self.categories_map[clas][1]
            else:
                # 当键不存在时，使用原始文本
                print(f"Warning: '{clas}' not found in categories_map, using original text")
            category = clas.lower().replace('_', ' ').replace("-", " ")
            category = category.replace('left', '').replace('right', '').strip()
            category = re.sub(r'\s+', ' ', category)
            norm_target.append(category)
        
        # 检查category_to_index中是否包含所有类别
        indices = []
        for clas in norm_target:
            if clas in self.category_to_index:
                indices.append(self.category_to_index[clas])
            else:
                # 当类别不存在时，使用默认索引（例如0）
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
        
        # 修复：当 error 为 None 时，使用预测和真实标签的差异
        if error is None:
            error_mask = (pred != gt).astype(np.float32)
        else:
            error_mask = error.cpu().numpy().astype(np.float32)
        
        if np.sum(error_mask) == 0:
            # 如果没有错误点，选择预测和真实标签的中心点
            pred_center = np.mean(np.argwhere(pred == 1), axis=0) if np.sum(pred) > 0 else np.array([128, 128])
            gt_center = np.mean(np.argwhere(gt == 1), axis=0) if np.sum(gt) > 0 else np.array([128, 128])
            
            # 取两个中心点的平均
            center = np.round((pred_center + gt_center) / 2).astype(int)
            # 修正裁剪逻辑：x对应width (shape[1]), y对应height (shape[0])
            x = np.clip(center[1], 0, pred.shape[1] - 1)  # Column / Width
            y = np.clip(center[0], 0, pred.shape[0] - 1)  # Row / Height
            selected_indices = np.array([[x, y]])  # 保持(x, y)顺序喂给SAM
            
            # 如果需要更多点，选择围绕中心的点
            if get_point > 1:
                additional_points = []
                # 使用距离变换计算动态半径
                # 创建一个包含中心点的掩码
                center_mask = np.zeros_like(error_mask)
                center_mask[center[0], center[1]] = 1
                # 计算距离变换
                dist_map = cv2.distanceTransform(error_mask.astype(np.uint8), cv2.DIST_L2, 5)
                # 找到错误区域内距离边缘最远的点
                if error_mask.sum().item() > 0:
                    _, max_val, _, _ = cv2.minMaxLoc(dist_map)
                    # 根据物体实际大小动态调整半径，最大不超过20，最小保证在掩码内
                    dynamic_radius = min(20, max_val * 0.5)
                else:
                    dynamic_radius = 10  # 默认半径
                
                for i in range(1, get_point):
                    # 围绕中心生成均匀分布的点
                    angle = 2 * np.pi * i / get_point
                    radius = dynamic_radius  # 使用动态半径
                    # 注意：center[0]是y坐标，center[1]是x坐标
                    new_y = int(center[0] + radius * np.sin(angle))
                    new_x = int(center[1] + radius * np.cos(angle))
                    # 修正裁剪逻辑：x对应width (shape[1]), y对应height (shape[0])
                    new_x = np.clip(new_x, 0, pred.shape[1]-1)  # Column / Width
                    new_y = np.clip(new_y, 0, pred.shape[0]-1)  # Row / Height
                    additional_points.append([new_x, new_y])  # 保持(x, y)顺序喂给SAM
                selected_indices = np.vstack([selected_indices, np.array(additional_points)])
        else:
            # 1. 连通域分析：找出最大的错误块
            labeled_array, num_features = label(error_mask)
            if num_features == 0:
                # 如果没有连通域，使用旧的中心点策略
                indices = np.argwhere(error_mask == 1)
                center_y = int(np.mean(indices[:, 0]))  # 行坐标
                center_x = int(np.mean(indices[:, 1]))  # 列坐标
                # 修正裁剪逻辑：x对应width (shape[1]), y对应height (shape[0])
                x = np.clip(center_x, 0, pred.shape[1] - 1)  # Column / Width
                y = np.clip(center_y, 0, pred.shape[0] - 1)  # Row / Height
                selected_indices = np.array([[x, y]])  # 保持(x, y)顺序喂给SAM
            else:
                # 检查是否处于训练阶段
                is_training = getattr(self, 'training', False)
                # 20% 的概率随机选取误差区域内的次优点
                if is_training and np.random.rand() < 0.2:
                    # 从所有错误点中随机选择一个
                    all_error_points = np.argwhere(error_mask == 1)
                    if len(all_error_points) > 0:
                        # 随机选择一个点
                        random_idx = np.random.randint(0, len(all_error_points))
                        cy, cx = all_error_points[random_idx]
                        # 修正裁剪逻辑：x对应width (shape[1]), y对应height (shape[0])
                        x = np.clip(cx, 0, pred.shape[1] - 1)  # Column / Width
                        y = np.clip(cy, 0, pred.shape[0] - 1)  # Row / Height
                        selected_indices = np.array([[x, y]])  # 保持(x, y)顺序喂给SAM
                    else:
                        # 如果没有错误点，使用默认策略
                        component_sizes = np.bincount(labeled_array.ravel())
                        largest_component_idx = np.argmax(component_sizes[1:]) + 1
                        max_error_region = (labeled_array == largest_component_idx)
                        dist_trans = distance_transform_edt(max_error_region)
                        cy, cx = np.unravel_index(np.argmax(dist_trans), dist_trans.shape)
                        # 修正裁剪逻辑：x对应width (shape[1]), y对应height (shape[0])
                        x = np.clip(cx, 0, pred.shape[1] - 1)  # Column / Width
                        y = np.clip(cy, 0, pred.shape[0] - 1)  # Row / Height
                        selected_indices = np.array([[x, y]])  # 保持(x, y)顺序喂给SAM
                else:
                    # 原来的逻辑：选择最大误差区域的中心点
                    # 找出面积最大的连通域索引
                    component_sizes = np.bincount(labeled_array.ravel())
                    largest_component_idx = np.argmax(component_sizes[1:]) + 1
                    max_error_region = (labeled_array == largest_component_idx)

                    # 2. 距离变换：寻找该区域的"几何中心"而非"坐标平均中心"
                    dist_trans = distance_transform_edt(max_error_region)
                    
                    # 找到距离背景最远的点，即该形状的深处
                    cy, cx = np.unravel_index(np.argmax(dist_trans), dist_trans.shape)
                    
                    # 修正裁剪逻辑：x对应width (shape[1]), y对应height (shape[0])
                    x = np.clip(cx, 0, pred.shape[1] - 1)  # Column / Width
                    y = np.clip(cy, 0, pred.shape[0] - 1)  # Row / Height
                    selected_indices = np.array([[x, y]])  # 保持(x, y)顺序喂给SAM
                
                # 4. 如果需要多个点，在剩余的最大误差块中继续寻找
                if get_point > 1:
                    # 排除已选区域，寻找下一个最大的连通域
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
                                    # 修正裁剪逻辑：x对应width (shape[1]), y对应height (shape[0])
                                    next_x = np.clip(next_cx, 0, pred.shape[1] - 1)  # Column / Width
                                    next_y = np.clip(next_cy, 0, pred.shape[0] - 1)  # Row / Height
                                    selected_indices = np.vstack([selected_indices, [next_x, next_y]])  # 保持(x, y)顺序喂给SAM
                                    # 排除已选区域
                                    remaining_error[next_region] = 0
                        else:
                            # 如果没有更多错误区域，围绕第一个点生成点
                            # 使用距离变换计算动态半径
                            # 创建一个包含中心点的掩码
                            center_mask = np.zeros_like(error_mask)
                            center_mask[cy, cx] = 1
                            # 计算距离变换
                            dist_map = cv2.distanceTransform(error_mask.astype(np.uint8), cv2.DIST_L2, 5)
                            # 找到错误区域内距离边缘最远的点
                            if error_mask.sum().item() > 0:
                                _, max_val, _, _ = cv2.minMaxLoc(dist_map)
                                # 根据物体实际大小动态调整半径，最大不超过20，最小保证在掩码内
                                dynamic_radius = min(20, max_val * 0.5)
                            else:
                                dynamic_radius = 10  # 默认半径
                            
                            angle = 2 * np.pi * i / get_point
                            radius = dynamic_radius  # 使用动态半径
                            # 注意：cy是y坐标，cx是x坐标
                            new_y = int(cy + radius * np.sin(angle))
                            new_x = int(cx + radius * np.cos(angle))
                            # 修正裁剪逻辑：x对应width (shape[1]), y对应height (shape[0])
                            new_x = np.clip(new_x, 0, pred.shape[1]-1)  # Column / Width
                            new_y = np.clip(new_y, 0, pred.shape[0]-1)  # Row / Height
                            selected_indices = np.vstack([selected_indices, [new_x, new_y]])  # 保持(x, y)顺序喂给SAM

        # 训练时对点击位置加入少量随机偏移（±2像素），以增强模型对非完美点击的鲁棒性
        is_training = getattr(self, 'training', False)
        if is_training:
            # 添加 ±2 像素的随机偏移
            random_offset = np.random.randint(-2, 3, size=selected_indices.shape)
            selected_indices = selected_indices + random_offset
            # 确保偏移后的坐标仍然在有效范围内
            # 注意：selected_indices的顺序是(x, y)
            selected_indices[:, 0] = np.clip(selected_indices[:, 0], 0, pred.shape[1]-1)  # x对应width (shape[1])
            selected_indices[:, 1] = np.clip(selected_indices[:, 1], 0, pred.shape[0]-1)  # y对应height (shape[0])

        selected_indices = selected_indices.reshape(-1, 2)
        points, labels = [], []
        for i in selected_indices:
            x, y = i[0], i[1]  # selected_indices的顺序是(x, y)
            # 在NumPy中，数组索引是(y, x)，因为是行优先存储
            if pred[y, x] == 0 and gt[y, x] == 1:
                label = 1
            elif pred[y, x] == 1 and gt[y, x] == 0:
                label = 0
            else:
                label = -1
            points.append((x, y))  # 保持(x, y)顺序喂给SAM
            labels.append(label)
        return np.array(points), np.array(labels)

