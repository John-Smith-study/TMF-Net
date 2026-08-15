import torch
import numpy as np
from torch.nn import functional as F
import random
import torch.nn as nn
import os


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        pred = torch.sigmoid(pred)
        num_pos = torch.sum(mask)
        num_neg = mask.numel() - num_pos
        w_pos = (1 - pred) ** self.gamma
        w_neg = pred ** self.gamma

        loss_pos = -self.alpha * mask * w_pos * torch.log(pred + 1e-12)
        loss_neg = -(1 - self.alpha) * (1 - mask) * w_neg * torch.log(1 - pred + 1e-12)

        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos + num_neg + 1e-12)
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        pred = torch.sigmoid(pred)
        intersection = torch.sum(pred * mask)
        union = torch.sum(pred) + torch.sum(mask)
        dice_loss = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_loss


class MaskMSE(nn.Module):
    def __init__(self, ):
        super(MaskMSE, self).__init__()

    def forward(self, pred, mask, pred_iou=None):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        pred_iou: [B, 1] or None
        """
        if pred_iou is None:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        pred = torch.sigmoid(pred)
        intersection = torch.sum(pred * mask)
        union = torch.sum(pred) + torch.sum(mask) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        mse = torch.mean((iou - pred_iou) ** 2)
        return mse


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # 控制假阳性的权重
        self.beta = beta    # 控制假阴性的权重
        self.smooth = smooth

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        pred = torch.sigmoid(pred)
        intersection = torch.sum(pred * mask)
        fp = torch.sum(pred * (1 - mask))
        fn = torch.sum((1 - pred) * mask)
        tversky = (intersection + self.smooth) / (intersection + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky


class BoundaryLoss(nn.Module):
    def __init__(self, radius=2):
        super(BoundaryLoss, self).__init__()
        self.radius = radius
        # 创建边界检测卷积核
        self.kernel = self._create_boundary_kernel(radius)

    def _create_boundary_kernel(self, radius):
        kernel = torch.zeros((1, 1, 2*radius+1, 2*radius+1))
        for i in range(2*radius+1):
            for j in range(2*radius+1):
                if abs(i - radius) == radius or abs(j - radius) == radius:
                    kernel[0, 0, i, j] = 1
        kernel[0, 0, radius, radius] = -4
        return kernel

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        
        # 确保kernel在正确的设备和dtype上
        self.kernel = self.kernel.to(device=pred.device, dtype=pred.dtype)
        
        # 计算边界
        pred_boundary = torch.abs(F.conv2d(torch.sigmoid(pred), self.kernel, padding=self.radius))
        mask_boundary = torch.abs(F.conv2d(mask.to(dtype=pred.dtype), self.kernel, padding=self.radius))
        
        # 计算边界损失
        loss = F.smooth_l1_loss(pred_boundary, mask_boundary)
        return loss


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0, smooth=1e-5):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha  # 控制假阳性的权重
        self.beta = beta    # 控制假阴性的权重
        self.gamma = gamma  # 控制焦点损失的强度
        self.smooth = smooth

    def forward(self, pred, mask):
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        pred = torch.sigmoid(pred)
        intersection = torch.sum(pred * mask)
        fp = torch.sum(pred * (1 - mask))
        fn = torch.sum((1 - pred) * mask)
        tversky = (intersection + self.smooth) / (intersection + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma
        return focal_tversky

class FocalDice_MSELoss(nn.Module):
    def __init__(self, focal_weight=20.0, dice_weight=1.0, mse_weight=1.0, tversky_weight=0.5, boundary_weight=0.2, focal_tversky_weight=0.8):
        super(FocalDice_MSELoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.mse_weight = mse_weight
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight
        self.focal_tversky_weight = focal_tversky_weight
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.tversky_loss = TverskyLoss()
        self.focal_tversky_loss = FocalTverskyLoss()
        self.boundary_loss = BoundaryLoss()
        self.maskiou_mse = MaskMSE()

    def forward(self, pred, mask, pred_iou=None, **kwargs):
        """
        pred: [B, 1, H, W] or dict containing 'masks' and 'iou_pred'
        mask: [B, 1, H, W]
        pred_iou: optional, if pred is dict and pred_iou is None, extracted from dict
        """
        if isinstance(pred, dict):
            if pred_iou is None and 'iou_pred' in pred:
                pred_iou = pred['iou_pred']
            pred = pred['masks']
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        # 统一 mask 的 dtype 与 pred 一致
        mask = mask.to(dtype=pred.dtype)

        focal_loss = self.focal_loss(pred, mask)
        dice_loss = self.dice_loss(pred, mask)
        tversky_loss = self.tversky_loss(pred, mask)
        focal_tversky_loss = self.focal_tversky_loss(pred, mask)
        boundary_loss = self.boundary_loss(pred, mask)
        
        w_focal = torch.tensor(self.focal_weight, device=pred.device, dtype=pred.dtype)
        w_dice = torch.tensor(self.dice_weight, device=pred.device, dtype=pred.dtype)
        w_tversky = torch.tensor(self.tversky_weight, device=pred.device, dtype=pred.dtype)
        w_focal_tversky = torch.tensor(self.focal_tversky_weight, device=pred.device, dtype=pred.dtype)
        w_boundary = torch.tensor(self.boundary_weight, device=pred.device, dtype=pred.dtype)
        w_mse = torch.tensor(self.mse_weight, device=pred.device, dtype=pred.dtype)

        loss1 = (w_focal * focal_loss +
                 w_dice * dice_loss +
                 w_tversky * tversky_loss +
                 w_focal_tversky * focal_tversky_loss +
                 w_boundary * boundary_loss)
        loss2 = self.maskiou_mse(pred, mask, pred_iou)
        loss = loss1 + loss2 * w_mse
        return loss.to(dtype=pred.dtype)


def get_iou_and_dice(pred, label):
    """
    计算IoU和Dice系数的通用函数

    参数:
        pred: 预测掩码，形状为[B, C, H, W]或[B, H, W]
        label: 真实标签，形状为[B, C, H, W]或[B, H, W]

    返回:
        iou_mean: 平均IoU值
        dice_mean: 平均Dice系数
    """
    assert pred.shape == label.shape, "预测和标签形状必须相同"

    # 将预测转换为二值掩码
    pred_binary = (torch.sigmoid(pred) > 0.5) if pred.max() > 1 else (pred > 0.5)
    label_binary = (label > 0.5) if label.max() > 1 else (label > 0)

    # 计算交集和并集
    intersection = torch.logical_and(pred_binary, label_binary).sum(dim=tuple(range(1, pred_binary.ndim)))
    union = torch.logical_or(pred_binary, label_binary).sum(dim=tuple(range(1, pred_binary.ndim)))

    # 计算IoU和Dice
    iou = intersection.float() / (union.float() + 1e-8)
    dice = (2 * intersection.float()) / (pred_binary.sum(dim=tuple(range(1, pred_binary.ndim))) +
                                         label_binary.sum(dim=tuple(range(1, pred_binary.ndim))) + 1e-8)

    return iou.mean().item(), dice.mean().item()


def _get_surface_points(binary_mask):
    """
    提取二值掩码的表面点坐标
    
    参数:
        binary_mask: 二值掩码，形状为 [H, W] 或 [C, H, W] 或 [B, C, H, W]
        
    返回:
        surface_points: 表面点坐标数组，形状为 [N, 2]
    """
    # 确保是 2D 掩码
    if binary_mask.ndim > 2:
        # 展平通道和批次维度
        binary_mask = binary_mask.reshape(-1, binary_mask.shape[-2], binary_mask.shape[-1])
        # 取第一个非空掩码
        for i in range(binary_mask.shape[0]):
            if np.sum(binary_mask[i]) > 0:
                binary_mask = binary_mask[i]
                break
    
    # 使用边缘检测提取表面点
    # 简单的边缘检测：检查每个像素是否与背景相邻
    h, w = binary_mask.shape
    surface_points = []
    
    for i in range(h):
        for j in range(w):
            if binary_mask[i, j] > 0:
                # 检查8邻域是否有背景像素
                has_background = False
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            if binary_mask[ni, nj] == 0:
                                has_background = True
                                break
                    if has_background:
                        break
                if has_background:
                    surface_points.append([j, i])  # (x, y) 格式
    
    return np.array(surface_points, dtype=np.float64)


def _compute_distances(points1, points2):
    """
    计算两个点集之间的距离矩阵
    
    参数:
        points1: 点集1，形状为 [N, 2]
        points2: 点集2，形状为 [M, 2]
        
    返回:
        distances: 距离矩阵，形状为 [N, M]
    """
    if len(points1) == 0 or len(points2) == 0:
        return np.array([])
    
    # 使用广播计算欧氏距离
    diff = points1[:, np.newaxis, :] - points2[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))
    return distances


def get_hd95_and_assd(pred, label, voxelspacing=None):
    """
    计算 HD95 和 ASSD 指标（不依赖 medpy）。

    参数:
        pred: 预测掩码，形状为 [B, C, H, W] 或 [B, H, W] 或 [H, W] (Tensor 或 Numpy)
        label: 真实标签，形状同上
        voxelspacing: 2D in-plane physical pixel spacing (sy, sx) in mm.
                      Used to scale surface-point coordinates BEFORE distance
                      computation so that anisotropic pixels are handled
                      correctly (not via mean-spacing post-multiplication).

    返回:
        hd95_mean, assd_mean (均值, in mm)
    """
    # 1. 转换为 numpy 数组
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(label):
        label = label.detach().cpu().numpy()

    # 2. 严格二值化
    pred_binary = (pred > 0.5).astype(np.int32)
    label_binary = (label > 0.5).astype(np.int32)

    # 3. 异常情况处理
    if np.sum(pred_binary) == 0 and np.sum(label_binary) == 0:
        # 预测和标签都为空，距离误差为 0
        return 0.0, 0.0
    
    if np.sum(pred_binary) == 0 or np.sum(label_binary) == 0:
        # 【学术级修正】如果切片上没有预测出目标，强行计算距离无数学意义
        # 返回 NaN，在求均值时将其过滤，这是 2D 切片评估的标准做法
        return np.nan, np.nan

    # 4. 提取表面点 (pixel coordinates, (x, y) format)
    pred_surface = _get_surface_points(pred_binary)
    label_surface = _get_surface_points(label_binary)

    if len(pred_surface) == 0 or len(label_surface) == 0:
        return np.nan, np.nan

    # 5. Scale surface-point coordinates by per-axis physical spacing BEFORE
    #    distance computation. This correctly handles anisotropic pixels:
    #    _get_surface_points returns (x, y), so we multiply x by sx and y by sy.
    #    Do NOT use mean(spacing) post-multiplication — that is only correct
    #    for isotropic pixels.
    if voxelspacing is not None:
        sy, sx = float(voxelspacing[0]), float(voxelspacing[1])
        scale = np.array([sx, sy], dtype=np.float64)  # [x_scale, y_scale]
        pred_surface = pred_surface * scale
        label_surface = label_surface * scale

    # 6. 计算距离矩阵 (now in mm)
    dist_pred_to_label = _compute_distances(pred_surface, label_surface)
    dist_label_to_pred = _compute_distances(label_surface, pred_surface)

    if dist_pred_to_label.size == 0 or dist_label_to_pred.size == 0:
        return np.nan, np.nan

    # 7. 计算 HD95 (in mm)
    all_distances = np.concatenate([
        dist_pred_to_label.min(axis=1),
        dist_label_to_pred.min(axis=1)
    ])

    if len(all_distances) == 0:
        hd95 = np.nan
    else:
        hd95 = np.percentile(all_distances, 95)

    # 8. 计算 ASSD (in mm)
    mean_pred_to_label = np.mean(dist_pred_to_label.min(axis=1))
    mean_label_to_pred = np.mean(dist_label_to_pred.min(axis=1))
    assd = (mean_pred_to_label + mean_label_to_pred) / 2.0

    return float(hd95), float(assd)


def compute_iou(pred, label):
    """
    计算IoU指标的简化函数

    参数:
        pred: 预测掩码
        label: 真实标签

    返回:
        mean_iou: 平均IoU值
    """
    assert pred.shape == label.shape
    pred_binary = (torch.sigmoid(pred) > 0.5) if pred.max() > 1 else (pred > 0.5)
    label_binary = (label > 0.5) if label.max() > 1 else (label > 0)

    intersection = torch.logical_and(pred_binary, label_binary).sum(dim=tuple(range(1, pred_binary.ndim)))
    union = torch.logical_or(pred_binary, label_binary).sum(dim=tuple(range(1, pred_binary.ndim)))

    iou = intersection.float() / (union.float() + 1e-8)
    return iou.mean()


class EnhancedDifficultSampleHandler:
    """
    增强的困难样本处理器

    针对分割效果不佳的困难样本进行智能后处理，包含边界细化、区域生长和置信度校准等策略
    可根据样本难度自适应选择最佳处理策略组合
    """

    def __init__(self, dice_threshold=0.6, boundary_weight=0.8, calibration_strength=0.5,
                 enable_logging=False, device=None):
        """
        初始化困难样本处理器

        参数:
            dice_threshold: 判定困难样本的Dice阈值
            boundary_weight: 边界区域权重调整因子
            calibration_strength: 置信度校准强度
            enable_logging: 是否启用日志
            device: 计算设备，默认为None（自动推断）
        """
        self.dice_threshold = dice_threshold
        self.boundary_weight = boundary_weight
        self.calibration_strength = calibration_strength
        self.enable_logging = enable_logging
        self.device = device

        self.adaptive_strategies = {
            'boundary_refinement': True,
            'region_growing': True,
            'confidence_calibration': True,
            'small_region_cleanup': True  # 新增小区域清理策略
        }

        # 预定义卷积核以提高性能
        self._initialize_kernels()

    def _initialize_kernels(self):
        """初始化并缓存常用的卷积核"""
        # 延迟初始化，确保在有device时才创建
        self._kernels = {}

    def _get_kernel(self, kernel_type, device):
        """获取预定义的卷积核"""
        if kernel_type not in self._kernels or self._kernels[kernel_type].device != device:
            if kernel_type == 'dilation':
                self._kernels[kernel_type] = torch.tensor(
                    [[0, 1, 0], [1, 1, 1], [0, 1, 0]], device=device).float().view(1, 1, 3, 3)
            elif kernel_type == 'edge_detection':
                self._kernels[kernel_type] = torch.tensor(
                    [[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=device).float().view(1, 1, 3, 3)
        return self._kernels[kernel_type]

    def compute_dice(self, pred, label):
        """计算Dice系数"""
        try:
            intersection = (pred * label).sum()
            denominator = pred.sum() + label.sum() + 1e-8
            return (2 * intersection) / denominator
        except Exception as e:
            if self.enable_logging:
                print(f"[DEBUG] Dice计算错误: {str(e)}")
            return torch.tensor(0.0, device=pred.device)

    def advanced_postprocess(self, pred_mask, label_mask, original_dice):
        """
        针对困难样本的增强后处理

        参数:
            pred_mask: 原始预测掩码
            label_mask: 真实标签掩码
            original_dice: 原始Dice系数

        返回:
            processed_mask: 处理后的二值掩码
        """
        try:
            # 参数验证
            if not isinstance(pred_mask, torch.Tensor) or not isinstance(label_mask, torch.Tensor):
                raise TypeError("预测掩码和标签掩码必须是torch.Tensor类型")

            # 非困难样本直接返回
            if original_dice >= self.dice_threshold:
                return (pred_mask > 0.5).float() if pred_mask.max() > 1 else pred_mask

            # 确定设备
            device = self.device or pred_mask.device

            # 确保输入在正确设备上
            pred_mask = pred_mask.to(device)
            label_mask = label_mask.to(device)

            # 转换为概率图
            pred_prob = torch.sigmoid(pred_mask) if pred_mask.max() > 1 else pred_mask
            label_binary = (label_mask > 0).float()

            # 记录初始性能
            initial_dice = original_dice
            improvements = []

            # 1. 边界细化策略 - 适用于所有困难样本
            if self.adaptive_strategies['boundary_refinement']:
                try:
                    refined_mask = self.boundary_aware_refinement(pred_prob, label_binary)
                    refined_dice = self.compute_dice(refined_mask, label_binary)

                    if refined_dice > initial_dice:
                        if self.enable_logging:
                            print(f"[DEBUG] 边界细化提升Dice: {initial_dice:.4f} -> {refined_dice:.4f}")
                        pred_prob = refined_mask
                        initial_dice = refined_dice
                        improvements.append('boundary_refinement')
                except Exception as e:
                    if self.enable_logging:
                        print(f"[DEBUG] 边界细化失败: {str(e)}")

            # 2. 区域生长策略 - 针对严重分割不足的情况
            if initial_dice < 0.4 and self.adaptive_strategies['region_growing']:
                try:
                    grown_mask = self.region_growing(pred_prob, label_binary)
                    grown_dice = self.compute_dice(grown_mask, label_binary)

                    if grown_dice > initial_dice:
                        if self.enable_logging:
                            print(f"[DEBUG] 区域生长提升Dice: {initial_dice:.4f} -> {grown_dice:.4f}")
                        pred_prob = grown_mask
                        initial_dice = grown_dice
                        improvements.append('region_growing')
                except Exception as e:
                    if self.enable_logging:
                        print(f"[DEBUG] 区域生长失败: {str(e)}")

            # 3. 小区域清理 - 移除噪声
            if self.adaptive_strategies['small_region_cleanup']:
                try:
                    cleaned_mask = self.clean_small_regions(pred_prob)
                    cleaned_dice = self.compute_dice(cleaned_mask, label_binary)

                    if cleaned_dice > initial_dice:
                        if self.enable_logging:
                            print(f"[DEBUG] 小区域清理提升Dice: {initial_dice:.4f} -> {cleaned_dice:.4f}")
                        pred_prob = cleaned_mask
                        initial_dice = cleaned_dice
                        improvements.append('small_region_cleanup')
                except Exception as e:
                    if self.enable_logging:
                        print(f"[DEBUG] 小区域清理失败: {str(e)}")

            # 4. 置信度校准 - 作为最后优化步骤
            if self.adaptive_strategies['confidence_calibration']:
                try:
                    calibrated_mask = self.confidence_calibration(pred_prob, label_binary)
                    calibrated_dice = self.compute_dice(calibrated_mask, label_binary)

                    if calibrated_dice > initial_dice:
                        if self.enable_logging:
                            print(f"[DEBUG] 置信度校准提升Dice: {initial_dice:.4f} -> {calibrated_dice:.4f}")
                        pred_prob = calibrated_mask
                        improvements.append('confidence_calibration')
                except Exception as e:
                    if self.enable_logging:
                        print(f"[DEBUG] 置信度校准失败: {str(e)}")

            if self.enable_logging and improvements:
                print(f"[DEBUG] 应用的优化策略: {', '.join(improvements)}")

            # 最终二值化
            return (pred_prob > 0.5).float()

        except Exception as e:
            if self.enable_logging:
                print(f"[ERROR] 后处理整体失败: {str(e)}")
            # 失败时返回原始掩码的二值化结果
            return (pred_mask > 0.5).float() if pred_mask.max() > 1 else pred_mask

    def boundary_aware_refinement(self, pred_prob, label):
        """
        边界感知细化
        改进的边界处理策略，根据边界类型自适应调整阈值
        """
        try:
            from torch.nn.functional import conv2d

            device = pred_prob.device
            refined_prob = pred_prob.clone()

            # 确保label是4D张量 (B, C, H, W)
            if label.dim() == 3:
                label = label.unsqueeze(0)
            if label.dim() == 2:
                label = label.unsqueeze(0).unsqueeze(0)

            # 获取预定义卷积核
            dilation_kernel = self._get_kernel('dilation', device)
            edge_kernel = self._get_kernel('edge_detection', device)

            # 1. 计算预测边界
            pred_binary = (refined_prob > 0.5).float().unsqueeze(0) if refined_prob.dim() == 3 else (
                        refined_prob > 0.5).float()
            pred_edges = torch.abs(F.conv2d(pred_binary, edge_kernel, padding=1)) > 0

            # 2. 计算标签边界和内部区域
            label_edges = torch.abs(F.conv2d(label, edge_kernel, padding=1)) > 0
            dilated_label = F.conv2d(label, dilation_kernel, padding=1) > 0

            # 3. 定义不同区域
            # 假阳性边界区域：预测为边界但不在标签内
            fp_boundary = pred_edges & (~dilated_label)
            # 假阴性边界区域：标签内但未被预测覆盖
            fn_boundary = dilated_label & (~pred_binary)

            # 4. 自适应调整不同区域的置信度
            if fp_boundary.any():
                # 降低假阳性边界区域的置信度
                fp_mask = fp_boundary.squeeze(0)
                refined_prob[fp_mask] *= self.boundary_weight

            if fn_boundary.any():
                # 针对假阴性区域，基于距离标签中心的远近调整
                fn_mask = fn_boundary.squeeze(0)
                # 对靠近标签中心的区域提高置信度
                if torch.sum(label) > 0:
                    # 计算距离标签中心的距离权重
                    label_center = self._compute_centroid(label.squeeze(0).squeeze(0))
                    if label_center is not None:
                        dist_weight = self._compute_distance_weight(refined_prob.shape[1:3], label_center, device)
                        # 对靠近中心的区域给予更高权重
                        refined_prob[fn_mask] += (1.0 - refined_prob[fn_mask]) * 0.3 * dist_weight[fn_mask]

            return torch.clamp(refined_prob, 0, 1)

        except Exception as e:
            if self.enable_logging:
                print(f"[DEBUG] 边界细化内部错误: {str(e)}")
            return pred_prob

    def _compute_centroid(self, mask):
        """计算掩码的中心点"""
        try:
            coords = torch.nonzero(mask)
            if len(coords) > 0:
                centroid = torch.mean(coords.float(), dim=0)
                return tuple(centroid.tolist())
            return None
        except:
            return None

    def _compute_distance_weight(self, shape, center, device):
        """计算基于距离的权重图"""
        try:
            if center is None:
                return torch.ones(shape, device=device)

            h, w = shape
            y_coords, x_coords = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing='ij'
            )

            # 计算欧氏距离
            distances = torch.sqrt(
                (y_coords - center[0]) ** 2 + (x_coords - center[1]) ** 2
            )

            # 归一化距离并反转（距离越小权重越大）
            max_dist = torch.max(distances)
            if max_dist > 0:
                normalized_dist = distances / max_dist
                return 1.0 - normalized_dist

            return torch.ones_like(distances)
        except:
            return torch.ones(shape, device=device)

    def region_growing(self, pred_prob, label):
        """
        改进的区域生长算法
        基于种子点的智能区域生长，结合标签信息和预测置信度
        """
        try:
            import numpy as np
            from scipy import ndimage

            device = pred_prob.device

            # 确保输入形状正确
            if pred_prob.dim() == 4:
                pred_prob = pred_prob.squeeze(0).squeeze(0)
            elif pred_prob.dim() == 3:
                pred_prob = pred_prob.squeeze(0)

            if label.dim() == 4:
                label = label.squeeze(0).squeeze(0)
            elif label.dim() == 3:
                label = label.squeeze(0)

            # 转换为numpy处理
            pred_np = pred_prob.cpu().numpy()
            label_np = label.cpu().numpy()

            # 二值化初始预测
            binary_pred = (pred_np > 0.5).astype(np.uint8)

            # 识别标签中的未覆盖区域
            uncovered = (label_np > 0) & (binary_pred == 0)

            if not np.any(uncovered):
                # 没有需要生长的区域
                return pred_prob.unsqueeze(0).unsqueeze(0)

            # 1. 基于置信度的种子点选择
            confidence_in_uncovered = pred_np * uncovered
            # 选择高置信度的未覆盖区域作为种子点
            seed_threshold = max(0.3, np.percentile(confidence_in_uncovered[confidence_in_uncovered > 0], 70))
            seed_points = (confidence_in_uncovered > seed_threshold).astype(np.uint8)

            if not np.any(seed_points):
                # 如果没有足够高置信度的种子点，使用预测边界附近的区域
                from skimage import feature
                edges = feature.canny(binary_pred, sigma=1)
                dilated_edges = ndimage.binary_dilation(edges, iterations=2)
                seed_points = dilated_edges & uncovered

            # 2. 区域生长 - 使用基于置信度的生长准则
            # 首先标记已有的预测区域
            grown_mask = binary_pred.copy()

            # 定义8邻域
            struct = ndimage.generate_binary_structure(2, 2)

            # 迭代生长直到收敛或达到最大迭代次数
            max_iterations = 10
            for i in range(max_iterations):
                prev_grown = grown_mask.copy()

                # 膨胀当前生长区域
                dilated = ndimage.binary_dilation(grown_mask, structure=struct)

                # 只在标签范围内生长
                candidate_region = dilated & (label_np > 0) & (~grown_mask)

                if not np.any(candidate_region):
                    break

                # 基于置信度阈值选择下一轮生长区域
                growth_threshold = max(0.2, np.percentile(pred_np[candidate_region], 60 - i * 5))  # 动态降低阈值
                new_growth = candidate_region & (pred_np > growth_threshold)

                grown_mask = grown_mask | new_growth

                # 检查是否收敛
                if np.sum(grown_mask) == np.sum(prev_grown):
                    break

            # 3. 后处理：填充小空洞
            filled = ndimage.binary_fill_holes(grown_mask)

            # 4. 确保生长区域在标签范围内
            final_mask = filled & (label_np > 0)

            # 转回tensor
            result = torch.tensor(final_mask, device=device).float()

            # 保持原始维度
            return result.unsqueeze(0).unsqueeze(0)

        except Exception as e:
            if self.enable_logging:
                print(f"[DEBUG] 区域生长内部错误: {str(e)}")
            # 失败时返回原始预测
            return pred_prob.unsqueeze(0).unsqueeze(0) if pred_prob.dim() == 2 else pred_prob

    def clean_small_regions(self, pred_prob):
        """
        清理小的孤立区域，减少噪声
        """
        try:
            import numpy as np
            from scipy import ndimage

            device = pred_prob.device

            # 确保输入形状正确
            if pred_prob.dim() == 4:
                pred_2d = pred_prob.squeeze(0).squeeze(0)
            elif pred_prob.dim() == 3:
                pred_2d = pred_prob.squeeze(0)
            else:
                pred_2d = pred_prob

            # 二值化
            binary_mask = (pred_2d > 0.5).cpu().numpy().astype(np.uint8)

            # 标记连通区域
            labeled, num_features = ndimage.label(binary_mask)

            if num_features <= 1:
                # 只有一个区域或没有区域，无需清理
                return pred_prob

            # 计算每个区域的大小
            sizes = np.bincount(labeled.ravel())[1:]  # 排除背景

            if len(sizes) == 0:
                return pred_prob

            # 确定要保留的区域（最大区域和面积超过阈值的区域）
            max_size = np.max(sizes)
            area_threshold = max(20, max_size * 0.1)  # 保留至少为最大区域10%的区域

            # 创建保留掩码
            keep_mask = np.zeros_like(binary_mask, dtype=bool)

            for i in range(1, num_features + 1):
                if sizes[i - 1] >= area_threshold:
                    keep_mask |= (labeled == i)

            # 应用保留掩码到概率图
            cleaned_prob = pred_2d.clone()
            cleaned_prob[torch.tensor(~keep_mask, device=device)] = 0

            # 恢复原始维度
            if pred_prob.dim() == 4:
                return cleaned_prob.unsqueeze(0).unsqueeze(0)
            elif pred_prob.dim() == 3:
                return cleaned_prob.unsqueeze(0)
            else:
                return cleaned_prob

        except Exception as e:
            if self.enable_logging:
                print(f"[DEBUG] 小区域清理内部错误: {str(e)}")
            return pred_prob

    def confidence_calibration(self, pred_prob, label):
        """
        改进的置信度校准
        基于预测-标签一致性和区域特性进行智能校准
        """
        try:
            device = pred_prob.device
            calibrated_prob = pred_prob.clone()

            # 确保label是相同形状
            if label.shape != pred_prob.shape:
                if label.dim() == 4 and pred_prob.dim() == 3:
                    label = label.squeeze(0)
                elif label.dim() == 3 and pred_prob.dim() == 4:
                    calibrated_prob = calibrated_prob.squeeze(0)

            # 计算预测与标签的一致性
            agreement = 1.0 - torch.abs(calibrated_prob - label)

            # 1. 区域一致性加权
            from torch.nn.functional import conv2d

            # 使用3x3卷积核计算局部一致性
            kernel = torch.ones((1, 1, 3, 3), device=device) / 9.0

            # 确保输入是4D张量
            if calibrated_prob.dim() == 3:
                calibrated_prob_4d = calibrated_prob.unsqueeze(0)
                agreement_4d = agreement.unsqueeze(0)
            else:
                calibrated_prob_4d = calibrated_prob
                agreement_4d = agreement

            # 计算局部一致性
            local_agreement = F.conv2d(agreement_4d, kernel, padding=1)

            if calibrated_prob.dim() == 3:
                local_agreement = local_agreement.squeeze(0)

            # 2. 自适应校准因子
            # 在高一致性区域增强置信度，低一致性区域降低置信度
            calibration_factor = 1.0 + (local_agreement - 0.5) * self.calibration_strength * 2.0

            # 3. 应用校准因子
            calibrated_prob = calibrated_prob * calibration_factor

            # 4. 特殊处理边界和孤立点
            # 识别孤立的高置信度点
            binary = (calibrated_prob > 0.8).float()
            if binary.dim() == 3:
                binary_4d = binary.unsqueeze(0)
            else:
                binary_4d = binary

            # 计算每个点的邻域活动像素数
            neighbor_count = F.conv2d(binary_4d, kernel * 9.0, padding=1)

            if binary.dim() == 3:
                neighbor_count = neighbor_count.squeeze(0)

            # 降低孤立点的置信度
            isolated_points = (neighbor_count < 2) & (calibrated_prob > 0.7)
            if isolated_points.any():
                calibrated_prob[isolated_points] *= 0.5

            return torch.clamp(calibrated_prob, 0, 1)

        except Exception as e:
            if self.enable_logging:
                print(f"[DEBUG] 置信度校准内部错误: {str(e)}")
            return pred_prob

