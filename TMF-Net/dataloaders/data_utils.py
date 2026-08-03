import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from monai import data, transforms
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
from skimage.measure import label, regionprops

class Resize(transforms.Transform):
    def __init__(self, keys, target_size):
        self.keys = keys
        self.target_size = target_size

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if len(d[key].shape) == 4:
                label = d[key]
                resized_labels = np.zeros((label.shape[0], self.target_size[0], self.target_size[1]))
                for i in range(label.shape[0]):
                    pil_label = to_pil_image(label[i])
                    resized_label = resize(pil_label, self.target_size, interpolation=InterpolationMode.NEAREST)
                    resized_labels[i] = np.array(resized_label)
                d[key] = resized_labels
            else:
                image = to_pil_image(d[key])
                d[key] = resize(image, self.target_size, interpolation=InterpolationMode.NEAREST)
                d[key] = np.array(d[key])
            
            if len(d[key].shape) == 2:
                d[key] = d[key][np.newaxis, ...]
        return d

class PermuteTransform(transforms.Transform):
    def __init__(self, keys, dims):
        self.dims = dims
        self.keys = keys
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = np.transpose(d[key], self.dims)
        return d

class LongestSidePadding(transforms.Transform):
    def __init__(self, keys, input_size):
        self.keys = keys
        self.input_size = input_size
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            h, w = d[key].shape[-2:]
            padh = self.input_size - h
            padw = self.input_size - w
            d[key] = F.pad(d[key], (0, padw, 0, padh))
        return d

class Normalization(transforms.Transform):
    def __init__(self, keys):
        self.keys = keys
        pixel_mean = (123.675, 116.28, 103.53)
        pixel_std = (58.395, 57.12, 57.375)
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            # 检查数据有效性
            if isinstance(d[key], np.ndarray):
                if np.isnan(d[key]).any() or np.isinf(d[key]).any():
                    raise ValueError(f"数据 {key} 包含 NaN 或 Inf 值！")
                if np.min(d[key]) < -1e6 or np.max(d[key]) > 1e6:
                    raise ValueError(f"数据 {key} 包含异常值，范围过大！")
            elif isinstance(d[key], torch.Tensor):
                if torch.isnan(d[key]).any() or torch.isinf(d[key]).any():
                    raise ValueError(f"数据 {key} 包含 NaN 或 Inf 值！")
                if torch.min(d[key]) < -1e6 or torch.max(d[key]) > 1e6:
                    raise ValueError(f"数据 {key} 包含异常值，范围过大！")
            
            d[key] = (d[key] - self.pixel_mean) / self.pixel_std
        return d
    

# 边界点检测函数
def get_points_from_mask(mask, get_point=3, top_num=0.7, add_boundary=True):

    if len(mask.shape) > 2:
        mask = mask.squeeze()

    if isinstance(mask, torch.Tensor):
        # 安全地将GPU张量转换为numpy数组
        mask = mask.detach().cpu().numpy()

    fg_coords = np.argwhere(mask == 1)[:,::-1]
    bg_coords = np.argwhere(mask == 0)[:,::-1]

    # 如果没有前景像素，返回背景点
    if len(fg_coords) == 0:
        # 确保返回固定数量的点
        coords = []
        labels = []
        for i in range(get_point):
            if i < len(bg_coords):
                coord = bg_coords[np.random.randint(len(bg_coords))]
                coords.append(coord)
                labels.append(0)
            else:
                # 如果背景点不够，重复使用最后一个点
                if coords:
                    coords.append(coords[-1])
                    labels.append(0)
                else:
                    # 如果连背景点都没有，返回原点
                    coords.append(np.array([0, 0]))
                    labels.append(0)
        coords = torch.as_tensor(coords, dtype=torch.float)
        labels = torch.as_tensor(labels, dtype=torch.int)
        return coords, labels
    
    selected_coords = []
    selected_labels = []
    
    # 1. 添加中心点（始终添加）
    centroid = np.mean(fg_coords, axis=0)
    selected_coords.append(centroid)
    selected_labels.append(1)
    
    # 2. 添加距离质心较近的点
    distances = np.sqrt(np.sum((fg_coords - centroid)**2, axis=1))
    sorted_indices = np.argsort(distances)
    num_points = len(fg_coords)
    top_k = max(1, int(num_points * top_num))
    top_indices = sorted_indices[:top_k]
    
    # 如果有足够的点，从中随机选择一些，但避免与中心点重复
    if len(top_indices) > 1 and get_point > 1:
        # 排除已经选择的中心点
        center_idx = np.argmin(distances)
        top_indices = [idx for idx in top_indices if idx != center_idx]
        
        # 从剩余的近点中随机选择
        num_near_points = min(get_point - 1, len(top_indices))
        if num_near_points > 0:
            near_indices = np.random.choice(top_indices, size=num_near_points, replace=False)
            selected_coords.extend(fg_coords[near_indices])
            selected_labels.extend([1] * num_near_points)
    
    # 3. 如果启用边界点添加
    if add_boundary and len(selected_coords) < get_point:
        # 使用形态学操作找到边界
        from scipy.ndimage import binary_erosion
        mask_binary = mask.astype(np.bool_)
        eroded = binary_erosion(mask_binary)
        boundary = mask_binary ^ eroded
        boundary_coords = np.argwhere(boundary == 1)[:,::-1]
        
        if len(boundary_coords) > 0:
            # 从边界点中随机选择
            num_boundary_points = min(get_point - len(selected_coords), len(boundary_coords))
            if num_boundary_points > 0:
                boundary_indices = np.random.choice(len(boundary_coords), size=num_boundary_points, replace=False)
                selected_coords.extend(boundary_coords[boundary_indices])
                selected_labels.extend([1] * num_boundary_points)
    
    # 4. 如果点数仍不足，从整个前景区域随机选择
    if len(selected_coords) < get_point:
        # 排除已经选择的点
        all_indices = np.arange(len(fg_coords))
        selected_indices = []
        for coord in selected_coords:
            dist_to_all = np.sqrt(np.sum((fg_coords - coord)**2, axis=1))
            selected_idx = np.argmin(dist_to_all)
            selected_indices.append(selected_idx)
        
        remaining_indices = [idx for idx in all_indices if idx not in selected_indices]
        
        if len(remaining_indices) > 0:
            num_random_points = min(get_point - len(selected_coords), len(remaining_indices))
            random_indices = np.random.choice(remaining_indices, size=num_random_points, replace=False)
            selected_coords.extend(fg_coords[random_indices])
            selected_labels.extend([1] * num_random_points)
    
    # 5. 如果仍然不足，添加足够的背景点以达到指定数量
    while len(selected_coords) < get_point:
        if len(bg_coords) > 0:
            # 从背景点中随机选择
            bg_idx = np.random.randint(len(bg_coords))
            selected_coords.append(bg_coords[bg_idx])
            selected_labels.append(0)
        else:
            # 如果没有背景点，重复使用最后一个选择的点
            if selected_coords:
                selected_coords.append(selected_coords[-1])
                selected_labels.append(selected_labels[-1])
            else:
                # 如果连前景点都没有，返回原点
                selected_coords.append(np.array([0, 0]))
                selected_labels.append(0)
    
    # 转换为张量
    coords = torch.as_tensor(np.array(selected_coords), dtype=torch.float)
    labels = torch.as_tensor(np.array(selected_labels), dtype=torch.int)
    
    return coords, labels


def get_bboxes_from_mask(masks, offset=0):
    if masks.size(1) == 1:
        masks = masks.squeeze(1)
    B, H, W = masks.shape
    bounding_boxes = []
    for i in range(B):
        mask = masks[i]
        y_coords, x_coords = torch.nonzero(mask, as_tuple=True)
        
        if len(y_coords) == 0 or len(x_coords) == 0:
            bounding_boxes.append((0, 0, 0, 0))
        else:
            y0, y1 = y_coords.min().item(), y_coords.max().item()
            x0, x1 = x_coords.min().item(), x_coords.max().item()

            if offset > 0:
                y0 = max(0, y0 + torch.randint(-offset, offset + 1, (1,)).item())
                y1 = min(W - 1, y1 + torch.randint(-offset, offset + 1, (1,)).item())
                x0 = max(0, x0 + torch.randint(-offset, offset + 1, (1,)).item())
                x1 = min(H - 1, x1 + torch.randint(-offset, offset + 1, (1,)).item())

            bounding_boxes.append((x0, y0, x1, y1))

    return torch.tensor(bounding_boxes, dtype=torch.float).unsqueeze(1)


def compute_dice_coefficient(pred, target, smooth=1e-5):
    """计算Dice系数"""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    pred = (pred > 0.5).astype(np.float32)
    target = target.astype(np.float32)
    
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


def compute_iou(pred, target, smooth=1e-5):
    """计算IoU (Intersection over Union)"""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    pred = (pred > 0.5).astype(np.float32)
    target = target.astype(np.float32)
    
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou


def compute_hausdorff_distance(pred, target, percentile=95):
    """计算Hausdorff距离（使用边界点）"""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # 转换为二值图像
    pred = (pred > 0.5).astype(np.bool_)
    target = target.astype(np.bool_)
    
    # 如果任一掩码为空，返回0
    if not np.any(pred) or not np.any(target):
        return 0.0
    
    try:
        from scipy.ndimage import binary_erosion
        from scipy.spatial.distance import directed_hausdorff
        
        # 计算边界
        pred_boundary = pred ^ binary_erosion(pred)
        target_boundary = target ^ binary_erosion(target)
        
        # 获取边界点坐标
        pred_points = np.argwhere(pred_boundary)
        target_points = np.argwhere(target_boundary)
        
        # 如果边界点为空，使用所有点
        if len(pred_points) == 0:
            pred_points = np.argwhere(pred)
        if len(target_points) == 0:
            target_points = np.argwhere(target)
        
        # 计算双向Hausdorff距离
        if len(pred_points) > 0 and len(target_points) > 0:
            d1 = directed_hausdorff(pred_points, target_points)[0]
            d2 = directed_hausdorff(target_points, pred_points)[0]
            hausdorff_dist = max(d1, d2)
        else:
            hausdorff_dist = 0.0
            
        return hausdorff_dist
    except ImportError:
        # 如果scipy不可用，返回0
        return 0.0
    except Exception:
        # 处理其他可能的错误
        return 0.0


if __name__ == '__main__':
    image_path = r'demo_image'
    import cv2
    import matplotlib.pyplot as plt 
    import numpy as np
    masks = cv2.imread(image_path, 0) // 255.
    bboxes = get_bboxes_from_mask(torch.tensor(masks).unsqueeze(0), offset=0)  
    bboxes = bboxes.squeeze(0).numpy()  
    plt.imshow(masks, cmap='gray')  
    for box in bboxes:  
        x0, y0, x1, y1 = box  
        plt.gca().add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, edgecolor='r', facecolor='none'))  
  
    plt.show()