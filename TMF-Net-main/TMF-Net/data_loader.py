import math
import os
import numpy as np
import re
import torch
from monai import data, transforms
import itertools
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, Sampler, DataLoader  # 新增：导入DataLoader
import os
import ast
from scipy import sparse
import random
from scipy.ndimage import binary_opening, binary_closing
from scipy.ndimage import label as label_structure
from scipy.ndimage import sum as sum_structure
import json
import torch.distributed as dist
from PIL import Image
from dataloaders.data_utils import (
    Resize,
    PermuteTransform,
    LongestSidePadding,
    Normalization,
    get_points_from_mask,
    get_bboxes_from_mask
)
import cv2


class PatientGroupSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.patient_groups = self._group_by_patient()
        # 为每个病人组初始化一个索引指针
        self.group_pointers = {i: 0 for i in range(len(self.patient_groups))}
        
    def _group_by_patient(self):
        patient_to_indices = {}
        # 直接访问dataset的datalist属性，获取原始数据
        for idx, item_dict in enumerate(self.dataset.datalist):
            # 从image路径中提取病人ID
            # 路径格式: image/ABD_001_74.png 或 image/patient001_frame01_1.png
            image_path = item_dict['image']
            filename = image_path.split('/')[-1]
            # 尝试匹配 ABD_xxx 或 amos_xxx 格式 (BTCV/AMOS)
            match = re.match(r'(ABD_\d+|amos_\d+)', filename)
            if match:
                patient_id = match.group(1)
            else:
                # 默认格式: patient001_frame01_1.png
                patient_id = filename.split('_')[0]
            if patient_id not in patient_to_indices:
                patient_to_indices[patient_id] = []
            patient_to_indices[patient_id].append(idx)
        return list(patient_to_indices.values())
    
    def __iter__(self):
        # 重置指针，确保每个epoch都从新开始
        self.group_pointers = {i: 0 for i in range(len(self.patient_groups))}

        # 打乱病人组的顺序
        group_indices = list(range(len(self.patient_groups)))
        random.shuffle(group_indices)

        # 确保每个batch中的元素来自不同的病人
        while True:
            # 修复点1：增加空值校验，避免索引越界
            if not group_indices:
                break
            # 选择batch_size个不同的病人组（最多取剩余数量）
            selected_groups = group_indices[:self.batch_size]
            if len(selected_groups) < self.batch_size:
                break

            batch = []
            valid = True
            selected_pointers = {}  # 临时存储本次选择的指针

            # 为每个选中的病人组取一个切片
            for group_idx in selected_groups:
                # 修复点2：校验group_idx是否在有效范围内
                if group_idx >= len(self.patient_groups):
                    valid = False
                    break
                group = self.patient_groups[group_idx]
                pointer = self.group_pointers[group_idx]

                # 检查是否还有切片可以取
                if pointer < len(group):
                    batch.append(group[pointer])
                    # 临时存储指针更新
                    selected_pointers[group_idx] = pointer + 1
                else:
                    # 这个病人组已经取完了，标记为无效
                    valid = False
                    break

            # 只有在valid的情况下才更新指针
            if valid and len(batch) == self.batch_size:
                for group_idx, new_pointer in selected_pointers.items():
                    self.group_pointers[group_idx] = new_pointer
                yield batch

            # 移除已经取完的病人组
            group_indices = [g for g in group_indices if self.group_pointers[g] < len(self.patient_groups[g])]

            if len(group_indices) < self.batch_size:
                break
    
    def __len__(self):
        # 计算可以生成的batch数量
        # 每个batch从不同病人组中各取一个切片，直到至少有一个病人组的切片被取完
        if not self.patient_groups:
            return 0
        min_group_size = min(len(g) for g in self.patient_groups)
        return min_group_size


class UniversalDataset(Dataset):
    def __init__(self, args, datalist, classes_list, transform):
        self.args = args
        self.data_dir = args.data_dir
        # 修复点3：datalist非空校验
        if not datalist:
            raise ValueError("datalist cannot be empty!")
        self.datalist = datalist
        self.test_mode = args.test_mode
        # 修复点4：classes_list空值兜底
        self.classes_list = classes_list.copy() if classes_list else ['LV', 'RV', 'Myo']
        if 'background' in self.classes_list:
            self.classes_list.remove('background')
        self.target_list = self.classes_list
        self.image_size = args.image_size
        self.mask_num = args.mask_num
        self.transform = transform
        # 重试次数限制，避免无限递归
        self.max_retry = 5

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        retry_count = 0
        while retry_count < self.max_retry:
            try:
                item_dict = self.datalist[idx]
                image_path = os.path.join(self.data_dir, item_dict['image'])
                label_path = os.path.join(self.data_dir, item_dict['label'])

                # --- 1. CT 图像读取与窗口化 ---
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if image is None:
                    # 如果读取失败，尝试使用PIL
                    image = np.array(Image.open(image_path.encode('utf-8').decode('utf-8')))

                # BTCV CT 窗口化：腹部窗口 Level=50, Width=450（-175 到 275）
                # 这是标准的腹部 CT 窗宽窗位，能最好地显示腹部器官
                if hasattr(self.args, 'task_name') and self.args.task_name == 'BTCV':
                    lower_bound = -175  # 标准腹部窗口下限
                    upper_bound = 275   # 标准腹部窗口上限
                    image = np.clip(image, lower_bound, upper_bound)
                    # 避免除零错误
                    if upper_bound > lower_bound:
                        image = ((image - lower_bound) / (upper_bound - lower_bound) * 255.0).astype(np.uint8)
                    else:
                        image = np.zeros_like(image, dtype=np.uint8)
                # [新增] AMOS2022_MR 自适应截断 (MR 图像没有绝对的 HU 值)
                elif hasattr(self.args, 'task_name') and self.args.task_name == 'AMOS2022_MR':
                    # 使用 1% - 99% 的百分位数进行自适应截断，去除极端异常噪点，拉伸对比度
                    lower_bound = np.percentile(image, 1)
                    upper_bound = np.percentile(image, 99)
                    image = np.clip(image, lower_bound, upper_bound)
                    if upper_bound > lower_bound:
                        image = ((image - lower_bound) / (upper_bound - lower_bound) * 255.0).astype(np.uint8)
                    else:
                        image = np.zeros_like(image, dtype=np.uint8)

                # 转换为 RGB 以适应 SAM
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 1:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                # --- 2. 标签读取 ---
                if label_path.endswith('.npz'):
                    gt_shape = ast.literal_eval(label_path.split('.')[-2])
                    allmatrix_sp = sparse.load_npz(label_path)
                    label_array = allmatrix_sp.toarray().reshape(gt_shape)
                else:
                    label_array = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                    label_array = np.squeeze(label_array).astype(np.uint8)

                # --- 3. 提取 Patient ID ---
                filename = os.path.basename(image_path)
                match = re.match(r'(ABD_\d+|amos_\d+)', filename)
                patient_id = match.group(1) if match else filename.split('_')[0]

                if self.test_mode:

                    item_ori = {'image': image, 'label': label_array}
                    item = self.transform(item_ori)
                    _, H, W = item['image'].shape

                    point_coords, point_labels, bboxes = [], [], []

                    label_ids = torch.sum(item['label'], dim=(1, 2))
                    label_ids = torch.nonzero(label_ids != 0, as_tuple=True)[0].tolist()

                    if len(label_ids) == 0:
                        retry_count += 1
                        idx = np.random.randint(self.__len__())
                        continue

                    nonzero_labels = torch.zeros(len(label_ids), 1, H, W)
                    nonzero_category = []
                    nonzero_ori_labels = []
                    for idx_label, region_id in enumerate(label_ids):
                        nonzero_labels[idx_label][0] = item['label'][region_id]
                        nonzero_ori_labels.append(torch.tensor(np.moveaxis(label_array[region_id], -1, 0)))

                        point_and_labels = get_points_from_mask(nonzero_labels[idx_label], top_num=0.5)
                        point_coords.append(torch.as_tensor(point_and_labels[0]))
                        point_labels.append(torch.as_tensor(point_and_labels[1]))

                        bboxes.append(torch.as_tensor(get_bboxes_from_mask(nonzero_labels[idx_label], offset=0)))

                        # 使用更可靠的类别提取逻辑：优先使用配置文件中的target_list
                        # 避免依赖脆弱的文件名正则解析
                        category = self.target_list[region_id] if region_id < len(self.target_list) else 'LV'
                        nonzero_category.append(category)

                    item['gt'] = nonzero_labels
                    item['ori_gt'] = torch.stack(nonzero_ori_labels, dim=0)

                    item['gt_target'] = nonzero_category
                    item['gt_point_coords'] = torch.stack(point_coords)
                    item['gt_point_labels'] = torch.stack(point_labels)
                    item['gt_bboxes'] = torch.stack(bboxes)

                    item['image_root'] = [image_path]
                    item['patient_id'] = patient_id

                else:
                    pseudo_path = os.path.join(self.data_dir, item_dict['imask'])

                    try:
                        pseudo_array = np.load(pseudo_path).astype(np.float32)
                    except Exception as e:
                        print(f'{pseudo_path} not load, error: {e}')
                        retry_count += 1
                        idx = np.random.randint(self.__len__())
                        continue

                    item_ori = {'image': image, 'label': label_array, 'pseudo': pseudo_array}
                    item = self.transform(item_ori)
                    item['pseudo'] = self.cleanse_pseudo_label(item['pseudo'])

                    pseudo_ids = torch.unique(item['pseudo'])
                    pseudo_ids = pseudo_ids[pseudo_ids != -1]

                    if len(pseudo_ids) == 0:
                        retry_count += 1
                        idx = np.random.randint(self.__len__())
                        continue

                    _, H, W = item['image'].shape
                    select_pseudo = torch.zeros(self.mask_num, 1, H, W)

                    (
                        select_pseudo,
                        point_coords_pseudo,
                        point_labels_pseudo,
                        bboxes_pseudo
                    ) = self.preprocess_pseudo(item['pseudo'], pseudo_ids, select_pseudo)

                    label_ids = torch.sum(item['label'], dim=(1, 2))
                    label_ids = torch.nonzero(label_ids != 0, as_tuple=True)[0].tolist()

                    if len(label_ids) == 0:
                        retry_count += 1
                        idx = np.random.randint(self.__len__())
                        continue

                    select_labels = torch.zeros(self.mask_num, 1, H, W)
                    # 提取图像文件名
                    image_filename = os.path.basename(image_path)
                    # 【关键修复】从文件名提取 patient_id
                    # BTCV格式: ABD_001_74.png -> ABD_001, AMOS格式: amos_0001_15.png -> amos_0001
                    match = re.match(r'(ABD_\d+|amos_\d+)', image_filename)
                    if match:
                        patient_id = match.group(1)
                    else:
                        patient_id = image_filename.split('_')[0]
                    (
                        select_labels,
                        point_coords,
                        point_labels,
                        bboxes,
                        nonzero_category
                    ) = self.preprocess_label(item['label'], label_ids, select_labels, image_filename=image_filename)

                    item['gt'] = select_labels
                    item['pseudo'] = select_pseudo

                    item['gt_point_coords'] = point_coords
                    item['gt_point_labels'] = point_labels
                    item['gt_bboxes'] = bboxes
                    item['gt_target'] = nonzero_category
                    item['pseudo_point_coords'] = point_coords_pseudo
                    item['pseudo_point_labels'] = point_labels_pseudo
                    item['pseudo_bboxes'] = bboxes_pseudo

                if type(item) == list:
                    assert len(item) == 1
                    item = item[0]

                assert type(item) != list

                # 在返回 post_item 之前添加
                if not self.test_mode:
                    post_item = self.std_keys(item)
                    post_item['patient_id'] = patient_id  # 添加病人ID
                    post_item['image_filename'] = image_filename  # 可选：添加完整文件名用于调试
                else:
                    post_item = self.std_keys(item)
                return post_item
            except Exception as e:
                print(f"Error loading sample {idx}, retry {retry_count+1}/{self.max_retry}, error: {e}")
                retry_count += 1
                idx = np.random.randint(self.__len__())
        # 超过最大重试次数，抛出异常
        raise RuntimeError(f"Failed to load valid sample after {self.max_retry} retries!")

    def get_preprocess_shape(self, oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def preprocess_pseudo(self, pseudo_label, pseudo_ids, select_pseudo):
        point_coords, point_labels, bboxes = [], [], []

        pseudo_region_ids = random.sample(list(pseudo_ids), k=self.mask_num) if len(
            pseudo_ids) >= self.mask_num else random.choices(list(pseudo_ids), k=self.mask_num)
        for idx, region_id in enumerate(pseudo_region_ids):
            select_pseudo[idx][pseudo_label == region_id.item()] = 1

            point_and_labels = get_points_from_mask(select_pseudo[idx], top_num=0.5)
            point_coords.append(torch.as_tensor(point_and_labels[0]))
            point_labels.append(torch.as_tensor(point_and_labels[1]))

            bboxes.append(torch.as_tensor(get_bboxes_from_mask(select_pseudo[idx], offset=5)))

        point_coords = torch.stack(point_coords)
        point_labels = torch.stack(point_labels)
        bboxes = torch.stack(bboxes)

        return select_pseudo, point_coords, point_labels, bboxes

    def preprocess_label(self, gt_label, label_ids, select_labels, image_filename=None):
        point_coords, point_labels, bboxes, categories = [], [], [], []
        label_region_ids = random.sample(list(label_ids), k=self.mask_num) if len(
            label_ids) >= self.mask_num else random.choices(list(label_ids), k=self.mask_num)

        for idx, region_id in enumerate(label_region_ids):
            select_labels[idx][0] = gt_label[region_id]

            point_and_labels = get_points_from_mask(select_labels[idx], top_num=0.5)
            point_coords.append(torch.as_tensor(point_and_labels[0]))
            point_labels.append(torch.as_tensor(point_and_labels[1]))

            bboxes.append(torch.as_tensor(get_bboxes_from_mask(select_labels[idx], offset=5)))

            # 使用更可靠的类别提取逻辑：优先使用配置文件中的target_list
            # 避免依赖脆弱的文件名正则解析
            category = self.target_list[region_id] if region_id < len(self.target_list) else 'LV'
            categories.append(category)

        point_coords = torch.stack(point_coords)
        point_labels = torch.stack(point_labels)
        bboxes = torch.stack(bboxes)

        return select_labels, point_coords, point_labels, bboxes, categories

    def std_keys(self, post_item):
        keys_to_remain = ['image', 'gt', 'ori_gt', 'image_root',
                        'gt_point_coords', 'gt_point_labels', 'gt_bboxes', 'gt_target',
                        'pseudo', 'pseudo_point_coords', 'pseudo_point_labels', 'pseudo_bboxes',
                        'patient_id', 'image_filename']  # 添加 patient_id 和 image_filename
        keys_to_remove = post_item.keys() - keys_to_remain
        for key in keys_to_remove:
            del post_item[key]
        return post_item

    def _get_sample_category(self, sample_filename, region_id=None):
        """修复的类别提取逻辑（适配ACDC数据集）
        ACDC文件名格式：patientXX_frameXX_1.png（1=LV，2=RV，3=Myo）
        """
        # 从文件名提取类别（ACDC数据集）
        match = re.search(r'_([0-9]+)\.png$', sample_filename)
        if match:
            category_id = int(match.group(1))
            category_map = {
                1: 'LV',  # 左心室
                2: 'RV',  # 右心室
                3: 'Myo'  # 心肌
            }
            return category_map.get(category_id, 'LV')  # 避免Unknown类别

        # 后备方案：处理其他数据集或格式不正确的文件名
        if region_id is not None:
            category_map = {
                1: 'LV',  # LV
                2: 'Myo',  # Myo
                3: 'RV'  # RV
            }
            return category_map.get(region_id, 'LV')

        # 最终后备
        return 'LV'

    def cleanse_pseudo_label(self, pseudo_seg):
        # 修复点6：统一张量/数组类型，处理设备一致性
        # 先将张量移到CPU，转为numpy数组处理
        device = pseudo_seg.device
        pseudo_seg_np = pseudo_seg.cpu().numpy() if pseudo_seg.is_cuda else pseudo_seg.numpy()
        
        total_voxels = pseudo_seg_np.size
        threshold = total_voxels * 0.0005
        unique_values = np.unique(pseudo_seg_np)

        for value in unique_values:
            voxel_count = (pseudo_seg_np == value).sum()
            if voxel_count < threshold:
                pseudo_seg_np[pseudo_seg_np == value] = -1

        for label in np.unique(pseudo_seg_np):
            if label == -1:
                continue

            binary_mask = pseudo_seg_np == label
            open = binary_opening(binary_mask.squeeze())
            close = binary_closing(open)
            processed_mask = close

            labeled_mask, num_labels = label_structure(processed_mask)
            label_sizes = sum_structure(processed_mask, labeled_mask, range(num_labels + 1))
            small_labels = np.where(label_sizes < threshold)[0]
            for label_del in small_labels:
                processed_mask[labeled_mask == label_del] = False

            pseudo_seg_np[binary_mask] = -1
            pseudo_seg_np[processed_mask.reshape(pseudo_seg_np.shape)] = label

        # 转回torch张量，并放回原设备
        pseudo_seg = torch.tensor(pseudo_seg_np, device=device, dtype=pseudo_seg.dtype)
        return pseudo_seg


def test_collate_fn(batch):
    assert len(batch) == 1, 'Please set batch size to 1 when testing mode'
    gt_prompt = {'point_coords': [], 'point_labels': [], 'bboxes': []}
    gt_prompt['point_coords'] = batch[0]['gt_point_coords']
    gt_prompt['point_labels'] = batch[0]['gt_point_labels']
    gt_prompt['bboxes'] = batch[0]['gt_bboxes']
    image_root = batch[0]['image_root']
    target_list = batch[0]['gt_target']
    patient_id = batch[0]['patient_id']
    num_masks = len(batch[0]['gt_target'])
    patient_ids = [patient_id] * num_masks  # 每个样本对应一个ID
    return {
        'image': batch[0]['image'].unsqueeze(0),  # 添加 batch 维度
        'label': batch[0]['gt'],
        'ori_label': batch[0]['ori_gt'],
        'gt_prompt': gt_prompt,
        'target_list': target_list,
        'image_root': image_root,
        'patient_ids': patient_ids,
    }


def train_collate_fn(batch):
    images, labels, pseudos, target_list, patient_ids = [], [], [], [], []
    gt_prompt = {'point_coords': [], 'point_labels': [], 'bboxes': []}
    pseudo_prompt = {'point_coords': [], 'point_labels': [], 'bboxes': []}

    for sample in batch:
        # 每个病人有mask_num个样本，每个样本都要加相同的patient_id
        num_masks = len(sample['gt_target'])  # 就是args.mask_num
        patient_id = sample['patient_id']
        
        # 【关键修复】每个样本对应一个patient_id，重复mask_num次
        patient_ids.extend([patient_id] * num_masks)
        
        # 原有逻辑不变
        images.append(sample['image'])
        labels.append(sample['gt'])
        gt_prompt['point_coords'].append(sample['gt_point_coords'])
        gt_prompt['point_labels'].append(sample['gt_point_labels'])
        gt_prompt['bboxes'].append(sample['gt_bboxes'])
        target_list.extend(sample['gt_target'])
        
        pseudos.append(sample['pseudo'])
        pseudo_prompt['point_coords'].append(sample['pseudo_point_coords'])
        pseudo_prompt['point_labels'].append(sample['pseudo_point_labels'])
        pseudo_prompt['bboxes'].append(sample['pseudo_bboxes'])

    images = torch.stack(images, dim=0)
    labels = torch.cat(labels, dim=0) if labels else torch.tensor([])  # 修复点7：空列表兜底
    pseudos = torch.cat(pseudos, dim=0) if pseudos else torch.tensor([])  # 修复点7：空列表兜底

    # 修复点8：优化prompt空值处理，返回空张量而非None
    gt_prompt = {
        key: torch.cat(value, dim=0) if len(value) != 0 else torch.tensor([]) 
        for key, value in gt_prompt.items()
    }
    pseudo_prompt = {
        key: torch.cat(value, dim=0) if len(value) != 0 else torch.tensor([]) 
        for key, value in pseudo_prompt.items()
    }

    return {
        'image': images,  # 已经是tensor
        'label': labels,
        'pseudo': pseudos,
        'target_list': target_list,
        'gt_prompt': gt_prompt,
        'pseudo_prompt': pseudo_prompt,
        'patient_ids': patient_ids,  # 现在长度=batch_size=16
    }


def get_loader(args):
    print("[DEBUG] Starting to initialize data loader")

    try:
        dataset_json = os.path.join(args.data_dir, 'dataset.json')
        print(f"[DEBUG] Trying to load dataset configuration file: {dataset_json}")

        if not os.path.exists(dataset_json):
            print(f"[ERROR] Dataset JSON file not found: {dataset_json}")
            raise FileNotFoundError(f"Dataset JSON file not found: {dataset_json}")

        dataset_dict = json.load(open(dataset_json, 'r'))
        print(f"[DEBUG] Successfully loaded dataset configuration file, containing keys: {list(dataset_dict.keys())}")

        target_size = (args.image_size, args.image_size)
        print(f"[DEBUG] Setting image target size: {target_size}")

        if args.test_mode:
            print("[DEBUG] Test mode: Loading test dataset")
            datalist = dataset_dict.get('test', [])
            print(f"[DEBUG] Test dataset length: {len(datalist)} samples")
            collate_fn = test_collate_fn
            transform = transforms.Compose(
                [
                    Resize(keys=["image", "label"], target_size=target_size),
                    PermuteTransform(keys=["image"], dims=(2, 0, 1)),
                    transforms.ToTensord(keys=["image", "label"]),
                    Normalization(keys=["image"]),
                    transforms.RandScaleIntensityd(keys="image", factors=0.2, prob=0.2),
                    transforms.RandShiftIntensityd(keys="image", offsets=0.2, prob=0.2),
                ]
            )
        else:
            print("[DEBUG] Training mode: Loading training dataset")
            datalist = dataset_dict.get('training', [])
            print(f"[DEBUG] Training dataset length: {len(datalist)} samples")
            collate_fn = train_collate_fn
            # 修复点9：补全训练模式的几何增强算子
            transform = transforms.Compose(
                [
                    Resize(keys=["image", "label", "pseudo"], target_size=target_size),
                    PermuteTransform(keys=["image"], dims=(2, 0, 1)),
                    transforms.ToTensord(keys=["image", "label", "pseudo"]),
                    Normalization(keys=["image"]),
                    # 几何增强（补全）
                    transforms.RandFlipd(keys=["image", "label", "pseudo"], prob=0.5, spatial_axis=0),
                    transforms.RandFlipd(keys=["image", "label", "pseudo"], prob=0.5, spatial_axis=1),
                    transforms.RandRotated(
                        keys=["image", "label", "pseudo"], 
                        range_x=(-15, 15), 
                        range_y=(-15, 15), 
                        range_z=(-15, 15),
                        prob=0.3
                    ),
                    transforms.RandZoomd(
                        keys=["image", "label", "pseudo"],
                        min_zoom=0.8,
                        max_zoom=1.2,
                        prob=0.3
                    ),
                    # 添加弹性变形
                    transforms.RandAffined(
                        keys=["image", "label", "pseudo"],
                        rotate_range=0.0,
                        shear_range=0.1,
                        translate_range=0.1,
                        scale_range=(0.9, 1.1),
                        prob=0.3
                    ),
                    # 强度增强
                    transforms.RandScaleIntensityd(keys="image", factors=0.2, prob=0.2),
                    transforms.RandShiftIntensityd(keys="image", offsets=0.2, prob=0.2),
                    # 增加强度增强的概率
                    transforms.RandScaleIntensityd(keys="image", factors=0.15, prob=0.3),
                    transforms.RandShiftIntensityd(keys="image", offsets=0.15, prob=0.3),
                ]
            )
        
        # 修复点10：补全数据集初始化逻辑
        # 优先使用 args.classes（从任务路由表设置），否则从 dataset.json 读取
        if hasattr(args, 'classes') and args.classes:
            classes_list = args.classes
        else:
            # 优先从 dataset.json 的 classes 字段读取，如果没有则从 labels 字段构建
            classes_list = dataset_dict.get('classes', None)
            if classes_list is None:
                # 从 labels 字段构建类别列表（按序号排序）
                labels_dict = dataset_dict.get('labels', {})
                if labels_dict:
                    # labels格式: {"0": "background", "1": "spleen", ...}
                    # 提取所有类别（不包括background）
                    classes_list = [labels_dict[str(i)] for i in range(len(labels_dict)) if labels_dict[str(i)] != 'background']
                else:
                    classes_list = ['LV', 'RV', 'Myo']
        # 初始化数据集
        dataset = UniversalDataset(
            args=args,
            datalist=datalist,
            classes_list=classes_list,
            transform=transform
        )
        print(f"[DEBUG] Successfully initialized dataset, total samples: {len(dataset)}")
        
        # 修复点11：补全DataLoader创建逻辑
        if args.test_mode:
            # 测试模式：batch_size=1，不使用sampler
            data_loader = DataLoader(
                dataset=dataset,
                batch_size=1,  # 测试模式强制batch_size=1
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=False
            )
        else:
            # 训练模式：使用PatientGroupSampler保证batch内样本来自不同病人
            sampler = PatientGroupSampler(dataset, batch_size=args.batch_size)
            data_loader = DataLoader(
                dataset=dataset,
                batch_sampler=sampler,  # 使用自定义sampler，不再指定batch_size
                num_workers=args.num_workers,
                collate_fn=collate_fn,
                pin_memory=True
            )
        
        print(f"[DEBUG] Successfully created DataLoader, total batches per epoch: {len(data_loader)}")
        return data_loader

    except Exception as e:
        print(f"[ERROR] Failed to initialize data loader: {e}")
        raise  # 重新抛出异常，便于上层捕获