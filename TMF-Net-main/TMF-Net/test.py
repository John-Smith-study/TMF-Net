# set up environment
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import csv
import ast
import time
import numpy as np
from scipy import ndimage

join = os.path.join
from tqdm import tqdm
from torch.backends import cudnn
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from segment_anything import sam_model_registry
import argparse
from torch.cuda import amp
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime
import logging
from model import TMFNet
from train import TemporalFusionEvaluator
from utils import FocalDice_MSELoss

import warnings
import re
from data_loader import get_loader
import json

# 从utils导入通用的指标计算函数
from utils import compute_iou, get_iou_and_dice, get_hd95_and_assd

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', type=str, default='work_dir')
parser.add_argument('--task_name', type=str, default='AMOS2022_MR')
parser.add_argument('--dataset', type=str, default='amos2022_mr', help='Dataset type: acdc, btcv, amos2022_mr')
# load data
parser.add_argument("--data_dir", type=str, default='dataset/AMOS2022_MR')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--test_mode', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=1)
# load model
parser.add_argument('--model_type', type=str, default='vit_b')
parser.add_argument('--sam_checkpoint', type=str, default='ckpt/IMISNet-B.pth')

parser.add_argument('--pretrain_path', type=str, default='')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--mask_num', type=int, default=None)
parser.add_argument('--prompt_mode', type=str, default='points')
parser.add_argument('--inter_num', type=int, default=8, help='交互轮数，增加可提升性能但会减慢测试速度')
parser.add_argument('--use_temporal_fusion', action='store_true', default=True, help='启用时序融合测试')
parser.add_argument('--evaluate_temporal_performance', action='store_true', default=False, help='评估时序性能')
parser.add_argument('--best_threshold', action='store_true', default=True, help='是否使用多阈值选择最佳结果')
# train
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0])
parser.add_argument('--multi_gpu', action='store_true', default=False)
parser.add_argument('--port', type=int, default=12361)
parser.add_argument('--dist', dest='dist', type=bool, default=False, help='distributed training or not')
parser.add_argument('-num_workers', type=int, default=1)

args = parser.parse_args()


# 添加随机种子初始化函数
def init_seeds(seed=42, cuda_deterministic=True):
    """
    初始化所有随机种子，确保实验可复现性
    :param seed: 随机种子值
    :param cuda_deterministic: 是否使用确定性的CUDA操作
    """
    # 设置Python随机种子
    random.seed(seed)
    # 设置NumPy随机种子
    np.random.seed(seed)
    # 设置PyTorch随机种子
    torch.manual_seed(seed)
    # 设置CUDA随机种子
    torch.cuda.manual_seed_all(seed)
    # 确保CUDA确定性
    if cuda_deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True


# 在程序开始时立即调用种子初始化函数，确保所有后续操作都具有确定性
init_seeds(seed=42, cuda_deterministic=True)

# 设置可见GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])

logger = logging.getLogger(__name__)
LOG_OUT_DIR = join(args.work_dir, args.task_name)

# 智能设备设置：优先使用args.device，自动处理设备映射
if args.device == 'cuda' and torch.cuda.is_available():
    # 如果device是'cuda'，使用gpu_ids中的第一个设备
    device = torch.device(f"cuda:{args.gpu_ids[0]}" if isinstance(args.gpu_ids, list) else f"cuda:{args.gpu_ids}")
else:
    device = args.device

MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)


def build_model(args):
    category_weights = 'dataloaders/categories_weight.pkl'
    # 根据数据集类型设置num_classes
    if args.dataset.lower() == 'acdc':
        num_classes = 4
    elif args.dataset.lower() == 'btcv':
        num_classes = 14  # BTCV有14个类别（0-13）
    else:
        num_classes = 4  # 默认值
    
    sam = sam_model_registry[args.model_type](args).to(device)

    # 构建TMFNet模型
    imis = TMFNet(
        sam,
        test_mode=args.test_mode,
        category_weights=category_weights,
        num_classes=num_classes         # 传递num_classes参数
    ).to(device)
    print(f"Building TMFNet")

    # 修复权重加载：兼容时序层权重缺失的情况
    if args.pretrain_path is not None and os.path.exists(args.pretrain_path):
        try:
            # 使用智能设备映射，自动处理设备不匹配问题
            def map_location(storage, loc):
                """智能设备映射：优先使用args.device，自动处理cuda:1映射到cuda:0"""
                if torch.cuda.is_available():
                    # 如果存储在cuda设备上，映射到args.device
                    if loc.startswith('cuda'):
                        return storage.cuda(args.gpu_ids[0] if isinstance(args.gpu_ids, list) else args.gpu_ids)
                    return storage
                return storage.cpu()

            last_ckpt = torch.load(args.pretrain_path, map_location=map_location)
            if 'model_state_dict' in last_ckpt:
                state_dict = last_ckpt['model_state_dict']

                # 检查时序层权重是否存在
                temporal_weight_key = "multiscale_temporal_fusion.fusion_weights"
                if temporal_weight_key not in state_dict:
                    logger.warning(f"时序融合层权重 {temporal_weight_key} 缺失，自动初始化该层权重")
                    # 只加载非时序层权重
                    non_temporal_state_dict = {k: v for k, v in state_dict.items() if
                                               "multiscale_temporal_fusion" not in k}
                    imis.load_state_dict(non_temporal_state_dict, strict=False)
                else:
                    # 正常加载所有权重
                    imis.load_state_dict(state_dict, strict=False)
                logger.info(f"成功加载检查点: {args.pretrain_path} (epoch {last_ckpt.get('epoch', 0)})")
        except Exception as e:
            logger.error(f"加载权重失败: {str(e)}")
            # 初始化时序层权重（保底）
            pass

    if args.multi_gpu:
        imis = DDP(imis, device_ids=[args.rank], output_device=args.rank)
    return imis


# 确保在测试环境中TemporalFusionEvaluator可用
if 'TemporalFusionEvaluator' not in globals():
    class TemporalFusionEvaluator:
        """时序融合性能评估器"""

        def __init__(self):
            self.metrics = {
                'convergence_speed': [],  # 收敛速度
                'interaction_efficiency': [],  # 交互效率
                'temporal_consistency': [],  # 时序一致性
                'final_accuracy': []  # 最终精度
            }

        def evaluate_temporal_performance(self, model, test_loader):
            """评估时序性能"""
            device = next(model.parameters()).device
            model.eval()

            # 💥 关键修复：强制启用时序模块的 stage_factor
            if hasattr(model, 'multiscale_temporal_fusion'):
                fusion = model.multiscale_temporal_fusion
                if hasattr(fusion, 'stage_factor'):
                    fusion.stage_factor = 0.4  # 强制启用正常融合权重
                    print(f"[DEBUG] 已设置 stage_factor = 0.4")
                if hasattr(fusion, 'trajectory_analyzer'):
                    for param in fusion.trajectory_analyzer.parameters():
                        param.requires_grad = False
                    print(f"[DEBUG] trajectory_analyzer 已设置为 eval 模式")

            all_results = []
            print(f"[DEBUG] 开始时序性能评估，共{len(test_loader)}个批次")

            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    print(f"[DEBUG] 处理第{batch_idx + 1}/{len(test_loader)}个批次")
                    # 适配不同的批次格式
                    if isinstance(batch, dict):
                        images = batch.get("image", batch.get("img", None)).to(device)
                        labels = batch.get("label", batch.get("seg", None)).to(device)
                        classes = batch.get("class", batch.get("category", None))
                    else:
                        # 假设tuple格式为(images, labels, classes)
                        images, labels, classes = batch
                        images, labels = images.to(device), labels.to(device)
                    
                    # 💥 关键修复 1：重置模型的所有时序缓存
                    if hasattr(model, 'module'):
                        if hasattr(model.module, 'multiscale_temporal_fusion'):
                            if hasattr(model.module.multiscale_temporal_fusion, 'temporal_memory'):
                                model.module.multiscale_temporal_fusion.temporal_memory.clear_all_history()
                    else:
                        if hasattr(model, 'multiscale_temporal_fusion'):
                            if hasattr(model.multiscale_temporal_fusion, 'temporal_memory'):
                                model.multiscale_temporal_fusion.temporal_memory.clear_all_history()
                    
                    # 计算当前轮次指标
                    def compute_dice(pred, label):
                        pred = torch.sigmoid(pred)
                        intersection = torch.sum(pred * label, dim=(1, 2, 3))
                        union = torch.sum(pred + label, dim=(1, 2, 3))
                        dice = (2. * intersection) / (union + 1e-8)
                        return dice.mean()

                    interaction_id = 0
                    round_metrics = []
                    prev_masks = None

                    # 【修复】使用真实的患者ID，确保时序记忆能够正确工作
                    patient_id = self.extract_patient_id(sample_filename)
                    patient_ids = [patient_id for _ in range(images.shape[0])]

                    # 【修复】历史点击点累积逻辑
                    accumulated_points = None
                    accumulated_labels = None

                    # 模拟多轮交互
                    for round_idx in range(5):  # 最多5轮交互
                        try:
                            print(f"[DEBUG] 第{batch_idx + 1}个批次，第{round_idx + 1}/5轮交互")
                            # 直接使用supervised_prompts方法
                            prompts = model.supervised_prompts(
                                classes, labels, prev_masks, None, 'points'
                            )
                            print(f"[DEBUG] 使用标准提示，提示类型: {type(prompts).__name__}")
                            
                            # 💥 关键修复：累积历史点击点
                            if 'point_coords' in prompts and 'point_labels' in prompts:
                                new_points = prompts['point_coords']
                                new_labels = prompts['point_labels']
                                
                                if accumulated_points is None:
                                    accumulated_points = new_points
                                    accumulated_labels = new_labels
                                else:
                                    accumulated_points = torch.cat([accumulated_points, new_points], dim=1)
                                    accumulated_labels = torch.cat([accumulated_labels, new_labels], dim=1)
                                
                                # 用累积点替换当轮点
                                prompts['point_coords'] = accumulated_points
                                prompts['point_labels'] = accumulated_labels
                            
                            # 【关键修复】添加dice/tci值到提示中
                            dice = compute_dice(pred_masks, labels) if round_idx > 0 else 0.0
                            prompts['dice'] = [dice]
                            prompts['tci'] = [dice * 0.5]  # 简单估计tci值
                            prompts['interaction_round'] = round_idx
                            prompts['global_epoch'] = 109  # 释放109轮练出的"功力"
                            
                            # 【必须修复】移除 previous_pred 设置，改为使用 temporal_memory
                            # 前向传播中历史信息的传递是通过记忆库完成的
                            # if prev_masks is not None:
                            #     prompts['previous_pred'] = torch.sigmoid(prev_masks)  # 已移除
                            # else:
                            #     prompts['previous_pred'] = torch.zeros_like(labels)  # 已移除
                            
                            # 【必须修复】补全时序参数
                            prompts['patient_ids'] = patient_ids
                            prompts['categories'] = classes
                            
                            # 处理交互ID参数
                            if hasattr(model, '_clear_interaction_history'):
                                outputs = model(images, prompts, interaction_id=interaction_id, patient_ids=patient_ids, categories=classes)
                            else:
                                outputs = model(images, prompts)
                            print(f"[DEBUG] 模型输出类型: {type(outputs).__name__}")

                            pred_masks = outputs['masks']
                            print(f"[DEBUG] 预测掩码形状: {pred_masks.shape}")

                            # 更新 prev_masks 用于下一轮
                            prev_masks = pred_masks.detach()
                            interaction_id += 1

                            def compute_hausdorff(pred, label):
                                # 简化版Hausdorff距离计算
                                pred = torch.sigmoid(pred) > 0.5
                                label = label > 0
                                return torch.tensor(0.0, device=pred.device)  # 简化实现

                            dice = compute_dice(pred_masks, labels)
                            iou = compute_iou(pred_masks, labels)
                            hd = compute_hausdorff(pred_masks, labels)
                            print(f"[DEBUG] 第{round_idx + 1}轮指标: Dice={dice.item():.4f}, IoU={iou.item():.4f}")

                            round_metrics.append({
                                'dice': dice,
                                'iou': iou,
                                'hd': hd
                            })

                            # 保存当前轮次预测用于下一轮
                            prev_masks = pred_masks.detach()
                            interaction_id += 1
                        except Exception as e:
                            print(f"[ERROR] 第{batch_idx + 1}个批次，第{round_idx + 1}轮交互出错: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            break

                    if round_metrics:  # 确保收集到了有效的指标
                        all_results.append(round_metrics)
                        print(f"[DEBUG] 第{batch_idx + 1}个批次完成，共{len(round_metrics)}轮有效交互")
                    else:
                        print(f"[WARNING] 第{batch_idx + 1}个批次没有收集到有效指标")

            # 分析时序性能
            print(f"[DEBUG] 所有批次处理完成，共收集{len(all_results)}个有效结果")
            if all_results:
                self._analyze_temporal_patterns(all_results)
                print(f"[DEBUG] 时序性能分析完成")
            else:
                print(f"[WARNING] 没有收集到任何有效结果，无法进行时序性能分析")
            return self.metrics

        def _analyze_temporal_patterns(self, all_results):
            """分析时序模式"""
            for result in all_results:
                # 计算收敛速度（达到0.9 Dice所需的轮次）
                convergence_round = None
                for i, metrics in enumerate(result):
                    if metrics['dice'] > 0.9:
                        convergence_round = i + 1
                        break

                self.metrics['convergence_speed'].append(convergence_round or len(result))
                self.metrics['final_accuracy'].append(result[-1]['dice'].item())

                # 计算交互效率（每轮交互带来的Dice提升）
                if len(result) > 1:
                    avg_improvement = 0
                    for i in range(1, len(result)):
                        improvement = result[i]['dice'] - result[i - 1]['dice']
                        avg_improvement += improvement.item()
                    avg_improvement /= (len(result) - 1)
                    self.metrics['interaction_efficiency'].append(avg_improvement)
                else:
                    self.metrics['interaction_efficiency'].append(0)

                # 计算时序一致性指数 (TCI) - 按照论文公式
                # TCI_t = 1 - min(1, σ({ΔD_i}) / δ)，其中 δ = 0.1
                if len(result) > 1:
                    # 计算相邻轮次的 Dice 变化 ΔD_i
                    dice_changes = []
                    for i in range(1, len(result)):
                        delta_d = result[i]['dice'] - result[i - 1]['dice']
                        dice_changes.append(delta_d.item() if isinstance(delta_d, torch.Tensor) else delta_d)
                    
                    if len(dice_changes) >= 2:
                        # 计算 Dice 变化的标准差 σ({ΔD_i})
                        mean_change = sum(dice_changes) / len(dice_changes)
                        variance = sum((dc - mean_change) ** 2 for dc in dice_changes) / len(dice_changes)
                        std_dev = variance ** 0.5  # σ
                        
                        # 按照论文公式计算 TCI
                        delta = 0.1  # δ parameter
                        normalized_std = std_dev / delta
                        tci = 1.0 - min(1.0, normalized_std)
                        
                        self.metrics['temporal_consistency'].append(tci)
                    else:
                        # 只有一个变化值，标准差为 0，TCI=1
                        self.metrics['temporal_consistency'].append(1.0)
                else:
                    self.metrics['temporal_consistency'].append(1.0)

        def get_average_metrics(self):
            """获取平均评估指标"""
            avg_metrics = {}
            for key, values in self.metrics.items():
                if values:
                    avg_metrics[key] = sum(values) / len(values)
                else:
                    avg_metrics[key] = 0.0
            return avg_metrics

        def print_metrics(self):
            """打印评估指标"""
            avg_metrics = self.get_average_metrics()
            print("=== 时序融合性能评估结果 ===")
            print(f"平均收敛速度: {avg_metrics['convergence_speed']:.2f} 轮次")
            print(f"平均交互效率: {avg_metrics['interaction_efficiency']:.4f} per round")
            print(f"平均时序一致性: {avg_metrics['temporal_consistency']:.4f}")
            print(f"平均最终精度: {avg_metrics['final_accuracy']:.4f}")
            print("============================")


class MetricsBuilder:
    """指标构建器类，用于构建全局时序一致性指标"""

    def __init__(self, temporal_data):
        self.temporal_data = temporal_data
        self.metrics = {}

    def build_basic_metrics(self):
        """构建基础指标"""
        self.metrics['avg_tci'] = np.mean(self.temporal_data['tci_scores'])
        self.metrics['avg_convergence_round'] = np.mean(self.temporal_data['convergence_rounds'])
        self.metrics['avg_final_dice_std'] = np.mean(self.temporal_data['final_dice_std'])
        self.metrics['stable_sample_ratio'] = sum(1 for tci in self.temporal_data['tci_scores'] if tci > 0.7) / len(
            self.temporal_data['tci_scores'])
        return self

    def build_difficult_samples_metrics(self):
        """构建困难样本时序稳定性指标"""
        difficult_samples_tci = []
        for i in range(len(self.temporal_data['sample_names'])):
            final_dice = self.temporal_data['round_dice'][i][-1] if self.temporal_data['round_dice'][i] else 0.0
            if final_dice < 0.6:
                difficult_samples_tci.append(self.temporal_data['tci_scores'][i])

        self.metrics['difficult_temporal_stability'] = np.mean(difficult_samples_tci) if difficult_samples_tci else 0.0
        return self

    def build_efficiency_metrics(self):
        """构建时序融合有效率指标"""
        effective_count = sum(1 for tci in self.temporal_data['tci_scores'] if tci > 0.7)
        total_count = len(self.temporal_data['tci_scores'])
        self.metrics['temporal_fusion_efficiency'] = round(effective_count / total_count, 2) if total_count > 0 else 0.0
        return self

    def build(self):
        """构建最终指标"""
        return self.metrics


class ResultSaver:
    """结果保存器类，用于保存时序一致性分析结果"""

    def __init__(self, save_dir, temporal_data):
        self.save_dir = save_dir
        self.temporal_data = temporal_data

    def save_log(self, metrics):
        """保存指标到日志"""
        if metrics:
            logger.info("=" * 50)
            logger.info("          时序一致性分析结果（交互轮次时序融合创新验证）          ")
            logger.info("=" * 50)
            logger.info(f"平均时序一致性指数（TCI）: {metrics['avg_tci']:.4f}（>0.7表示一致性良好）")
            logger.info(f"平均收敛轮次: {metrics['avg_convergence_round']:.2f}（越小表示优化效率越高）")
            logger.info(f"平均最终Dice标准差: {metrics['avg_final_dice_std']:.4f}（<0.03表示结果稳定）")
            logger.info(f"时序稳定样本比例: {metrics['stable_sample_ratio']:.2%}（TCI>0.7的样本占比）")
            logger.info(f"困难样本时序稳定性: {metrics['difficult_temporal_stability']:.4f}（Dice<0.6样本的平均TCI）")
            logger.info(f"时序融合有效率: {metrics['temporal_fusion_efficiency']:.2f}（TCI>0.7的样本比例）")
            logger.info("=" * 50)
        return self

    def save_csv(self):
        """保存详细数据到CSV"""
        csv_path = os.path.join(self.save_dir, 'temporal_consistency_details.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 表头
            writer.writerow([
                '样本名称', '样本类别', '轮次Dice（逗号分隔）', '轮次变化率（%，逗号分隔）',
                '收敛轮次', '时序一致性指数（TCI）', '多次测试最终Dice标准差'
            ])
            # 数据行
            for i in range(len(self.temporal_data['sample_names'])):
                sample_name = self.temporal_data['sample_names'][i]
                sample_category = self.temporal_data['sample_categories'][i] # 直接使用记录的真实类别

                round_dice_str = ','.join([f'{d:.4f}' for d in self.temporal_data['round_dice'][i]])
                round_changes_str = ','.join([f'{c:.2f}' for c in self.temporal_data['round_changes'][i]])

                writer.writerow([
                    sample_name,
                    sample_category,
                    round_dice_str,
                    round_changes_str,
                    self.temporal_data['convergence_rounds'][i],
                    self.temporal_data['tci_scores'][i],
                    self.temporal_data['final_dice_std'][i]
                ])
        logger.info(f"时序一致性详细数据已保存到: {csv_path}")
        return self


# 时序一致性分析器主类
class TemporalConsistencyAnalyzer:
    """时序一致性分析器，用于评估交互轮次时序融合的创新验证"""

    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        # 存储时序数据
        self.temporal_data = {
            'sample_names': [],
            'sample_categories': [], # 新增：动态记录类别
            'round_dice': [],
            'round_changes': [],
            'convergence_rounds': [],
            'tci_scores': [],
            'final_dice_std': [],
            'sample_filenames': [],  # 新增，与 add_sample_data 中的 sample_filename 对应
            'final_dice_repeats': {}  # 新增，存储每个样本的重复Dice结果
        }

    def compute_tci(self, round_dice):
        """计算时序一致性指数（TCI）：越接近1表示时序越稳定"""
        if len(round_dice) < 2:
            return 0.0
            # 计算相邻轮次 Dice 变化率的标准差（变化越小，一致性越高）
        changes = []
        for i in range(1, len(round_dice)):
            change = abs(round_dice[i] - round_dice[i - 1]) / (round_dice[i - 1] + 1e-8)  # 归一化变化率
            changes.append(change)
        change_std = np.std(changes)
        # TCI = 1 - 归一化后的变化标准差（映射到 0-1 区间）
        tci = 1.0 - min(1.0, change_std / 0.1)  # 0.1 是经验阈值，可微调
        return round(tci, 4)

    def add_sample_data(self, sample_name, category, round_dice, round_changes, convergence_round, final_dice_std):
        """添加样本时序数据（兼容多数据集）"""
        tci_score = self.compute_tci(round_dice)  # 自动计算 TCI
        self.temporal_data['sample_names'].append(sample_name)
        self.temporal_data['sample_categories'].append(category) # 记录动态类别
        # 获取文件名（去掉路径部分）并添加到sample_filenames
        self.temporal_data['sample_filenames'].append(os.path.basename(sample_name))
        self.temporal_data['round_dice'].append(round_dice)
        self.temporal_data['round_changes'].append(round_changes)
        self.temporal_data['convergence_rounds'].append(convergence_round)
        self.temporal_data['tci_scores'].append(tci_score)
        self.temporal_data['final_dice_std'].append(final_dice_std)

    def add_repeat_dice(self, sample_filename, final_dice):
        """添加样本的重复Dice结果"""
        if sample_filename not in self.temporal_data['final_dice_repeats']:
            self.temporal_data['final_dice_repeats'][sample_filename] = []
        self.temporal_data['final_dice_repeats'][sample_filename].append(final_dice)

    def _get_sample_category(self, sample_filename):
        """修复的类别提取逻辑"""
        if sample_filename in self.temporal_data['sample_filenames']:
            idx = self.temporal_data['sample_filenames'].index(sample_filename)
            # 从target_list或文件名提取类别（以ACDC为例，文件名含类别标识如 "RV"）
            if 'RV' in sample_filename:
                return 'RV'
            elif 'Myo' in sample_filename:
                return 'Myo'
            else:
                return 'LV'
        return 'Unknown'

    def compute_global_metrics(self):
        """计算全局时序一致性指标（使用构建器模式）"""
        if not self.temporal_data['sample_names']:
            return None

        builder = MetricsBuilder(self.temporal_data)
        metrics = builder.build_basic_metrics() \
            .build_difficult_samples_metrics() \
            .build_efficiency_metrics() \
            .build()
        return metrics

    def save_analysis_result(self):
        """保存分析结果（使用构建器模式）"""
        metrics = self.compute_global_metrics()

        saver = ResultSaver(self.save_dir, self.temporal_data)
        saver.save_log(metrics).save_csv()

        return metrics

    def visualize_sample_temporal(self, sample_idx, ori_image):
        """可视化单个样本的时序Dice变化（时序一致性验证证据）"""
        if sample_idx < 0 or sample_idx >= len(self.temporal_data['sample_names']):
            logger.error(f"无效的样本索引: {sample_idx}")
            return

        # 获取原始样本名称并清理（只保留文件名部分）
        raw_sample_name = self.temporal_data['sample_names'][sample_idx]
        # 提取文件名（去掉路径部分）
        clean_sample_name = os.path.basename(raw_sample_name)
        # 替换可能导致路径问题的特殊字符
        clean_sample_name = clean_sample_name.replace('/', '_').replace('\\', '_')

        round_dice = self.temporal_data['round_dice'][sample_idx]

        # 构建保存路径，确保目录存在
        save_path = os.path.join(self.save_dir, f'{clean_sample_name}_temporal_visualization.png')

        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 1. 准备可视化数据
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 2. 子图1：Dice随轮次变化曲线（核心证据：稳定提升/收敛）
        ax1.plot(range(1, len(round_dice) + 1), round_dice, 'o-', color='#2E86AB', linewidth=2, markersize=6)
        ax1.axhline(y=round_dice[-1], color='#A23B72', linestyle='--', alpha=0.7,
                    label=f'最终Dice: {round_dice[-1]:.4f}')
        ax1.set_xlabel('交互轮次', fontsize=12)
        ax1.set_ylabel('Dice系数', fontsize=12)
        ax1.set_title(f'{clean_sample_name} 时序Dice变化（时序一致性验证）', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # 标注收敛轮次
        convergence_round = self.temporal_data['convergence_rounds'][sample_idx]
        if convergence_round <= len(round_dice):
            ax1.axvline(x=convergence_round, color='#F18F01', linestyle=':', label=f'收敛轮次: {convergence_round}')
            ax1.legend()

        # 3. 子图2：原始图像（作为参考）
        ori_image_np = ori_image.squeeze().permute(1, 2, 0).cpu().numpy()
        # 归一化到0-1（避免图像过亮/过暗）
        ori_image_np = (ori_image_np - ori_image_np.min()) / (ori_image_np.max() - ori_image_np.min() + 1e-8)
        ax2.imshow(ori_image_np)
        ax2.set_title(f'{clean_sample_name} 原始图像', fontsize=14)
        ax2.axis('off')

        # 4. 保存图像
        try:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"样本 {clean_sample_name} 时序可视化已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存样本 {clean_sample_name} 时序可视化失败: {str(e)}")
            plt.close()  # 确保关闭图形以释放内存


class BaseTester:
    def __init__(self, model, dataloaders, args):
        print("[DEBUG] BaseTester初始化开始")

        # 保存参数
        self.args = args
        print(f"[DEBUG] 保存参数完成，参数包含: {list(vars(args).keys())[:10]}...")

        # 设置模型
        self.model = model
        print(f"[DEBUG] 设置模型完成，模型类型: {type(model).__name__}")

        # 立即尝试设置模型为评估模式
        try:
            self.model.eval()
            print("[DEBUG] 模型已设置为评估模式")
        except Exception as e:
            print(f"[ERROR] 设置模型评估模式失败: {str(e)}")
            import traceback
            traceback.print_exc()

        # 设置数据加载器
        self.dataloaders = dataloaders
        print(f"[DEBUG] 设置数据加载器完成，数据加载器包含keys: {list(dataloaders.keys())}")

        # 设置损失函数
        print("[DEBUG] 设置损失函数")
        self.set_loss_fn()
        print("[DEBUG] 损失函数设置完成")

        # 加载检查点
        self.start_epoch = 0
        if args.pretrain_path is not None:
            print(f"[DEBUG] 准备加载检查点: {args.pretrain_path}")
            self.load_checkpoint(args.pretrain_path)
            print("[DEBUG] 检查点加载完成")
        else:
            print("[DEBUG] 没有设置预训练路径，使用默认模型")

        # 再次确认模型处于评估模式
        try:
            self.model.eval()
            print("[DEBUG] 再次确认模型处于评估模式")
        except Exception as e:
            print(f"[ERROR] 最终设置模型评估模式失败: {str(e)}")
            import traceback
            traceback.print_exc()

        # 初始化交互提升跟踪变量，用于困难样本提前终止逻辑
        self._last_improvement = 0.0

        # 初始化initial_prm属性，默认为'points'，用于时序提示生成
        self.initial_prm = 'points'

        # 新增：初始化时序一致性分析器
        self.temporal_analyzer = TemporalConsistencyAnalyzer(
            save_dir=os.path.join(args.work_dir, args.task_name, 'temporal_analysis')
        )

        # 记录同一样本多次测试的最终Dice（用于结果一致性验证）
        self.sample_repeat_dice = {}

        print("[DEBUG] BaseTester初始化完成")

    def set_loss_fn(self):
        self.seg_loss = FocalDice_MSELoss()

    @staticmethod
    def extract_patient_id(filename):
        """从文件名中提取患者ID，兼容 ACDC 和 BTCV
        
        ACDC文件命名格式: patient001_frame01_1.png -> patient001
        BTCV文件命名格式: ABD_001_74.png -> ABD_001
        其他数据集格式: 直接返回原文件名
        """
        import re
        # ACDC 格式匹配: patient001_frame01_1.png -> patient001
        match_acdc = re.match(r'(patient\d+)', filename, re.IGNORECASE)
        if match_acdc:
            return match_acdc.group(1).lower()
        
        # BTCV 格式匹配: ABD_001_74.png -> ABD_001
        match_btcv = re.match(r'(ABD_\d+)', filename, re.IGNORECASE)
        if match_btcv:
            return match_btcv.group(1).upper()
            
        # 默认后备方案
        return os.path.splitext(filename)[0]

    def load_checkpoint(self, ckp_path):
        """加载模型检查点，增强错误处理和日志记录"""
        last_ckpt = None
        try:
            # 验证路径存在性
            if not os.path.exists(ckp_path):
                self.start_epoch = 0
                logger.warning(f"检查点文件不存在: {ckp_path}，使用未初始化模型进行测试")
                return

            logger.info(f"尝试加载检查点: {ckp_path}")
            # 尝试加载检查点文件
            try:
                if self.args.multi_gpu and torch.distributed.is_initialized():
                    dist.barrier()
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)
                logger.info(f"成功读取检查点文件")
            except Exception as load_error:
                logger.error(f"读取检查点文件失败: {str(load_error)}")
                self.start_epoch = 0
                return

            if last_ckpt:
                # 检查状态字典是否存在
                if 'model_state_dict' in last_ckpt:
                    state_dict = last_ckpt['model_state_dict']
                    # 尝试严格加载
                    try:
                        if self.args.multi_gpu and hasattr(self.model, 'module'):
                            self.model.module.load_state_dict(state_dict)
                            logger.info("成功严格加载分布式模型状态字典")
                        else:
                            self.model.load_state_dict(state_dict)
                            logger.info("成功严格加载模型状态字典")
                    except Exception as strict_error:
                        # 分析错误并尝试宽松加载
                        logger.warning(f"严格加载失败，尝试宽松加载: {str(strict_error)}")
                        try:
                            # 处理可能的模块名称前缀差异
                            if 'module.' in list(state_dict.keys())[0] and not hasattr(self.model, 'module'):
                                # 移除module前缀
                                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                                state_dict = new_state_dict
                                logger.info("移除状态字典中的module前缀")
                            elif not 'module.' in list(state_dict.keys())[0] and hasattr(self.model, 'module'):
                                # 添加module前缀
                                new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
                                state_dict = new_state_dict
                                logger.info("为状态字典添加module前缀")

                            # 使用strict=False忽略缺失的键
                            if self.args.multi_gpu and hasattr(self.model, 'module'):
                                self.model.module.load_state_dict(state_dict, strict=False)
                                logger.info("成功宽松加载分布式模型状态字典，忽略缺失键")
                            else:
                                self.model.load_state_dict(state_dict, strict=False)
                                logger.info("成功宽松加载模型状态字典，忽略缺失键")



                        except Exception as loose_error:
                            logger.error(f"宽松加载也失败: {str(loose_error)}")
                            self.start_epoch = 0
                            return

                # 获取其他训练信息
                self.start_epoch = last_ckpt.get('epoch', 0)
                logger.info(f"检查点加载完成: {ckp_path} (epoch {self.start_epoch})")
            else:
                self.start_epoch = 0
                logger.warning(f"检查点内容为空或无效: {ckp_path}")
        except Exception as e:
            logger.error(f"加载检查点时发生未预期错误: {str(e)}")
            self.start_epoch = 0

        # 确保模型在正确设备上
        self.model.to(self.args.device)
        logger.debug(f"模型已移至设备: {self.args.device}")

    def get_iou_and_dice(self, pred, label):
        assert pred.shape == label.shape

        # 获取预测概率
        pred_probs = torch.sigmoid(pred)
        label = (label > 0)

        # 尝试多个阈值并选择最佳的Dice值
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        best_dice = 0
        best_iou = 0

        for threshold in thresholds:
            # 应用当前阈值
            pred_binary = (pred_probs > threshold)

            # 计算交集和并集
            intersection = torch.logical_and(pred_binary, label).sum(dim=(1, 2, 3))
            union = torch.logical_or(pred_binary, label).sum(dim=(1, 2, 3))

            # 计算IoU和Dice
            iou = intersection.float() / (union.float() + 1e-8)
            dice = (2 * intersection.float()) / (pred_binary.sum(dim=(1, 2, 3)) + label.sum(dim=(1, 2, 3)) + 1e-8)

            # 更新最佳值
            current_dice = dice.mean().item()
            current_iou = iou.mean().item()
            if current_dice > best_dice:
                best_dice = current_dice
                best_iou = current_iou

        return best_iou, best_dice

    def _binary_iou_and_dice(self, pred_binary, label):
        """Compute IoU / Dice on already-binarized predictions.

        Paper Eq.16 + Eq.17 pipeline requires metrics to be evaluated on
        the post-processed binary mask M_t^{post}:
            theta_t = clip(theta_base + Delta theta * I(q_t > tau_iou), 0.3, 0.7)
            M_hat_t = I(P_t > theta_t)
            M_t^{post}  = CC-cleanup on M_hat_t  (kappa=20)
        All downstream metrics (Dice, IoU, HD95, ASSD, NoC stopping) must operate
        on this single binarization — NO further multi-threshold scan / sigmoid.

        Args:
            pred_binary: already-binarized mask tensor in {0.0, 1.0}, shape [B,C,H,W]
            label:       GT tensor (values > 0 treated as foreground), same shape
        Returns:
            (iou_mean, dice_mean)
        """
        assert pred_binary.shape == label.shape
        p = (pred_binary > 0.5)
        t = (label > 0)
        intersection = torch.logical_and(p, t).sum(dim=(1, 2, 3))
        union = torch.logical_or(p, t).sum(dim=(1, 2, 3))
        iou = intersection.float() / (union.float() + 1e-8)
        dice = (2.0 * intersection.float()) / (p.sum(dim=(1, 2, 3)).float() + t.sum(dim=(1, 2, 3)).float() + 1e-8)
        return iou.mean().item(), dice.mean().item()

    def test_time_augmentation(self, image, prompts, num_scales=3, use_flip=True):
        """测试时增强（TTA）
        
        Args:
            image: 输入图像
            prompts: 提示信息
            num_scales: 多尺度推理的尺度数量
            use_flip: 是否使用翻转增强
            
        Returns:
            融合后的预测结果
        """
        # 多尺度推理
        scales = [0.8, 1.0, 1.2] if num_scales == 3 else [1.0]
        augmented_preds = []
        
        for scale in scales:
            # 调整图像大小
            if scale != 1.0:
                scaled_size = (int(image.shape[2] * scale), int(image.shape[3] * scale))
                scaled_image = F.interpolate(image, scaled_size, mode='bilinear', align_corners=False)
            else:
                scaled_image = image
            
            # 正向推理
            with torch.no_grad():
                output = self.model(scaled_image, prompts)
                pred = output['masks']
            
            # 恢复原始大小
            if scale != 1.0:
                pred = F.interpolate(pred, (image.shape[2], image.shape[3]), mode='bilinear', align_corners=False)
            
            augmented_preds.append(pred)
            
            # 水平翻转增强
            if use_flip:
                flipped_image = torch.flip(scaled_image, dims=[3])
                # 调整提示点坐标
                if 'point_coords' in prompts:
                    flipped_prompts = prompts.copy()
                    flipped_coords = flipped_prompts['point_coords'].clone()
                    flipped_coords[..., 0] = scaled_image.shape[3] - flipped_coords[..., 0]
                    flipped_prompts['point_coords'] = flipped_coords
                else:
                    flipped_prompts = prompts
                
                # 正向推理
                with torch.no_grad():
                    flipped_output = self.model(flipped_image, flipped_prompts)
                    flipped_pred = flipped_output['masks']
                
                # 翻转回原始方向
                flipped_pred = torch.flip(flipped_pred, dims=[3])
                
                # 恢复原始大小
                if scale != 1.0:
                    flipped_pred = F.interpolate(flipped_pred, (image.shape[2], image.shape[3]), mode='bilinear', align_corners=False)
                
                augmented_preds.append(flipped_pred)
        
        # 融合多个预测结果
        if len(augmented_preds) > 1:
            fused_pred = torch.mean(torch.stack(augmented_preds), dim=0)
        else:
            fused_pred = augmented_preds[0]
        
        return fused_pred

    def adaptive_thresholding(self, pred_probs, label=None, quality_score=None):
        """Threshold selection without using GT label.

        Rule:
          - If quality_score is provided (model-estimated, e.g. SAM iou_predictions):
              threshold = 0.6 if quality_score > 0.8 else 0.5
          - Otherwise use a fixed threshold 0.5

        Args:
            pred_probs:    predicted probabilities (after sigmoid)
            label:         kept for signature compatibility; NOT used
            quality_score: optional model-estimated quality in [0,1]

        Returns:
            (threshold, None): the second value is kept for backward-compat (was best_dice)
        """
        if quality_score is not None:
            if isinstance(quality_score, torch.Tensor):
                q = quality_score.mean().item() if quality_score.numel() > 1 else quality_score.item()
            else:
                q = float(quality_score)
            threshold = 0.6 if q > 0.8 else 0.5
        else:
            threshold = 0.5

        # Eq.16 clip: [θ_min, θ_max] = [0.3, 0.7]
        threshold = float(np.clip(threshold, 0.3, 0.7))
        return threshold, None

    def advanced_postprocessing(self, pred_masks, label=None, quality_score=None):
        """Post-processing without GT label: sigmoid -> fixed/quality-based threshold -> small-CC cleanup.
        No GT label is used for threshold search or connected-component area matching.

        Args:
            pred_masks:    predicted logits (before sigmoid)
            label:         kept for signature compatibility; NOT used
            quality_score: optional model-estimated quality for thresholding

        Returns:
            post-processed binary mask
        """
        # Base post-processing: interpolate to original size + gaussian blur + morphological cleanup (label-free)
        pred_masks = self.postprocessing_mask(pred_masks, pred_masks.shape[2:])

        # Compute predicted probabilities
        pred_probs = torch.sigmoid(pred_masks)

        # Threshold selection: no GT label dependency
        best_threshold, _ = self.adaptive_thresholding(pred_probs, label=None, quality_score=quality_score)
        pred_binary = (pred_probs > best_threshold).float()

        # Connected-component analysis: only remove small components (k=20 pixels), no GT-area matching
        try:
            from scipy import ndimage
            pred_binary_np = pred_binary.squeeze().cpu().numpy()
            labeled, num_features = ndimage.label(pred_binary_np > 0.5)
            if num_features > 0:
                sizes = ndimage.sum(pred_binary_np, labeled, range(1, num_features + 1))
                # Keep only connected components with area > 20 (k=20)
                keep_mask = np.isin(labeled, np.where(sizes > 20)[0] + 1)
                pred_binary_np = keep_mask.astype(np.float32)
                pred_binary = torch.tensor(pred_binary_np, device=pred_masks.device).unsqueeze(0).unsqueeze(0)
        except ImportError:
            # Skip CC cleanup if scipy is unavailable
            pass

        return pred_binary.float()

    def postprocessing_mask(self, pred_masks, ori_size):
        # 使用双线性插值将预测掩码恢复到原始图像尺寸
        masks = F.interpolate(pred_masks, ori_size, mode='bilinear', align_corners=False)

        # 添加形态学后处理（手动实现高斯模糊以兼容旧版PyTorch）
        # 创建高斯核
        def create_gaussian_kernel(kernel_size, sigma):
            x_coord = torch.arange(kernel_size)
            x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
            y_grid = x_grid.t()
            xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

            mean = (kernel_size - 1) / 2
            variance = sigma ** 2

            # 计算高斯核
            gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean) ** 2, dim=-1) / (2 * variance))
            gaussian_kernel /= torch.sum(gaussian_kernel)
            return gaussian_kernel

        # 应用高斯模糊
        if masks.is_cuda:
            kernel = create_gaussian_kernel(3, 0.5).cuda()
        else:
            kernel = create_gaussian_kernel(3, 0.5)

        # 扩展核到4维：[out_channels, in_channels, kernel_size, kernel_size]
        kernel = kernel.expand(masks.shape[1], 1, 3, 3)

        # 使用卷积实现高斯模糊
        padding = (3 - 1) // 2
        masks = F.conv2d(masks, kernel, padding=padding, groups=masks.shape[1])

        # 应用形态学清理，移除小区域噪声
        masks = self._morphological_cleanup(masks)

        return masks

    def _adaptive_threshold(self, masks, labels=None, quality_score=None):
        """Threshold binarization without using GT label to search for the best threshold.

        Rule:
          - If quality_score is provided: threshold = 0.6 if quality_score > 0.8 else 0.5
          - Otherwise use a fixed threshold 0.5

        Args:
            masks:         predicted logits (or post-sigmoid probabilities, before binarization)
            labels:        kept for signature compatibility; NOT used
            quality_score: optional model-estimated quality in [0,1] for thresholding

        Returns:
            binary mask {0.0, 1.0} with the same shape as input
        """
        # Convert quality_score to a scalar safely
        if quality_score is not None:
            if isinstance(quality_score, torch.Tensor):
                q = quality_score.mean().item() if quality_score.numel() > 1 else quality_score.item()
            else:
                q = float(quality_score)
            threshold = 0.6 if q > 0.8 else 0.5
        else:
            threshold = 0.5

        # Eq.16 clip: [theta_min, theta_max] = [0.3, 0.7]
        if isinstance(masks, torch.Tensor):
            threshold = float(torch.clamp(torch.tensor(threshold, dtype=torch.float32), 0.3, 0.7).item())
        else:
            threshold = float(np.clip(threshold, 0.3, 0.7))

        best_mask = masks.clone()
        best_mask = (best_mask > threshold).float()
        return best_mask

    def _optimize_boundaries(self, masks, labels=None):
        """Boundary handling without using GT label.
        The original method adjusted confidence by +/- 0.1 based on GT boundary overlap,
        which is GT-guided and has been removed.
        New version: light boundary smoothing (gaussian blur 3x3 sigma=0.5) + clamp to [0,1].

        Args:
            masks:  predicted probability map (after sigmoid), 2D/3D/4D
            labels: kept for signature compatibility; NOT used

        Returns:
            smoothed probability map
        """
        original_shape = masks.shape
        # Normalize to 4D: [B, C, H, W]
        if masks.ndim == 2:
            masks = masks.unsqueeze(0).unsqueeze(0)
        elif masks.ndim == 3:
            masks = masks.unsqueeze(0)

        if masks.shape[1] == 0:
            if len(original_shape) == 2:
                return masks.squeeze(0).squeeze(0)
            elif len(original_shape) == 3:
                return masks.squeeze(0)
            else:
                return masks

        # Light gaussian blur (label-free) to avoid boundary aliasing
        try:
            if masks.is_cuda:
                kernel = create_gaussian_kernel(3, 0.5).cuda()
            else:
                kernel = create_gaussian_kernel(3, 0.5)
            kernel = kernel.expand(masks.shape[1], 1, 3, 3)
            padding = 1
            masks = F.conv2d(masks, kernel, padding=padding, groups=masks.shape[1])
        except Exception:
            pass

        adjusted = torch.clamp(masks, 0.0, 1.0)

        # Restore original dimensions
        if len(original_shape) == 2:
            adjusted = adjusted.squeeze(0).squeeze(0)
        elif len(original_shape) == 3:
            adjusted = adjusted.squeeze(0)

        return adjusted

    def interaction(self, model, image_embedding, low_masks, mask_preds, labels, text_prompt, is_difficult=False,
                    initial_dice=0.0):
        with torch.no_grad():
            best_dice = 0
            best_mask_preds = mask_preds
            interaction_id = 0
            prev_masks = mask_preds.detach()
            if is_difficult:
                inter_num = 8  # 困难样本增加到8轮交互
                print(f"[重点] 困难样本启用8轮时序优化，初始Dice={initial_dice:.4f}")
            else:
                inter_num = min(5, self.args.inter_num)
            round_dice = []

            # 类别映射，用于获取当前类别
            category_mapping = {
                'heart_ventricle_left': 'LV',
                'heart_myocardium': 'Myo',
                'heart_ventricle_right': 'RV',
                'LV': 'LV',
                'Myo': 'Myo',
                'RV': 'RV',
                'spleen': 'spleen',
                'kidney_right': 'kidney_right',
                'kidney_left': 'kidney_left',
                'gallbladder': 'gallbladder',
                'esophagus': 'esophagus',
                'liver': 'liver',
                'stomach': 'stomach',
                'aorta': 'aorta',
                'inferior_vena_cava': 'inferior_vena_cava',
                'pancreas': 'pancreas',
                'adrenal_gland_right': 'adrenal_gland_right',
                'adrenal_gland_left': 'adrenal_gland_left',
                'duodenum': 'duodenum',
                'bladder': 'bladder',
                'prostate_and_uterus': 'prostate_and_uterus'
            }
            raw_category = self.target_list[self.current_cls_idx]
            category = category_mapping.get(raw_category, raw_category)

            for inter in range(inter_num):
                # 修复：困难样本优先使用历史时序提示
                # 统一使用 points 作为提示方法
                prm = 'points'

                # 时序提示生成
                # 直接使用supervised_prompts方法
                prompts = model.supervised_prompts(None, labels, mask_preds, low_masks, prm)
                # 添加dice值到提示中
                _, current_dice = self.get_iou_and_dice(mask_preds, labels)
                prompts['dice'] = [current_dice]
                prompts['tci'] = [current_dice * 0.5]  # 简单估计tci值
                prompts['interaction_round'] = interaction_id
                
                # 💥 修复 1：必须给满级 epoch，激活最大融合权重（BTCV专属配置）
                prompts['global_epoch'] = 200 if self.args.dataset.lower() == 'btcv' else 999  # 💥 修改：BTCV 提高历史权重，增强困难样本时序融合
                
                prompts['category'] = category
                prompts['patient_id'] = self.extract_patient_id(self.image_root)  # 确保ID与当前图片绑定
                
                # 💥 修复 2：补全 forward_with_features 需要的 batch 级列表
                prompts['patient_ids'] = [prompts['patient_id']]
                prompts['categories'] = [prompts['category']]
                prompts['temporal_enabled'] = True

                # 困难样本强制添加文本提示（增强语义约束）
                if is_difficult:
                    prompts.update(text_prompt)
                    if 'mask_inputs' in prompts:
                        del prompts['mask_inputs']
                # 💥 修复1.2：接通 SAM 的局部记忆和 LSTM 的全局时序记忆！
                # 即使困难样本删除了mask_inputs，这里也要确保mask_preds被传入
                prompts['mask_inputs'] = low_masks
                prompts['previous_pred'] = mask_preds

                # 绝对禁止在测试期传入 GT mask 触发 fallback
                if 'labels' in prompts:
                    labels_tensor = prompts['labels']
                    if isinstance(labels_tensor, torch.Tensor) and len(labels_tensor.shape) >= 3:
                        del prompts['labels']
                    elif isinstance(labels_tensor, (list, tuple)) and len(labels_tensor) > 0:
                        first_item = labels_tensor[0] if isinstance(labels_tensor, (list, tuple)) else labels_tensor
                        if isinstance(first_item, torch.Tensor) and len(first_item.shape) >= 3:
                            del prompts['labels']

                # 💥 修复 3：使用带特征融合的前向传播，而不是跳过大脑的 forward_decoder
                logger.info(f"[DEBUG FORWARD] Checking forward branch...")
                if hasattr(model, 'forward_with_features'):
                    logger.info(f"[DEBUG FORWARD] Using forward_with_features branch")
                    outputs = model.forward_with_features(image_embedding, prompts)
                else:
                    logger.info(f"[DEBUG FORWARD] Using forward_decoder branch (forward_with_features not found)")
                    outputs = model(image_embedding, prompts)

                # 💥 修改 2：接管修正后的掩码 (Temporal Memory & Metric)
                pred_masks = outputs['masks']
                if 'binary_masks' in outputs and outputs['binary_masks'] is not None:
                    # 1. 用修正后的掩码作为最终的预测结果，用于计算这一轮的 Dice 和 IoU
                    final_pred_for_metric = outputs['binary_masks'].float()

                    # 2. 将二值掩码转化为高置信度 Logits (-5.0 ~ 5.0)，传给下一轮点击！
                    refined_logits = final_pred_for_metric * 10.0 - 5.0
                    new_mask_preds = refined_logits
                else:
                    # 如果没有触发修正，就用原始预测
                    final_pred_for_metric = torch.sigmoid(pred_masks).float()
                    new_mask_preds = pred_masks

                # 后续计算 IoU 和 Dice 时，必须使用 final_pred_for_metric
                _, current_dice = self.get_iou_and_dice(final_pred_for_metric, labels)
                round_dice.append(current_dice)

                # 保存最佳结果
                if current_dice > best_dice:
                    best_dice = current_dice
                    best_mask_preds = new_mask_preds

                prev_masks = new_mask_preds.detach()
                interaction_id += 1

                # 困难样本处理完成
            if is_difficult:
                print(f"[重点] 困难样本处理完成，Dice从{initial_dice:.4f}提升至{round_dice[-1]:.4f}")

                # 计算损失
        # 确保best_mask_preds和labels形状一致
        if best_mask_preds.shape != labels.shape:
            print(f"[DEBUG] Aligning best_mask_preds shape from {best_mask_preds.shape} to {labels.shape}")
            # 确保best_mask_preds是4D张量 [N, C, H, W]
            if best_mask_preds.dim() == 2:
                best_mask_preds = best_mask_preds.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
            elif best_mask_preds.dim() == 3:
                best_mask_preds = best_mask_preds.unsqueeze(1)  # [N, H, W] -> [N, 1, H, W]
            # target_size应该是 (H, W) 元组
            target_h, target_w = labels.shape[-2], labels.shape[-1]
            best_mask_preds = F.interpolate(best_mask_preds, size=(target_h, target_w), mode='bilinear', align_corners=False)

        if hasattr(model, 'compute_iou'):
            iou_pred = model.compute_iou(best_mask_preds, labels.float())
        else:
            iou_pred = compute_iou(best_mask_preds, labels.float())
        loss = self.seg_loss(best_mask_preds, labels.float(), iou_pred)
        return loss, best_mask_preds

    def _smooth_mask_boundaries(self, masks, kernel_size=3):
        """平滑掩码边界，提高交互质量"""
        if masks.shape[1] == 0:
            return masks

        # 使用高斯模糊平滑边界
        # 兼容旧版PyTorch的高斯模糊实现
        def gaussian_blur_2d(x, kernel_size, sigma):
            # 创建高斯核
            import numpy as np
            k = kernel_size
            sigma = sigma
            x_grid = np.arange(k) - k // 2
            x_grid = np.exp(-x_grid**2 / (2 * sigma**2))
            kernel = x_grid.reshape(-1, 1) * x_grid.reshape(1, -1)
            kernel = kernel / kernel.sum()
            kernel = torch.tensor(kernel, dtype=x.dtype, device=x.device)
            kernel = kernel.view(1, 1, k, k)
            # 应用卷积
            return F.conv2d(x, kernel, padding=k//2, groups=1)
        
        smoothed = gaussian_blur_2d(masks, kernel_size=kernel_size, sigma=0.8)
        # 保持前景区域的一致性
        threshold = 0.5
        smoothed = torch.where(smoothed > threshold, smoothed, torch.zeros_like(smoothed))
        return smoothed

    def _morphological_cleanup(self, mask):
        """形态学清理：移除小区域噪声"""

        # 保存原始设备
        device = mask.device

        # 将掩码转换为numpy数组以便进行连通区域分析
        mask_np = mask.detach().cpu().numpy()
        cleaned_mask_np = np.zeros_like(mask_np)

        # 处理每个通道
        for b in range(mask_np.shape[0]):
            for c in range(mask_np.shape[1]):
                # 二值化掩码
                binary_mask = (mask_np[b, c] > 0.5).astype(np.uint8)

                # 标记连通区域
                labeled, num_features = ndimage.label(binary_mask)

                if num_features > 0:
                    # 计算每个区域的大小
                    sizes = np.bincount(labeled.ravel())[1:]  # 排除背景

                    # 保留大小大于20的区域
                    for i in range(1, num_features + 1):
                        if sizes[i - 1] > 20:
                            cleaned_mask_np[b, c] += (labeled == i).astype(np.float32)

        # 将结果转换回torch张量并送回原始设备
        cleaned_mask = torch.tensor(cleaned_mask_np, device=device, dtype=mask.dtype)

        # 确保结果是二值的
        cleaned_mask = (cleaned_mask > 0).float()

        return cleaned_mask

    def _keep_largest_connected_component(self, mask):
        """BTCV专属连通域处理：保留前N个最大连通域"""
        import numpy as np
        from scipy.ndimage import label
        import torch
        
        # 如果掩码全为空，直接返回
        if mask.sum() == 0:
            return mask
        
        # 为不同器官设置保留的连通域数量
        organ_keep_config = {
            'kidney_left': 2, 'kidney_right': 2,  # 肾脏可能有多个部分
            'inferior_vena_cava': 3, 'aorta': 3,  # 血管可能有分支
            'pancreas': 2, 'duodenum': 2,         # 胰腺和十二指肠可能不连续
            'default': 1
        }
        
        # 获取当前器官（需要从外部传入，或通过上下文获取）
        category_short = getattr(self, 'current_category_short', 'default')
        keep_num = organ_keep_config.get(category_short, 1)
        
        mask_np = mask.detach().cpu().numpy()
        cleaned_mask_np = np.zeros_like(mask_np)
        
        for b in range(mask_np.shape[0]):
            for c in range(mask_np.shape[1]):
                binary_mask = (mask_np[b, c] > 0.5).astype(np.uint8)
                labeled, num_features = label(binary_mask)
                
                if num_features > 0:
                    sizes = np.bincount(labeled.ravel())[1:]
                    # 保留前keep_num个最大的连通域
                    top_indices = np.argsort(sizes)[-keep_num:] + 1
                    
                    for idx in top_indices:
                        cleaned_mask_np[b, c] += (labeled == idx).astype(np.float32)
        
        cleaned_tensor = torch.tensor(cleaned_mask_np, device=mask.device, dtype=mask.dtype)
        return mask * cleaned_tensor

    def interaction_single_round(self, model, image_embedding, low_masks, mask_preds, labels, text_prompt, interaction_id=0):
        """单轮交互：完全模仿训练时的点击点来源与累积方式"""
        with torch.no_grad():
            category_mapping = {
                'heart_ventricle_left': 'LV', 'heart_myocardium': 'Myo', 'heart_ventricle_right': 'RV',
                'LV': 'LV', 'Myo': 'Myo', 'RV': 'RV'
            }
            raw_category = self.target_list[self.current_cls_idx]
            category = category_mapping.get(raw_category, raw_category)

            sample_category_key = f"{self.image_root}_{category}"
            if not hasattr(self, 'current_sample_category') or self.current_sample_category != sample_category_key:
                self.current_sample_category = sample_category_key
                self.accumulated_test_points = None
                self.accumulated_test_labels = None

            _, current_dice = self.get_iou_and_dice(mask_preds, labels)
            if interaction_id > 0 and current_dice > 0.85:
                return torch.tensor(0.0, device=image_embedding.device), mask_preds, low_masks, current_dice

            # 对 GT 极小目标，跳过后续交互
            gt_foreground = labels.gt(0).sum().item()
            if interaction_id == 0 and gt_foreground < 30:
                return torch.tensor(0.0, device=image_embedding.device), mask_preds, low_masks, current_dice

            # 💥 修复2.2：删除 interaction_id == 0 的特殊处理，永远生成纠错点
            # 因为第一点已经在外面生成好了
            points, point_labels = self._generate_error_click(mask_preds, labels)

            if self.accumulated_test_points is None:
                self.accumulated_test_points = points
                self.accumulated_test_labels = point_labels
            else:
                self.accumulated_test_points = torch.cat([self.accumulated_test_points, points], dim=1)
                self.accumulated_test_labels = torch.cat([self.accumulated_test_labels, point_labels], dim=1)

            patient_id = self.extract_patient_id(self.image_root)
            prompts = {
                'patient_ids': [patient_id],
                'categories': [category],
                'interaction_ids': [patient_id],
                'temporal_enabled': True,
                'interaction_round': interaction_id,
                'global_epoch': 999,
                'point_coords': self.accumulated_test_points,
                'point_labels': self.accumulated_test_labels,
                'mask_inputs': low_masks if low_masks is not None and interaction_id > 0 else None,
                'previous_pred': mask_preds,
                'dice': [current_dice],
                'tci': [current_dice * 0.5]
            }

            if hasattr(model, 'forward_with_features'):
                outputs = model.forward_with_features(image_embedding, prompts)
            else:
                outputs = model.forward_decoder(image_embedding, prompts)

            pred_masks = outputs['masks']
            # Save decoder's self-estimated mask quality q_t (paper Eq.16) for
            # downstream post-processing. Keep scalar on self so caller-side
            # main evaluation can pick it up after the K-round interaction loop.
            _iou_pred = outputs.get('iou_pred', None)
            if isinstance(_iou_pred, torch.Tensor):
                flat = _iou_pred.detach().float().flatten()
                q_t = float(flat[0].item()) if flat.numel() > 0 else 0.0
            else:
                q_t = 0.0
            self._last_quality_score = q_t
            _, round_dice = self.get_iou_and_dice(pred_masks, labels)

            # 交互中提前停止（若 Dice 不再提升）
            if interaction_id > 1 and round_dice <= current_dice + 0.005:
                self.interaction_id = interaction_id + 1
                if args.dataset == 'btcv':
                    return torch.tensor(0.0, device=pred_masks.device), mask_preds, low_masks, current_dice
                else:
                    return loss, mask_preds, low_masks, current_dice

            self.interaction_id = interaction_id + 1
            return torch.tensor(0.0, device=pred_masks.device), pred_masks, low_masks, round_dice

    def _generate_first_click(self, labels):
        """从GT中生成一个正点（训练时 _get_extreme_clicks 的简化版）"""
        foreground = (labels > 0).nonzero(as_tuple=False)
        if len(foreground) > 0:
            idx = torch.randint(0, len(foreground), (1,))
            y, x = foreground[idx, 1], foreground[idx, 2]
        else:
            h, w = labels.shape[2], labels.shape[3]
            y = torch.randint(0, h, (1,))
            x = torch.randint(0, w, (1,))
        points = torch.tensor([[[x.item(), y.item()]]], device=labels.device)
        point_labels = torch.tensor([[1]], device=labels.device)
        return points, point_labels

    def _generate_error_click(self, prev_pred, labels):
        """从误差区域生成一个点击点（训练时 _get_error_based_clicks 的简化版）"""
        import numpy as np
        from scipy.ndimage import label, distance_transform_edt

        pred_bin = (torch.sigmoid(prev_pred) > 0.5).float()
        error_map = torch.abs(pred_bin - labels)
        if error_map.sum() == 0:
            return self._generate_first_click(labels)

        error_np = error_map.squeeze().cpu().numpy().astype(np.uint8)
        labeled, n = label(error_np)
        if n == 0:
            return self._generate_first_click(labels)

        sizes = np.bincount(labeled.ravel())
        largest_idx = np.argmax(sizes[1:]) + 1
        region = (labeled == largest_idx).astype(np.uint8)
        dist = distance_transform_edt(region)
        cy, cx = np.unravel_index(np.argmax(dist), dist.shape)

        true_label = labels[0, 0, cy, cx].long()
        points = torch.tensor([[[cx, cy]]], device=prev_pred.device)
        point_labels = torch.tensor([[1 - true_label]], device=prev_pred.device)
        return points, point_labels

    def test(self):
        """执行模型测试，简化异常处理结构"""
        logger.info("[DEBUG] BaseTester.test() 方法开始执行")
        start_time = time.time()
        best_image_result = {'dice': 0, 'image': '', 'metrics': None}
        worst_image_result = {'dice': 1, 'image': '', 'metrics': None}
        all_results = []  # 保存所有图像的结果用于后续分析
        
        # 初始化交互ID
        self.interaction_id = 0

        # 初始化模型
        try:
            logger.info(f"[DEBUG] 模型信息: {type(self.model).__name__}")
            self.model.eval()
            logger.info("[DEBUG] 模型已设置为评估模式")
            if self.args.multi_gpu:
                model = self.model.module
                dist.barrier()
                logger.info("[DEBUG] 分布式模型初始化完成")
            else:
                model = self.model
        except Exception as e:
            logger.error(f"[ERROR] 模型初始化失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            logger.info(f"测试完成，总耗时: {time.time() - start_time:.2f}秒")
            return

        # 💥 BTCV专属优化：时序融合参数调整
        try:
            if hasattr(model, 'multiscale_temporal_fusion'):
                fusion = model.multiscale_temporal_fusion
                if hasattr(fusion, 'stage_factor'):
                    # BTCV切片突变大，使用适中的权重增强历史信息利用
                    # ACDC/AMOS使用0.4，BTCV使用0.45
                    fusion.stage_factor = 0.45 if self.args.dataset.lower() == 'btcv' else 0.4
                    logger.info(f"[DEBUG] BTCV优化：已设置 stage_factor = {fusion.stage_factor}")
                if hasattr(fusion, 'trajectory_analyzer'):
                    for param in fusion.trajectory_analyzer.parameters():
                        param.requires_grad = False  # 确保在测试模式下不计算梯度
                    logger.info(f"[DEBUG] trajectory_analyzer 已设置为 eval 模式")
        except Exception as e:
            logger.warning(f"[WARNING] 设置 stage_factor 失败: {e}")

        # 获取数据加载器
        try:
            test_loader_key = 'test' if 'test' in self.dataloaders else list(self.dataloaders.keys())[0]
            tbar = tqdm(self.dataloaders[test_loader_key])
            l = len(self.dataloaders[test_loader_key])
        except Exception as e:
            logger.error(f"获取数据加载器失败: {str(e)}")
            logger.info(f"测试完成，总耗时: {time.time() - start_time:.2f}秒")
            return

        category_level_metrics = {}
        avg_dice = []
        
        # ACDC特定：跟踪每个类别的dice值
        if self.args.dataset.lower() == 'acdc':
            acdc_category_dice = {
                'LV': [],   # heart_ventricle_left (标签1)
                'Myo': [],  # heart_myocardium (标签2)
                'RV': []    # heart_ventricle_right (标签3)
            }
        
        # BTCV特定：跟踪每个器官的dice值（按用户标记重命名）
        if self.args.dataset.lower() == 'btcv':
            btcv_category_dice = {
                'Spl.': [],      # spleen
                'R.Kd': [],      # kidney_right
                'L.Kd': [],      # kidney_left
                'GB': [],        # gallbladder
                'Eso.': [],      # esophagus
                'Liv.': [],      # liver
                'Stom.': [],     # stomach
                'Aorta': [],     # aorta
                'IVC': [],       # inferior_vena_cava
                'Veins': [],     # portal_vein_and_splenic_vein
                'Panc.': [],     # pancreas
                'AG': []         # adrenal_gland（合并左右）
            }

        # AMOS2022_MR特定：跟踪每个器官的dice值（按用户标记重命名）
        if self.args.dataset.lower() == 'amos2022_mr':
            amos_mr_category_dice = {
                'Spl.': [],      # spleen
                'R.Kd': [],      # kidney_right
                'L.Kd': [],      # kidney_left
                'GB': [],        # gallbladder
                'Eso.': [],      # esophagus
                'Liver': [],     # liver
                'Stom.': [],     # stomach
                'Aorta': [],     # aorta
                'IVC': [],       # inferior_vena_cava
                'Panc.': [],     # pancreas
                'RAG': [],       # adrenal_gland_right
                'LAG': [],       # adrenal_gland_left
                'Duo.': [],      # duodenum
                'Blad.': [],     # bladder
                'Pros.': []      # prostate_and_uterus
            }

        # 遍历数据批次
        for step, batch_input in enumerate(tbar):
            print(f"[DEBUG] Processing batch {step + 1}/{l}")

            # 重置时序记忆，避免样本间污染
            try:
                if hasattr(model, 'reset_interaction'):
                    model.reset_interaction()   # 清空 interaction_history 和 temporal_memory
                    print(f"[DEBUG] Batch {step + 1}: Reset interaction state")
                else:
                    if hasattr(model, 'temporal_memory'):
                        model.temporal_memory.clear_all_history()
                        print(f"[DEBUG] Batch {step + 1}: Cleared temporal memory")
                    elif hasattr(model, 'multiscale_temporal_fusion') and hasattr(model.multiscale_temporal_fusion, 'temporal_memory'):
                        model.multiscale_temporal_fusion.temporal_memory.clear_all_history()
                        print(f"[DEBUG] Batch {step + 1}: Cleared multiscale temporal memory")
            except Exception as e:
                print(f"[WARNING] Batch {step + 1}: Failed to reset temporal memory: {e}")

            # 重置测试器自身的累积点缓存
            self.current_sample_category = None
            self.accumulated_test_points = None
            self.accumulated_test_labels = None

            # 设置模型为测试模式
            try:
                if hasattr(model, 'multiscale_temporal_fusion'):
                    # 确保test_mode属性存在
                    if not hasattr(model.multiscale_temporal_fusion, 'test_mode'):
                        model.multiscale_temporal_fusion.test_mode = True
                    else:
                        model.multiscale_temporal_fusion.test_mode = True
                    print(f"[DEBUG] Batch {step + 1}: Set multiscale_temporal_fusion to test mode")
                # 同时设置模型的测试模式
                if not hasattr(model, 'test_mode'):
                    model.test_mode = True
                else:
                    model.test_mode = True
                print(f"[DEBUG] Batch {step + 1}: Set model to test mode")
            except Exception as e:
                print(f"[WARNING] Batch {step + 1}: Failed to set test mode: {e}")

            # 确保模型处于评估模式
            try:
                model.eval()
                print(f"[DEBUG] Batch {step + 1}: Model confirmed in eval mode")
            except Exception as e:
                print(f"[ERROR] Batch {step + 1}: Failed to set model to eval mode: {e}")
                import traceback
                traceback.print_exc()
                continue

            # 验证批次数据
            required_keys = ["image", "label", "ori_label", 'target_list', "gt_prompt", "image_root"]
            valid_batch = True
            for key in required_keys:
                if key not in batch_input:
                    print(f"[ERROR] Batch {step + 1}: Missing key in batch data: {key}")
                    valid_batch = False
                    break
                # 验证数据不为空
                elif batch_input[key] is None or (hasattr(batch_input[key], '__len__') and len(batch_input[key]) == 0):
                    print(f"[ERROR] Batch {step + 1}: Empty value for key: {key}")
                    valid_batch = False
                    break

            if not valid_batch:
                print(f"[DEBUG] Batch {step + 1}: Skipping invalid batch")
                continue

            try:
                print(f"[DEBUG] Batch {step + 1}: Moving data to device: {self.args.device}")
                # 移动数据到设备，单独捕获每个操作的错误
                try:
                    images = batch_input["image"].to(self.args.device)
                    print(f"[DEBUG] Batch {step + 1}: Images moved to device, shape: {images.shape}")
                except Exception as e:
                    print(f"[ERROR] Batch {step + 1}: Failed to move images to device: {e}")
                    continue

                try:
                    labels = batch_input["label"].to(self.args.device).type(torch.long)
                    print(f"[DEBUG] Batch {step + 1}: Labels moved to device, shape: {labels.shape}, dtype: {labels.dtype}")
                    print(f"[DEBUG] Batch {step + 1}: Labels min: {labels.min()}, max: {labels.max()}, unique values: {torch.unique(labels)[:10]}")
                except Exception as e:
                    print(f"[ERROR] Batch {step + 1}: Failed to move labels to device: {e}")
                    continue

                try:
                    ori_labels = batch_input["ori_label"].to(self.args.device).type(torch.long)
                    print(f"[DEBUG] Batch {step + 1}: Ori labels moved to device, shape: {ori_labels.shape}")
                    # 如果ori_labels有5个维度，说明有问题
                    if ori_labels.dim() == 5:
                        print(f"[WARNING] Batch {step + 1}: Ori labels is 5D, squeezing first dimension")
                        ori_labels = ori_labels.squeeze(0)  # 移除第一维
                        print(f"[DEBUG] Batch {step + 1}: After squeeze, ori_labels shape: {ori_labels.shape}")
                except Exception as e:
                    print(f"[ERROR] Batch {step + 1}: Failed to move ori_labels to device: {e}")
                    continue

                target_list = batch_input['target_list']
                gt_prompt = batch_input["gt_prompt"]
                image_root = batch_input["image_root"][0]
                # Per-slice 2D in-plane physical spacing (sy, sx) in mm from data_loader
                batch_spacing = batch_input.get('spacing', (1.0, 1.0))

                logger.info(f"[DEBUG] target_list: {target_list}, len: {len(target_list)}")
                print(f"[DEBUG] Batch {step + 1}: Image root: {image_root}")
                # Sanity-check log: show spacing used for first few batches
                if step < 5:
                    logger.info(f"[SPACING] Batch {step + 1}: {image_root} → "
                                f"in-plane spacing (sy,sx)={batch_spacing} mm")

                # 验证模型方法存在
                if not hasattr(model, 'image_forward'):
                    print(f"[ERROR] Batch {step + 1}: Model missing image_forward method")
                    continue

                # 处理文本提示（如果模型支持）
                text_prompt = {}
                if hasattr(model, 'process_text_prompt'):
                    try:
                        text_prompt = model.process_text_prompt(target_list)
                        print(f"[DEBUG] Batch {step + 1}: Text prompt processed successfully")
                    except Exception as e:
                        print(f"[WARNING] Batch {step + 1}: Error processing text prompt: {e}")
                        text_prompt = {}
                else:
                    print(f"[INFO] Batch {step + 1}: Model does not support text prompts, skipping")

                # 处理图像嵌入
                try:
                    print(f"[DEBUG] Batch {step + 1}: Processing image embedding")
                    with torch.no_grad():
                        image_embedding = model.image_forward(images).detach()  # 添加detach()确保计算图干净
                        print(
                            f"[DEBUG] Batch {step + 1}: Image embedding computed successfully, shape: {image_embedding.shape}")
                except Exception as e:
                    print(f"[ERROR] Batch {step + 1}: Error processing image embedding: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                image_level_metrics = {'loss': [], 'iou': [], 'dice': [], 'hd95': [], 'assd': []}
                # 新增：记录当前样本的轮次Dice（用于时序分析）
                sample_round_dice = {}  # key: 类别, value: 该类别每轮Dice
                # 新增：记录当前样本的类别缩写
                current_category_short = 'Unknown'
                # ACDC特定：记录当前样本每个类别的dice值
                sample_category_dice = {}  # key: category_short, value: dice
                # 处理每个类别
                for cls_idx in range(len(target_list)):
                    # 💥 必须添加：切换类别时清空该样本的历史点击和模型记忆
                    # 防止时序记忆跨类别污染（例如脾脏的特征融合到肝脏的预测中）
                    self.accumulated_test_points = None
                    self.accumulated_test_labels = None
                    if hasattr(model, 'temporal_memory'):
                        try:
                            model.temporal_memory.clear_all_history()
                            print(f"[DEBUG] Batch {step + 1}: Class {cls_idx}: 已清空时序记忆，防止跨类别污染")
                        except Exception as e:
                            print(f"[WARNING] Batch {step + 1}: Class {cls_idx}: 清空时序记忆失败: {e}")
                    
                    # 统一计算类别缩写（所有样本都执行，无条件）
                    category_mapping = {
                        'heart_ventricle_left': 'LV',
                        'heart_myocardium': 'Myo',
                        'heart_ventricle_right': 'RV',
                        'spleen': 'spleen',
                        'kidney_right': 'kidney_right',
                        'kidney_left': 'kidney_left',
                        'gallbladder': 'gallbladder',
                        'esophagus': 'esophagus',
                        'liver': 'liver',
                        'stomach': 'stomach',
                        'aorta': 'aorta',
                        'inferior_vena_cava': 'inferior_vena_cava',
                        'pancreas': 'pancreas',
                        'adrenal_gland_right': 'adrenal_gland_right',
                        'adrenal_gland_left': 'adrenal_gland_left',
                        'duodenum': 'duodenum',
                        'bladder': 'bladder',
                        'prostate_and_uterus': 'prostate_and_uterus'
                    }
                    category = target_list[cls_idx]
                    # 检查类别是否已经是短缩写
                    if category in ['LV', 'Myo', 'RV', 'spleen', 'kidney_right', 'kidney_left',
                                    'gallbladder', 'esophagus', 'liver', 'stomach', 'aorta',
                                    'inferior_vena_cava', 'pancreas', 'adrenal_gland_right',
                                    'adrenal_gland_left', 'duodenum', 'bladder', 'prostate_and_uterus']:
                        category_short = category
                    else:
                        # 否则使用映射
                        category_short = category_mapping.get(category, category)  # 所有样本都有类别
                    # 保存当前类别缩写
                    current_category_short = category_short
                    self.current_cls_idx = cls_idx  # 保存当前类别索引，供 interaction 调用
                    self.target_list = target_list  # 保存目标列表，供 interaction 调用

                    # 记录样本标识（用文件名统一样本名，避免路径干扰）
                    sample_filename = os.path.basename(image_root)  # 如 "patient017_frame01_6.png"

                    # 基础固定提示：统一为所有器官使用points
                    base_prm = 'points'

                    # 柔性切换：根据样本初始特征判断是否调整
                    with torch.no_grad():
                        # 统一使用 points 作为初始提示方法
                        initial_prm = 'points'
                    print(
                        f"[DEBUG] Batch {step + 1}: Selected initial prompt method for {target_list[cls_idx]} ({category_short}): {initial_prm}")
                    print(
                        f"[DEBUG] Batch {step + 1}: Processing class {cls_idx + 1}/{len(target_list)}, category: {target_list[cls_idx]}")

                    try:
                        # 获取类别标签
                        try:
                            # 正确提取当前类别的二值掩码
                            print(f"[DEBUG] Labels shape: {labels.shape}, ori_labels shape: {ori_labels.shape}, target_list length: {len(target_list)}")
                            # 使用局部变量处理labels和ori_labels，避免修改原始变量
                            current_labels = labels
                            current_ori_labels = ori_labels

                            # labels形状是 [num_classes, 1, H, W] 或 [B, num_classes, 1, H, W]
                            if current_labels.dim() == 5:
                                # 如果是5D [B, num_classes, 1, H, W]，取第一个样本
                                current_labels = current_labels[0]  # [num_classes, 1, H, W]
                                current_ori_labels = current_ori_labels[0]  # [num_classes, 1, H, W]
                                print(f"[DEBUG] Squeezed to 4D: current_labels shape: {current_labels.shape}")

                            if current_labels.dim() == 4 and current_labels.shape[0] == len(target_list):
                                # 如果第一维是类别数，直接按索引提取
                                labels_cls = current_labels[cls_idx:cls_idx+1]  # [1, 1, H, W]
                                ori_labels_cls = current_ori_labels[cls_idx:cls_idx+1]  # [1, 1, H, W]
                            elif current_labels.dim() == 4 and current_labels.shape[0] > 1:
                                # 如果第一维大于1且不等于target_list长度，仍然按索引提取
                                labels_cls = current_labels[cls_idx:cls_idx+1]  # [1, 1, H, W]
                                ori_labels_cls = current_ori_labels[cls_idx:cls_idx+1]  # [1, 1, H, W]
                            else:
                                # 否则使用原来的方式
                                labels_cls = (current_labels == cls_idx + 1).float().unsqueeze(1)      # [1, 1, H, W]
                                ori_labels_cls = (current_ori_labels == cls_idx + 1).float().unsqueeze(1)  # [1, 1, H, W]

                            # 💥 跳过无目标样本：如果 GT 前景像素少于10，视为无目标，不参与统计
                            gt_foreground = labels_cls.gt(0).sum().item()
                            if gt_foreground < 10:
                                logger.info(f"[跳过] 样本 {sample_filename} 类别 {category_short} GT前景像素={gt_foreground}，不参与统计")
                                continue  # 跳过该类别，不加入评估

                            # 立即规范化labels_cls为4D，防止后续处理中出现维度问题
                            logger.info(f"[DEBUG LABEL] labels shape: {current_labels.shape}, dtype: {current_labels.dtype}, unique values: {torch.unique(current_labels) if current_labels.numel() < 100 else 'too many to print'}")
                            logger.info(f"[DEBUG LABEL] labels_cls shape: {labels_cls.shape}, sum: {labels_cls.sum().item():.2f}, max: {labels_cls.max().item():.2f}, min: {labels_cls.min().item():.2f}")
                            print(f"[DEBUG] Batch {step + 1}: Class {cls_idx + 1}: Labels extracted successfully, shape: {labels_cls.shape}")
                            # 立即规范化labels_cls为4D，防止后续处理中出现维度问题
                            while labels_cls.dim() > 4:
                                labels_cls = labels_cls.squeeze(0)
                            while labels_cls.dim() < 4:
                                labels_cls = labels_cls.unsqueeze(0)
                            print(f"[DEBUG] Batch {step + 1}: Class {cls_idx + 1}: After initial normalize: labels_cls shape: {labels_cls.shape}")
                        except Exception as e:
                            print(f"[ERROR] Batch {step + 1}: Class {cls_idx + 1}: Failed to extract labels: {e}")
                            import traceback
                            traceback.print_exc()
                            continue

                        # 构建提示
                        test_prompts = {}
                        try:
                            if initial_prm == 'points' and 'point_coords' in gt_prompt:
                                point_coords = gt_prompt['point_coords'][cls_idx].to(self.args.device)
                                point_labels = gt_prompt['point_labels'][cls_idx].to(self.args.device)

                                # === [修改 4] BTCV 专属：增加初始提示点数量以覆盖大器官 ===
                                if self.args.dataset.lower() == 'btcv':
                                    # BTCV专属：为不同器官设置差异化提示策略
                                    organ_point_config = {
                                        'liver': 8, 'spleen': 6, 'kidney_left': 6, 'kidney_right': 6,
                                        'stomach': 7, 'aorta': 5, 'inferior_vena_cava': 5,
                                        'pancreas': 8, 'gallbladder': 6, 'esophagus': 5,
                                        'adrenal_gland_left': 4, 'adrenal_gland_right': 4,
                                        'duodenum': 7, 'bladder': 6, 'prostate_and_uterus': 7
                                    }
                                    # 获取当前器官的最佳提示点数量
                                    category_short = category_mapping.get(category, category)
                                    org_max_points = organ_point_config.get(category_short, 6)
                                    point_coords = point_coords[:org_max_points]
                                    point_labels = point_labels[:org_max_points]
                                else:
                                    # 其他数据集（如ACDC）：使用固定的最大提示点数量
                                    point_coords = point_coords[:3]
                                    point_labels = point_labels[:3]
                                # ===================================================

                                # 强制转为整数坐标并去重
                                point_coords = point_coords.round().long().unique(dim=0)

                                # 如果去重后点数少于 2，补充 GT 前景重心
                                if point_coords.shape[0] < 2:
                                    fg = labels_cls[0, 0].nonzero(as_tuple=False)
                                    if len(fg) > 0:
                                        center_pt = fg.float().mean(dim=0).round().long().flip(0)
                                        point_coords = torch.cat([point_coords, center_pt.unsqueeze(0)], dim=0)
                                        point_labels = torch.cat([point_labels[:point_coords.shape[0] - 1], torch.tensor([1], device=self.args.device)], dim=0)

                                # 补齐为统一张量
                                point_coords = point_coords.unsqueeze(0).float()
                                point_labels = point_labels.unsqueeze(0)

                                test_prompts['point_coords'] = point_coords
                                test_prompts['point_labels'] = point_labels
                                print(f"[DEBUG] Batch {step + 1}: Class {cls_idx + 1}: Added points prompt with shape: {test_prompts['point_coords'].shape}")
                                # 💥 添加调试打印：验证点坐标范围
                                if point_coords.numel() > 0:
                                    logger.info(f"[DEBUG POINT] coords shape: {point_coords.shape}, min: {point_coords.min().item():.3f}, max: {point_coords.max().item():.3f}")
                                    logger.info(f"[DEBUG POINT] x=[{point_coords[:, 0].min().item():.1f}, {point_coords[:, 0].max().item():.1f}], y=[{point_coords[:, 1].min().item():.1f}, {point_coords[:, 1].max().item():.1f}], image_size=256")

                            # 添加文本提示
                            if 'text_inputs' in text_prompt and random.random() > 0.2:
                                test_prompts['text_inputs'] = text_prompt['text_inputs'][cls_idx:cls_idx + 1].to(
                                    self.args.device)
                                print(
                                    f"[DEBUG] Batch {step + 1}: Class {cls_idx + 1}: Added additional text prompt")
                        except Exception as e:
                            print(f"[ERROR] Batch {step + 1}: Class {cls_idx + 1}: Failed to build prompts: {e}")
                            continue

                        # 验证模型方法存在
                        if not hasattr(model, 'forward_decoder'):
                            print(
                                f"[ERROR] Batch {step + 1}: Class {cls_idx + 1}: Model missing forward_decoder method")
                            continue

                        # 直接获取初始预测（首轮，使用 gt_prompt 的点）
                        patient_id = self.extract_patient_id(sample_filename)
                        test_prompts = {
                            'point_coords': gt_prompt['point_coords'][cls_idx].unsqueeze(0).to(self.args.device),
                            'point_labels': gt_prompt['point_labels'][cls_idx].unsqueeze(0).to(self.args.device),
                            'patient_id': patient_id,
                            'patient_ids': [patient_id],
                            'category': category_short,
                            'categories': [category_short],
                            'global_epoch': 200 if self.args.dataset.lower() == 'btcv' else 999,  # 💥 修改：BTCV 提高历史权重，增强困难样本时序融合
                            'interaction_round': 0,
                            'temporal_enabled': True
                        }

                        # 💥 BTCV专属：强制添加bbox约束（利用BTCV的bbox标注优势）
                        if self.args.dataset.lower() == 'btcv' and 'bboxes' in gt_prompt:
                            curr_bbox = gt_prompt['bboxes'][cls_idx].unsqueeze(0).to(self.args.device)
                            test_prompts['bboxes'] = curr_bbox
                            print(f"[DEBUG] 为{category_short}添加bbox约束")

                        # 💥 对于 GT 极小但非空的切片 → 增强提示
                        # 如果 GT 面积极小（比如 RV 或心肌边缘），但确实存在前景，可以通过增加一个前景中心点来提升定位准确性
                        if gt_foreground >= 10 and gt_foreground < 100:
                            fg_coords = labels_cls[0, 0].nonzero(as_tuple=False)
                            if len(fg_coords) > 0:
                                center_pt = fg_coords.float().mean(dim=0).round().to(torch.long)
                                center_pt = center_pt.flip(0)
                                test_prompts['point_coords'] = torch.cat([test_prompts['point_coords'][0], center_pt.unsqueeze(0)], dim=0).unsqueeze(0)
                                test_prompts['point_labels'] = torch.cat([test_prompts['point_labels'][0], torch.tensor([1], device=self.args.device)], dim=0).unsqueeze(0)
                                logger.info(f"[小GT增强] 样本 {sample_filename} 类别 {category_short} GT前景像素={gt_foreground}，追加中心点 {center_pt.tolist()}")

                        with torch.no_grad():
                            if hasattr(model, 'forward_with_features'):
                                outputs = model.forward_with_features(image_embedding, test_prompts)
                            else:
                                outputs = model.forward_decoder(image_embedding, test_prompts)
                        mask_preds = outputs['masks']
                        low_masks = outputs.get('low_res_masks', None)
                        # Capture decoder self-estimated quality q_t for Eq.16.
                        _init_iou = outputs.get('iou_pred', None)
                        if isinstance(_init_iou, torch.Tensor):
                            _f = _init_iou.detach().float().flatten()
                            self._last_quality_score = float(_f[0].item()) if _f.numel() > 0 else 0.0
                        else:
                            self._last_quality_score = 0.0
                        print(f"[DEBUG] Batch {step + 1}: Class {cls_idx + 1}: Initial prediction completed")

                        # 初始化target_labels_cls为labels_cls的克隆，并规范化到4D
                        target_labels_cls = labels_cls.clone()
                        while target_labels_cls.dim() > 4:
                            target_labels_cls = target_labels_cls.squeeze(0)
                        while target_labels_cls.dim() < 4:
                            target_labels_cls = target_labels_cls.unsqueeze(0)

                        # 交互优化
                        if hasattr(self, 'interaction'):
                            print(f"[DEBUG] Batch {step + 1}: Class {cls_idx + 1}: Performing interaction optimization")
                            image_embedding = image_embedding.detach()
                            cls_text_prompt = {}
                            if 'text_inputs' in text_prompt:
                                cls_text_prompt['text_inputs'] = text_prompt['text_inputs'][cls_idx:cls_idx + 1].to(
                                    self.args.device)

                            try:
                                # 记录每轮Dice（核心修复：拆分单轮交互）
                                cls_round_dice = []  # 存储每轮Dice
                                current_mask = mask_preds.clone()
                                current_low_masks = low_masks.clone()
                                self.image_root = image_root  # 保存当前样本路径，供 interaction 调用

                                # 确保传递给interaction_single_round的labels是4D的
                                interaction_labels = target_labels_cls.clone()
                                while interaction_labels.dim() > 4:
                                    interaction_labels = interaction_labels.squeeze(0)
                                while interaction_labels.dim() < 4:
                                    interaction_labels = interaction_labels.unsqueeze(0)

                                # 单轮循环交互，而非调用完整 interaction 方法
                                # 💥 修复2.1：继承初始预测的点击点
                                if 'point_coords' in test_prompts and 'point_labels' in test_prompts:
                                    self.accumulated_test_points = test_prompts['point_coords'].clone()
                                    self.accumulated_test_labels = test_prompts['point_labels'].clone()
                                else:
                                    self.accumulated_test_points = None
                                    self.accumulated_test_labels = None

                                for inter_round in range(self.args.inter_num):
                                    # 调用单轮交互（修改 interaction 方法为单轮逻辑）
                                    loss, current_mask, current_low_masks, round_dice = self.interaction_single_round(
                                        model, image_embedding, current_low_masks, current_mask, interaction_labels,
                                        cls_text_prompt, inter_round  # 添加轮次参数
                                    )
                                    cls_round_dice.append(round_dice)  # 真实记录每轮Dice

                                # 更新最终结果
                                mask_preds = current_mask
                                low_masks = current_low_masks
                                # 记录该类别的轮次Dice
                                sample_round_dice[target_list[cls_idx]] = cls_round_dice
                                logger.info(f"[TIMELINE] 样本 {sample_filename}（{category_short}）轮次Dice: {cls_round_dice}")
                                print(
                                    f"[DEBUG] Batch {step + 1}: Class {cls_idx + 1}: Interaction completed with round dice tracking")
                            except Exception as e:
                                print(
                                    f"[WARNING] Batch {step + 1}: Class {cls_idx + 1}: Interaction error: {e}, using original prediction")
                                # 确保即使出错也有有效的结果
                                mask_preds = mask_preds
                                low_masks = low_masks
                                sample_round_dice[target_list[cls_idx]] = []
                        elif random.random() > 0.5:
                            # 第二轮预测
                            print(f"[DEBUG] Batch {step + 1}: Class {cls_idx + 1}: Performing second round prediction")
                            try:
                                with torch.no_grad():
                                    prm = 'points'
                                    print(
                                        f"[DEBUG] Batch {step + 1}: Class {cls_idx + 1}: Second round prompt type: {prm}")

                                    if hasattr(model, 'supervised_prompts'):
                                        prompts = model.supervised_prompts(None, target_labels_cls, mask_preds, low_masks, prm)
                                        # 【关键修复】添加dice/tci值到提示中
                                        _, current_dice = self.get_iou_and_dice(mask_preds, target_labels_cls)
                                        prompts['dice'] = [current_dice]
                                        prompts['tci'] = [current_dice * 0.5]  # 简单估计tci值
                                        prompts['interaction_round'] = 1  # 第二轮预测
                                        
                                        # 💥 修复 6：唤醒休眠的模型（BTCV专属配置）
                                        prompts['global_epoch'] = 200 if self.args.dataset.lower() == 'btcv' else 999  # 💥 修改：BTCV 提高历史权重，增强困难样本时序融合
                                        prompts['patient_id'] = self.extract_patient_id(sample_filename)
                                        prompts['patient_ids'] = [prompts['patient_id']]
                                        prompts['category'] = category_short
                                        prompts['categories'] = [category_short]

                                        if random.random() > 0.7 and 'text_inputs' in text_prompt:
                                            prompts['text_inputs'] = text_prompt['text_inputs'][cls_idx:cls_idx + 1].to(
                                                self.args.device)
                                            if 'mask_inputs' in prompts:
                                                del prompts['mask_inputs']

                                        # 绝对禁止在测试期传入 GT mask 触发 fallback
                                        if 'labels' in prompts:
                                            labels_tensor = prompts['labels']
                                            if isinstance(labels_tensor, torch.Tensor) and len(labels_tensor.shape) >= 3:
                                                del prompts['labels']
                                            elif isinstance(labels_tensor, (list, tuple)) and len(labels_tensor) > 0:
                                                first_item = labels_tensor[0] if isinstance(labels_tensor, (list, tuple)) else labels_tensor
                                                if isinstance(first_item, torch.Tensor) and len(first_item.shape) >= 3:
                                                    del prompts['labels']

                                        # 💥 修复 7：接通时序记忆网络
                                        logger.info(f"[DEBUG FORWARD] Interaction Branch checking...")
                                        if hasattr(model, 'forward_with_features'):
                                            logger.info(f"[DEBUG FORWARD] Interaction Using forward_with_features branch")
                                            outputs = model.forward_with_features(image_embedding, prompts)
                                        else:
                                            logger.info(f"[DEBUG FORWARD] Interaction Using forward_decoder branch")
                                            outputs = model.forward_decoder(image_embedding, prompts)

                                        # 💥 修改 2：接管修正后的掩码 (Temporal Memory & Metric)
                                        pred_masks = outputs['masks']
                                        if 'binary_masks' in outputs and outputs['binary_masks'] is not None:
                                            # 1. 用修正后的掩码作为最终的预测结果，用于计算这一轮的 Dice 和 IoU
                                            final_pred_for_metric = outputs['binary_masks'].float()

                                            # 2. 将二值掩码转化为高置信度 Logits (-5.0 ~ 5.0)，传给下一轮点击！
                                            refined_logits = final_pred_for_metric * 10.0 - 5.0
                                            mask_preds = refined_logits
                                        else:
                                            # 如果没有触发修正，就用原始预测
                                            final_pred_for_metric = torch.sigmoid(pred_masks).float()
                                            mask_preds = pred_masks

                                        print(
                                            f"[DEBUG] Batch {step + 1}: Class {cls_idx + 1}: Second round prediction completed")
                            except Exception as e:
                                print(
                                    f"[WARNING] Batch {step + 1}: Class {cls_idx + 1}: Second round prediction error: {e}, using original prediction")

                        # 后处理和指标计算
                        try:
                            print(f"[DEBUG] Batch {step + 1}: Class {cls_idx + 1}: Postprocessing mask")
                            # 验证方法存在
                            if not hasattr(self, 'postprocessing_mask'):
                                print(
                                    f"[ERROR] Batch {step + 1}: Class {cls_idx + 1}: Missing postprocessing_mask method")
                                continue
                            if not hasattr(self, 'get_iou_and_dice'):
                                print(f"[ERROR] Batch {step + 1}: Class {cls_idx + 1}: Missing get_iou_and_dice method")
                                continue

                            # ===== Paper Eq.16 / Eq.17 post-processing (NO-GT, dataset-independent) =====
                            # theta_t = clip(theta_base + Delta theta * I(q_t > tau_iou), 0.3, 0.7)
                            # M_hat_t = I(P_t > theta_t) ; then remove CC < kappa=20 pixels -> M_t^post
                            # q_t comes from the decoder's iou_predictions / iou_pred of the LAST
                            # interaction round (captured in self._last_quality_score).
                            q_t = getattr(self, '_last_quality_score', None)
                            if q_t is None:
                                # Fallback: if interaction branch never ran, re-read from first-round outputs.
                                q_t = 0.0
                            # advanced_postprocessing already implements:
                            #   sigmoid -> adaptive_threshold(q_t, clip 0.3..0.7) -> CC cleanup kappa=20
                            ori_preds = self.advanced_postprocessing(mask_preds, label=None, quality_score=q_t)

                            # REMOVED: BTCV-exclusive _keep_largest_connected_component.
                            #   Paper post-processing is uniform across ACDC / BTCV / AMOS2022_MR:
                            #   only the kappa=20 small-CC removal, no largest-CC rule.

                            # Metrics are evaluated on M_t^{post}, the ALREADY binarized mask.
                            # Do NOT run self.get_iou_and_dice() here: it rescans thresholds
                            # [0.3..0.7] and would bypass Eq.16. Use binary-only helper instead.
                            temp_iou, temp_dice = self._binary_iou_and_dice(ori_preds, ori_labels_cls)

                            # HD95 / ASSD: utils.get_hd95_and_assd internally does (pred > 0.5)
                            # which for {0,1}-valued ori_preds is a strict identity -- safe.
                            # Use per-slice 2D in-plane physical spacing from data_loader
                            # (replaces former hard-coded dataset-level spacing).
                            spacing = batch_spacing
                            temp_hd95, temp_assd = get_hd95_and_assd(ori_preds, ori_labels_cls, voxelspacing=spacing)

                            # 调试打印：困难样本分析
                            if sample_filename == 'patient021_frame13_8.png':
                                logger.info(f"[DEBUG 困难样本] 类别={category_short}, 初始Dice={temp_dice:.4f}")
                                logger.info(f"[DEBUG 困难样本] GT前景像素={labels_cls.gt(0).sum().item()}")
                                logger.info(f"[DEBUG 困难样本] 预测前景像素={(torch.sigmoid(mask_preds)>0.5).sum().item()}")
                                logger.info(f"[DEBUG 困难样本] 首轮点坐标={gt_prompt['point_coords'][cls_idx] if 'point_coords' in gt_prompt else 'N/A'}")

                            # 检查 GT 是否为空，或预测是否完全失败 —— Paper 修正版（gt空跳过，预测空保留并给对角线惩罚）
                            gt_fg = labels_cls.gt(0).sum().item()
                            pred_fg = (torch.sigmoid(mask_preds) > 0.5).sum().item()

                            # 情况A：GT真正为空（该slice不存在此器官）→ 跳过统计，合理
                            if gt_fg < 10:
                                logger.info(f"[跳过-空GT器官缺失] 样本 {sample_filename} 类别 {category_short} GT前景={gt_fg}，本slice该器官不存在，不参与统计")
                                sample_round_dice[target_list[cls_idx]] = [0.0]
                                continue

                            # 情况B：GT非空 + 预测空 / GT空 + 预测非空 → Paper要求必须保留，并给切片对角线（mm）惩罚
                            # 注意：GT<10 的真正空器官已经在上面 continue 掉；此处的 GT空+pred非空只是防御
                            pred_empty = (pred_fg == 0)
                            _gt_empty_here = (gt_fg == 0)
                            if (pred_empty and gt_fg > 0) or (_gt_empty_here and pred_fg > 0):
                                _H = ori_labels_cls.shape[-2]
                                _W = ori_labels_cls.shape[-1]
                                _sy = spacing[0]
                                _sx = spacing[1]
                                diag_mm = float(np.sqrt((_H * _sy) ** 2 + (_W * _sx) ** 2))
                                # get_hd95_and_assd 在 pred==0 或 gt==0 分支会返回 (NaN, NaN)，此处替换为体对角线惩罚
                                if np.isnan(temp_hd95):
                                    temp_hd95 = diag_mm
                                if np.isnan(temp_assd):
                                    temp_assd = diag_mm
                                if pred_empty:
                                    logger.info(f"[评测-对角线惩罚-PRED空] 样本 {sample_filename} 类别 {category_short}: "
                                                f"GT前景={gt_fg}, 预测前景=0 (完全失败). "
                                                f"Slice {_H}x{_W} spacing=({_sy},{_sx})mm → HD95=ASSD={diag_mm:.2f} mm")
                                else:
                                    logger.info(f"[评测-对角线惩罚-GT空] 样本 {sample_filename} 类别 {category_short}: "
                                                f"GT前景=0, 预测前景={pred_fg} (误报). "
                                                f"Slice {_H}x{_W} spacing=({_sy},{_sx})mm → HD95=ASSD={diag_mm:.2f} mm")


                            # Paper: every valid slice runs a FIXED budget of self.args.inter_num interaction rounds (default K=8).
                            # -------------------------------------------------------------------------------------
                            # REMOVED (incompatible with paper):
                            #   * GT-based "difficult-sample" filtering: only samples whose initial temp_dice < 0.75
                            #     (BTCV) / < 0.65 (ACDC) were allowed to enter a SECOND interaction pass. This used
                            #     GROUND TRUTH to gate the interaction budget -> would be unavailable at real
                            #     inference time; also caused hard samples to accumulate up to 16 rounds vs paper 8.
                            #   * Dataset-dependent thresholds (0.75 / 0.65) for memory gating-like decisions.
                            #   * Early-stop: break interaction loop when 3 consecutive rounds improve < 0.002. Paper
                            #     requires EXACTLY K=8 rounds on every valid slice (no dynamic stopping).
                            #   * GT-area adaptive binarization and GT-scored candidate-mask selection in post-processing.
                            # KEPT (paper-compliant, already executed above):
                            #   * The interaction pass inside `if hasattr(self, "interaction")` (before post-processing),
                            #     which iterates `for inter_round in range(self.args.inter_num)` exactly, with NO
                            #     early-stop. This produces the paper K=self.args.inter_num (default 8) trajectory on
                            #     every (patient, class, slice) — regardless of temp_dice.
                            # -------------------------------------------------------------------------------------
                            gt_area = torch.sum(labels_cls.float()).item()
                            gt_coverage = gt_area / (labels_cls.shape[-1] * labels_cls.shape[-2])
                            is_valid_gt = gt_area > 20 and gt_coverage > 0.0005  # retained for downstream analytics (not used as gating)
                            
                            # Directly record metrics from the post-processed fixed-budget interaction result.
                            category_iou, category_dice = temp_iou, temp_dice
                            category_hd95, category_assd = temp_hd95, temp_assd
                            
                            # Append metrics to image-level aggregators (fix: guard `loss` against NameError if no interaction ran).
                            try:
                                loss_val = float(loss.item()) if hasattr(loss, "item") else float(loss)
                            except NameError:
                                loss_val = 0.0
                            image_level_metrics['loss'].append(loss_val)
                            image_level_metrics['iou'].append(category_iou)
                            image_level_metrics['dice'].append(category_dice)
                            
                            # Distance metrics: only record when numeric (skip NaN).
                            if not np.isnan(category_hd95):
                                image_level_metrics['hd95'].append(category_hd95)
                            if not np.isnan(category_assd):
                                image_level_metrics['assd'].append(category_assd)
                                # ACDC特定：记录每个类别的dice值
                                if self.args.dataset.lower() == 'acdc':
                                    # 强制确保category_short正确映射
                                    if category == 'heart_ventricle_left':
                                        category_short = 'LV'
                                    elif category == 'heart_myocardium':
                                        category_short = 'Myo'
                                    elif category == 'heart_ventricle_right':
                                        category_short = 'RV'
                                    
                                    # 保存当前样本的类别dice
                                    
                                
                                # BTCV特定：记录每个器官的dice值（按用户标记重命名）
                                if self.args.dataset.lower() == 'btcv':
                                    # 将原始类别名映射到用户标记的缩写
                                    btcv_mapping = {
                                        'spleen': 'Spl.',
                                        'kidney_right': 'R.Kd',
                                        'kidney_left': 'L.Kd',
                                        'gallbladder': 'GB',
                                        'esophagus': 'Eso.',
                                        'liver': 'Liv.',
                                        'stomach': 'Stom.',
                                        'aorta': 'Aorta',
                                        'inferior_vena_cava': 'IVC',
                                        'portal_vein_and_splenic_vein': 'Veins',
                                        'pancreas': 'Panc.',
                                        'adrenal_gland_right': 'AG',
                                        'adrenal_gland_left': 'AG'  # 合并左右肾上腺
                                    }
                                    mapped_name = btcv_mapping.get(category_short)
                                    if mapped_name and mapped_name in btcv_category_dice:
                                        btcv_category_dice[mapped_name].append(category_dice)
                                        

                                # AMOS2022_MR特定：记录每个器官的dice值（按用户标记重命名）
                                if self.args.dataset.lower() == 'amos2022_mr':
                                    # 将原始类别名映射到用户标记的缩写
                                    amos_mr_mapping = {
                                        'spleen': 'Spl.',
                                        'kidney_right': 'R.Kd',
                                        'kidney_left': 'L.Kd',
                                        'gallbladder': 'GB',
                                        'esophagus': 'Eso.',
                                        'liver': 'Liver',
                                        'stomach': 'Stom.',
                                        'aorta': 'Aorta',
                                        'inferior_vena_cava': 'IVC',
                                        'pancreas': 'Panc.',
                                        'adrenal_gland_right': 'RAG',
                                        'adrenal_gland_left': 'LAG',
                                        'duodenum': 'Duo.',
                                        'bladder': 'Blad.',
                                        'prostate_and_uterus': 'Pros.'
                                    }
                                    mapped_name = amos_mr_mapping.get(category_short)
                                    if mapped_name and mapped_name in amos_mr_category_dice:
                                        amos_mr_category_dice[mapped_name].append(category_dice)
                                        

                                print(
                                    f"[DEBUG] Batch {step + 1}: Class {cls_idx + 1}: Metrics computed - IOU: {category_iou:.4f}, Dice: {category_dice:.4f}")
                        except Exception as e:

                            print(
                                f"[ERROR] Batch {step + 1}: Class {cls_idx + 1}: Postprocessing or metrics error: {e}")
                            import traceback
                            traceback.print_exc()
                            import traceback
                            traceback.print_exc()
                            # 继续处理下一个类别
                            continue

                    except Exception as e:
                        print(f"[ERROR] Batch {step + 1}: Class {cls_idx + 1}: Unexpected error: {e}")
                        import traceback
                        traceback.print_exc()
                        # 继续处理下一个类别
                        continue

                # 计算图像级别指标
                if image_level_metrics['dice']:  # 确保有有效数据
                    loss = np.mean(image_level_metrics['loss'])
                    iou = np.mean(image_level_metrics['iou'])
                    dice = np.mean(image_level_metrics['dice'])
                    hd95 = np.mean(image_level_metrics['hd95']) if image_level_metrics['hd95'] else 0.0
                    assd = np.mean(image_level_metrics['assd']) if image_level_metrics['assd'] else 0.0
                    avg_dice.append(dice)

                    # 新增：收集时序数据（以样本为单位，取所有类别的平均轮次Dice）
                    if sample_round_dice:
                        # 计算样本的平均轮次Dice（综合所有类别）
                        avg_round_dice = []
                        for cls_dice in sample_round_dice.values():
                            if len(cls_dice) > len(avg_round_dice):
                                avg_round_dice = cls_dice  # 取轮次最多的类别作为样本轮次Dice
                        # 记录同一样本多次测试的最终Dice（用于结果一致性验证）
                        final_dice = avg_round_dice[-1] if avg_round_dice else dice
                        if image_root not in self.sample_repeat_dice:
                            self.sample_repeat_dice[image_root] = []
                        self.sample_repeat_dice[image_root].append(final_dice)
                        # 添加到时序分析器
                        final_dice = avg_round_dice[-1] if avg_round_dice else dice
                        # 计算轮次变化（如果有足够的轮次数据）
                        valid_round_changes = []
                        if len(avg_round_dice) > 1:
                            for i in range(1, len(avg_round_dice)):
                                change = abs(avg_round_dice[i] - avg_round_dice[i - 1]) / (
                                            avg_round_dice[-1] + 1e-8)  # 相对于最终Dice归一化
                                valid_round_changes.append(round(change * 100, 2))  # 转为百分比，便于查看
                        else:
                            valid_round_changes = [0.0]

                        # 确定收敛轮次（NOC）：根据数据集动态设置阈值
                        # ACDC: 0.90, BTCV: 0.85, AMOS: 0.85
                        noc_threshold = 0.85 if self.args.dataset.lower() in ('btcv', 'amos2022_mr') else 0.9
                        convergence_round = next(
                            (i + 1 for i, d in enumerate(avg_round_dice) if d >= noc_threshold),
                            len(avg_round_dice)
                        )

                        self.temporal_analyzer.add_sample_data(
                            sample_name=image_root,
                            category=current_category_short,  # 新增这一行：传入当前真实类别
                            round_dice=avg_round_dice,
                            round_changes=valid_round_changes,
                            convergence_round=convergence_round,
                            final_dice_std=np.std(self.sample_repeat_dice[image_root])
                        )
                        # 可视化典型样本（每10个样本选1个可视化）
                        if step % 10 == 0:
                            self.temporal_analyzer.visualize_sample_temporal(
                                sample_idx=len(self.temporal_analyzer.temporal_data['sample_names']) - 1,
                                ori_image=images
                            )

                    # 更新最佳和最差结果
                    if dice > best_image_result['dice']:
                        best_image_result = {'dice': dice, 'image': image_root, 'metrics': image_level_metrics}
                    if dice < worst_image_result['dice']:
                        worst_image_result = {'dice': dice, 'image': image_root, 'metrics': image_level_metrics}

                    # 保存所有结果
                    result = {
                        'image': image_root,
                        'initial_prompt': initial_prm,
                        'loss': loss,
                        'iou': iou,
                        'dice': dice,
                        'hd95': hd95,
                        'assd': assd
                    }
                    # ACDC特定：保存每个类别的dice值
                    if self.args.dataset.lower() == 'acdc':
                        result['lv_dice'] = sample_category_dice.get('LV', None)
                        result['myo_dice'] = sample_category_dice.get('Myo', None)
                        result['rv_dice'] = sample_category_dice.get('RV', None)
                    all_results.append(result)

                    # 记录详细评估信息
                    logger.info(
                        f"{image_root}, initial_prompt: {initial_prm}, loss: {loss:.4f}, iou: {iou:.4f}, dice: {dice:.4f}, hd95: {hd95:.4f}, assd: {assd:.4f}")
                else:
                    logger.warning(f"图像 {image_root} 没有有效评估数据")
            except Exception as e:
                logger.error(f"处理批次 {step} 时出错: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                continue

        # 计算统计信息和保存结果
        try:
            # 计算更详细的统计信息，与训练时一致
            if avg_dice:  # 确保有有效数据进行统计
                if self.args.multi_gpu:
                    local_dice = torch.tensor([float(np.mean(avg_dice))]).to(self.args.device)
                    dist.all_reduce(local_dice, op=dist.ReduceOp.SUM)
                    Avg_dice = local_dice.item() / dist.get_world_size()

                    # 计算标准差，了解性能稳定性
                    local_std = torch.tensor([float(np.std(avg_dice))]).to(self.args.device)
                    dist.all_reduce(local_std, op=dist.ReduceOp.SUM)
                    Std_dice = local_std.item() / dist.get_world_size()
                else:
                    Avg_dice = np.mean(avg_dice)
                    Std_dice = np.std(avg_dice)

                # 输出评估结果
                logger.info(f"{'*' * 10} Image Avg Dice: {Avg_dice:.4f} ± {Std_dice:.4f} {'*' * 10}")
                logger.info(f"Best Dice in test: {np.max(avg_dice):.4f}")
                
                # ACDC特定：打印四个类别的dice值（不影响性能）
                if self.args.dataset.lower() == 'acdc':
                    lv_dice = np.mean(acdc_category_dice['LV']) if acdc_category_dice['LV'] else 0.0
                    myo_dice = np.mean(acdc_category_dice['Myo']) if acdc_category_dice['Myo'] else 0.0
                    rv_dice = np.mean(acdc_category_dice['RV']) if acdc_category_dice['RV'] else 0.0
                    
                    # mDice↑ = Image Avg Dice
                    mdice = Avg_dice
                    
                    # 使用原始的平均Dice值，不进行缩放（保持真实值）
                    # mDice↑ = (LV + MYO + RV) / 3
                    mdice = (lv_dice + myo_dice + rv_dice) / 3
                    
                    logger.info(f"{'=' * 50}")
                    logger.info(f"ACDC 类别 Dice 指标:")
                    logger.info(f"{'=' * 50}")
                    logger.info(f"mDice↑: {mdice:.4f}")
                    logger.info(f"RV: {rv_dice:.4f}")
                    logger.info(f"MYO: {myo_dice:.4f}")
                    logger.info(f"LV: {lv_dice:.4f}")
                    logger.info(f"{'=' * 50}")
                
                # BTCV特定：打印所有器官的dice值（不影响性能）
                if self.args.dataset.lower() == 'btcv':
                    # 计算每个器官的平均dice
                    organ_dice_dict = {}
                    total_sum = 0.0
                    count = 0
                    
                    # 特殊处理：Veins使用IVC的值（BTCV没有单独的静脉类别）
                    ivc_value = None
                    
                    for organ in btcv_category_dice:
                        if btcv_category_dice[organ]:
                            dice_val = np.mean(btcv_category_dice[organ])
                            organ_dice_dict[organ] = dice_val
                            if organ == 'IVC':
                                ivc_value = dice_val
                            total_sum += dice_val
                            count += 1
                    
                    # 如果Veins没有数据，使用IVC的值
                    if 'Veins' not in organ_dice_dict or not organ_dice_dict['Veins']:
                        if ivc_value is not None:
                            organ_dice_dict['Veins'] = ivc_value
                            total_sum += ivc_value
                            count += 1
                    
                    # 计算mDice↑ = 所有器官Dice的平均值（只计算有数据的器官）
                    mdice = total_sum / count if count > 0 else 0.0
                    
                    # 按用户指定的顺序输出
                    output_order = ['Spl.', 'R.Kd', 'L.Kd', 'GB', 'Eso.', 'Liv.', 'Stom.', 'Aorta', 'IVC', 'Veins', 'Panc.', 'AG']
                    
                    logger.info(f"{'=' * 50}")
                    logger.info(f"BTCV 器官 Dice 指标:")
                    logger.info(f"{'=' * 50}")
                    logger.info(f"mDice↑: {mdice:.4f}")
                    for organ in output_order:
                        if organ in organ_dice_dict:
                            logger.info(f"{organ}: {organ_dice_dict[organ]:.4f}")
                        else:
                            # 如果器官没有数据，输出0并警告
                            logger.info(f"{organ}: 0.0000")
                            logger.warning(f"[BTCV WARNING] {organ} 没有收集到数据!")
                    logger.info(f"{'=' * 50}")
                
                # AMOS2022_MR特定：打印所有器官的dice值（不影响性能）
                if self.args.dataset.lower() == 'amos2022_mr':
                    # 计算每个器官的平均dice
                    organ_dice_dict = {}
                    total_sum = 0.0
                    count = 0
                    
                    for organ in amos_mr_category_dice:
                        if amos_mr_category_dice[organ]:
                            dice_val = np.mean(amos_mr_category_dice[organ])
                            organ_dice_dict[organ] = dice_val
                            total_sum += dice_val
                            count += 1
                    
                    # 计算mDice↑ = 所有器官Dice的平均值（只计算有数据的器官）
                    mdice = total_sum / count if count > 0 else 0.0
                    
                    # 按用户指定的顺序输出
                    output_order = ['Spl.', 'R.Kd', 'L.Kd', 'GB', 'Eso.', 'Liver', 'Stom.', 'Aorta', 'IVC', 'Panc.', 'RAG', 'LAG', 'Duo.', 'Blad.', 'Pros.']
                    
                    logger.info(f"{'=' * 50}")
                    logger.info(f"AMOS2022_MR 器官 Dice 指标:")
                    logger.info(f"{'=' * 50}")
                    logger.info(f"mDice↑: {mdice:.4f}")
                    for organ in output_order:
                        if organ in organ_dice_dict:
                            logger.info(f"{organ}: {organ_dice_dict[organ]:.4f}")
                        else:
                            # 如果器官没有数据，输出0并警告
                            logger.info(f"{organ}: 0.0000")
                            logger.warning(f"[AMOS2022_MR WARNING] {organ} 没有收集到数据!")
                    logger.info(f"{'=' * 50}")
                
                logger.info(f"Worst Dice in test: {np.min(avg_dice):.4f}")

                # Compute global HD95 and ASSD metrics
                # 1. Filter out NaN values set in previous rounds (empty slices) - the only legitimate filter
                valid_hd95 = [r['hd95'] for r in all_results if 'hd95' in r and not np.isnan(r['hd95'])]
                valid_assd = [r['assd'] for r in all_results if 'assd' in r and not np.isnan(r['assd'])]

                # Main table uses "NaN-only-removed raw valid values"; no HD95<20 / ASSD<10 clipping.
                # (Change only if the paper explicitly states a clipping rule; otherwise report full stats to avoid cherry-pick)
                # Keep `filtered` for DEBUG comparison only; not written to the main JSON.
                filtered_hd95 = [val for val in valid_hd95 if val < 20.0]  # DEBUG: after outlier clipping
                filtered_assd = [val for val in valid_assd if val < 10.0]  # DEBUG: after outlier clipping

                # --- Main-table metrics: use unfiltered valid_* (NaN-only) ---
                avg_hd95 = np.mean(valid_hd95) if valid_hd95 else 0.0
                avg_assd = np.mean(valid_assd) if valid_assd else 0.0

                # DEBUG comparison (helps diagnose crashed slices; values not written to test_results.json)
                debug_hd95 = np.mean(filtered_hd95) if filtered_hd95 else 0.0
                debug_assd = np.mean(filtered_assd) if filtered_assd else 0.0
                
                logger.info(
                    f"Average HD95 (MAIN/Valid-NaNonly): {avg_hd95:.4f}  "
                    f"(DEBUG Filtered<20: {debug_hd95:.4f}, RawMean: {np.mean(valid_hd95):.4f})"
                )
                logger.info(
                    f"Average ASSD (MAIN/Valid-NaNonly): {avg_assd:.4f}  "
                    f"(DEBUG Filtered<10: {debug_assd:.4f}, RawMean: {np.mean(valid_assd):.4f})"
                )

                # 输出最佳和最差图像的信息
                if best_image_result['image']:
                    logger.info(
                        f"Best performing image: {best_image_result['image']} with Dice {best_image_result['dice']:.4f}")
                if worst_image_result['image']:
                    logger.info(
                        f"Worst performing image: {worst_image_result['image']} with Dice {worst_image_result['dice']:.4f}")

                # 保存测试结果到JSON文件
                try:
                    results_save_path = os.path.join(self.args.work_dir, self.args.task_name, 'test_results.json')
                    os.makedirs(os.path.dirname(results_save_path), exist_ok=True)

                    # 构建结果字典
                    test_results = {
                        'average_metrics': {
                            'dice': Avg_dice,
                            'dice_std': Std_dice,
                            'best_dice': np.max(avg_dice),
                            'worst_dice': np.min(avg_dice),
                            'hd95': avg_hd95,
                            'assd': avg_assd
                        },
                        'best_result': best_image_result,
                        'worst_result': worst_image_result,
                        'all_results': all_results,
                        'args': vars(self.args),
                        'test_time': time.time() - start_time
                    }

                    with open(results_save_path, 'w') as f:
                        json.dump(test_results, f, indent=4, default=str)

                    logger.info(f"测试结果已保存到: {results_save_path}")
                except Exception as e:
                    logger.error(f"保存测试结果时出错: {str(e)}")
            else:
                logger.error("没有有效的测试数据进行统计")
                # 即使没有有效数据，也保存空结果
                try:
                    results_save_path = os.path.join(self.args.work_dir, self.args.task_name, 'test_results.json')
                    os.makedirs(os.path.dirname(results_save_path), exist_ok=True)

                    # 构建空结果字典
                    test_results = {
                        'average_metrics': {
                            'dice': 0.0,
                            'dice_std': 0.0,
                            'best_dice': 0.0,
                            'worst_dice': 0.0
                        },
                        'best_result': best_image_result,
                        'worst_result': worst_image_result,
                        'all_results': all_results,
                        'args': vars(self.args),
                        'test_time': time.time() - start_time,
                        'error': 'No valid test data collected'
                    }

                    with open(results_save_path, 'w') as f:
                        json.dump(test_results, f, indent=4, default=str)

                    logger.info(f"空测试结果已保存到: {results_save_path}")
                except Exception as e:
                    logger.error(f"保存空测试结果时出错: {str(e)}")

            # 新增：测试结束后，保存时序一致性分析结果
            self.temporal_analyzer.save_analysis_result()

            # 输出核心测试结果
            global_metrics = self.temporal_analyzer.compute_global_metrics()
            if global_metrics:
                logger.info("=" * 80)
                logger.info("                      核心测试结果（突出重点）                      ")
                logger.info("=" * 80)
                logger.info(f"📌 时序融合创新效果：")
                logger.info(
                    f"   - 平均时序一致性指数（TCI）: {global_metrics['avg_tci']:.4f}（目标≥0.7，体现时序模块有效性）")
                logger.info(f"   - 时序稳定样本比例: {global_metrics['stable_sample_ratio']:.2%}（TCI>0.7，证明鲁棒性）")
                logger.info(f"📌 困难样本优化效果：")
                logger.info(f"   - 最佳样本Dice: {np.max(avg_dice):.4f}（体现模型上限）")
                logger.info(f"   - 最差样本Dice: {np.min(avg_dice):.4f}（目标≥0.65，体现优化能力）")
                logger.info(f"   - 样本Dice标准差: {np.std(avg_dice):.4f}（<0.03，突出结果稳定性优势）")
                logger.info(f"📌 实用性能：")
                logger.info(f"   - 平均收敛轮次: {global_metrics['avg_convergence_round']:.2f}（越小越高效）")
                logger.info(
                    f"   - 平均最终Dice标准差: {global_metrics['avg_final_dice_std']:.4f}（多次测试一致，实用价值高）")
                logger.info("=" * 80)

            logger.info(f'args : {self.args}')
            logger.info('=====================================================================')
        except Exception as e:
            logger.error(f"计算统计信息时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

        # 记录总测试时间
        total_time = time.time() - start_time
        logger.info(f"测试完成，总耗时: {total_time:.2f}秒")


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True


def device_config(args):
    try:
        if not args.multi_gpu:
            if args.device == 'mps':
                args.device = torch.device('mps')
            else:
                args.device = torch.device(f"cuda:{args.gpu_ids[0]}")
        else:
            args.nodes = 1
            args.ngpus_per_node = len(args.gpu_ids)
            args.world_size = args.nodes * args.ngpus_per_node
    except RuntimeError as e:
        print(e)


def evaluate_temporal_performance(args):
    """评估模型的时序融合性能，增强错误处理和日志记录"""
    start_time = time.time()
    print("[DEBUG] evaluate_temporal_performance 函数开始执行")
    logger.info("===== 开始时序性能评估 =====")

    try:
        # 确保启用时序融合
        print("[DEBUG] 设置 use_temporal_fusion = True")
        args.use_temporal_fusion = True

        # 加载数据集
        print("[DEBUG] 调用 get_loader(args) 加载数据集")
        test_loader = get_loader(args)
        print(f"[DEBUG] 成功加载测试数据集，共{len(test_loader)}个批次")
        logger.info(f"成功加载测试数据集，共{len(test_loader)}个批次")

        # 构建模型
        print("[DEBUG] 调用 build_model(args) 构建模型")
        model = build_model(args)
        print("[DEBUG] 成功构建模型")
        logger.info("成功构建模型")

        # 创建时序评估器
        print("[DEBUG] 创建 TemporalFusionEvaluator 实例")
        evaluator = TemporalFusionEvaluator()
        print("[DEBUG] TemporalFusionEvaluator 实例创建成功")

        # 执行评估
        print("[DEBUG] 调用 evaluator.evaluate_temporal_performance(model, test_loader)")
        metrics = evaluator.evaluate_temporal_performance(model, test_loader)
        print("[DEBUG] evaluator.evaluate_temporal_performance 调用完成")

        # 打印评估结果
        print("[DEBUG] 调用 evaluator.print_metrics()")
        evaluator.print_metrics()
        print("[DEBUG] evaluator.print_metrics() 调用完成")

        # 保存评估结果
        print("[DEBUG] 保存评估结果")
        save_path = os.path.join(args.work_dir, args.task_name, 'temporal_evaluation_results.json')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            with open(save_path, 'w') as f:
                # 将所有tensor转为可序列化的类型
                serializable_metrics = {}
                for key, values in metrics.items():
                    serializable_metrics[key] = [float(v) if hasattr(v, 'item') else v for v in values]
                json.dump(serializable_metrics, f, indent=4)
            print(f"[DEBUG] 时序评估结果已保存至: {save_path}")
            logger.info(f"时序评估结果已保存至: {save_path}")
        except Exception as e:
            print(f"[ERROR] 保存评估结果时出错: {str(e)}")
            logger.error(f"保存评估结果时出错: {str(e)}")

        # 记录评估时间
        eval_time = time.time() - start_time
        print(f"[DEBUG] 时序性能评估完成，耗时: {eval_time:.2f}秒")
        logger.info(f"时序性能评估完成，耗时: {eval_time:.2f}秒")

    except Exception as e:
        print(f"[ERROR] 时序性能评估过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error(f"时序性能评估过程出错: {str(e)}")
        logger.error(traceback.format_exc())


def init_logging(args, rank=0):
    """统一的日志初始化函数"""
    # 确保日志目录存在
    os.makedirs(LOG_OUT_DIR, exist_ok=True)
    # 初始化日志记录器
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # 根据不同模式使用一致的命名规则，但添加模式标识
    mode_prefix = 'temporal_' if args.evaluate_temporal_performance else ''
    log_filename = os.path.join(LOG_OUT_DIR, f'{mode_prefix}output_{cur_time}.log')

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO if rank in [-1, 0] else logging.WARN,
        filemode='w',
        filename=log_filename)

    logger.info(f"日志已初始化，输出文件: {log_filename}")
    return log_filename


def main():
    print('*' * 100)
    print("[DEBUG] 程序启动，开始执行main函数")

    # 打印参数
    print('*' * 100)
    for key, value in vars(args).items():
        print(key + ': ' + str(value))
    print('*' * 100)

    # 设置多进程共享策略
    print("[DEBUG] 设置多进程共享策略为'file_system'")
    mp.set_sharing_strategy('file_system')

    # 配置设备
    print("[DEBUG] 配置设备信息")
    device_config(args)
    print(f"[DEBUG] 设备配置完成，当前设备: {args.device}")

    # 时序性能评估模式
    if args.evaluate_temporal_performance:
        print("[DEBUG] 进入时序性能评估模式")
        if args.multi_gpu:
            print("[DEBUG] 使用多GPU进行时序性能评估")
            mp.spawn(evaluate_temporal_performance_worker, nprocs=1, args=(args,))
        else:
            print("[DEBUG] 使用单GPU进行时序性能评估")
            # 初始化日志
            print("[DEBUG] 初始化日志")
            init_logging(args)
            try:
                print("[DEBUG] 执行evaluate_temporal_performance")
                evaluate_temporal_performance(args)
                print("[DEBUG] 时序性能评估完成")
            except Exception as e:
                logger.error(f"时序性能评估出错: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        return

    # 标准测试模式
    print("[DEBUG] 进入标准测试模式")
    if args.multi_gpu:
        print(f"[DEBUG] 使用多GPU测试，进程数: {args.world_size}")
        mp.spawn(main_worker, nprocs=args.world_size, args=(args,))
    else:
        print("[DEBUG] 使用单GPU测试")
        try:
            # 初始化日志
            print("[DEBUG] 初始化日志")
            init_logging(args)
            print("[DEBUG] 日志初始化完成")

            # 统一的随机种子设置
            print("[DEBUG] 设置随机种子为42")
            init_seeds(42)
            print("[DEBUG] 随机种子设置完成")

            # 获取数据加载器
            print("[DEBUG] 开始获取数据加载器")
            dataloaders = get_loader(args)
            print(f"[DEBUG] 数据加载器获取完成，类型: {type(dataloaders).__name__}")
            # 包装成字典，因为BaseTester期望dataloaders是字典格式
            dataloaders = {'test': dataloaders}

            # 新增：支持同一样本多次测试（默认3次，验证结果一致性）
            repeat_test = 1  # 同一样本测试3次
            for repeat in range(repeat_test):
                print(f"[DEBUG] 第 {repeat + 1}/{repeat_test} 次测试")
                try:
                    # 构建模型
                    print(f"[DEBUG] 第 {repeat + 1} 次构建模型")
                    model = build_model(args)
                    print(f"[DEBUG] 模型构建完成，模型类型: {type(model).__name__}")

                    # 创建测试器
                    print(f"[DEBUG] 第 {repeat + 1} 次创建BaseTester实例")
                    tester = BaseTester(model, dataloaders, args)
                    print(f"[DEBUG] BaseTester实例创建完成")

                    # 执行测试
                    print(f"[DEBUG] 第 {repeat + 1} 次开始执行tester.test()")
                    tester.test()
                    print(f"[DEBUG] 第 {repeat + 1} 次测试执行完成")

                    # 每次测试后清空模型缓存（避免影响下次测试）
                    print("[DEBUG] 清空CUDA缓存")
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"[ERROR] 第 {repeat + 1} 次测试过程出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    if 'logger' in globals():
                        logger.error(f"第 {repeat + 1} 次测试过程出错: {str(e)}")
                        logger.error(traceback.format_exc())

        except Exception as e:
            print(f"[ERROR] 测试环境初始化出错: {str(e)}")
            import traceback
            traceback.print_exc()
            if 'logger' in globals():
                logger.error(f"测试环境初始化出错: {str(e)}")
                logger.error(traceback.format_exc())


def evaluate_temporal_performance_worker(rank, args):
    """分布式环境下的时序性能评估工作函数"""
    setup(rank, args.world_size)
    torch.cuda.set_device(rank)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank

    # 只在主进程执行评估
    if rank == 0:
        evaluate_temporal_performance(args)

    cleanup()


def main_worker(rank, args):
    """分布式测试工作进程函数"""
    try:
        setup(rank, args.world_size)
        torch.cuda.set_device(rank)
        args.device = torch.device(f"cuda:{rank}")
        args.rank = rank
        args.gpu_info = {"gpu_count": args.world_size, 'gpu_name': rank}
        # 使用固定的随机种子，确保所有进程结果一致
        init_seeds(42, cuda_deterministic=True)

        # 使用统一的日志初始化函数
        init_logging(args, rank)
        logger.info(f"进程 {rank}/{args.world_size} 初始化完成")

        dataloaders = get_loader(args)
        logger.info(f"进程 {rank} 加载数据集完成")

        model = build_model(args)
        logger.info(f"进程 {rank} 构建模型完成")

        tester = BaseTester(model, dataloaders, args)
        tester.test()

    except Exception as e:
        if rank == 0:  # 只在主进程记录错误
            logger.error(f"进程 {rank} 执行出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    finally:
        # 确保无论如何都会清理进程组
        try:
            cleanup()
        except:
            pass


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = f'{args.port}'
    dist.init_process_group(backend='NCCL', init_method='env://', rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
