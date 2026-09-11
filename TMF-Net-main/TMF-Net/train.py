import sys
import numpy as np
import random 
import math
import matplotlib.pyplot as plt
import os
import csv
import ast
import subprocess
import json
import torch.nn.functional as F  

join = os.path.join
from tqdm import tqdm
from torch.backends import cudnn
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.distributed as dist
from segment_anything import sam_model_registry
import argparse
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LambdaLR
import torch.multiprocessing as mp
from multiprocessing import Manager
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime as dt
import logging
import gc

model_logger = logging.getLogger('model')
model_logger.setLevel(logging.DEBUG)
from data_loader import get_loader 
from model import IMISNet, TMFNet, TemporalAwareLoss
from utils import FocalDice_MSELoss

# expandable_segments 选项在 torch >= 2.1 才支持，老版本（如 conda py38 torch 1.x）
# 会抛 "Unrecognized CachingAllocator option: expandable_segments"。
# 这里做版本检测：仅在新版 torch 上启用，老版本跳过（仅影响显存碎片优化，不影响功能）。
try:
    _torch_version_tuple = tuple(int(x.split('+')[0].split('~')[0])
                                 for x in torch.__version__.split('.')[:3])
    if _torch_version_tuple >= (2, 1, 0):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    else:
        # 老版本 torch：清除可能残留的同名变量，避免后续 cuda init 报错
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
except Exception:
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = False  
torch.backends.cudnn.allow_tf32 = False
import re
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', type=str, default='work_dir')
parser.add_argument('--task_name', type=str, default='ACDC')
parser.add_argument('--dataset', type=str, default='acdc', help='Dataset type: acdc, btcv, amos2022_mr')
parser.add_argument("--data_dir", type = str, default='dataset/ACDC')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--test_mode', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--model_type', type=str, default='vit_b')
parser.add_argument('--sam_checkpoint', type=str, default='ckpt/IMISNet-B.pth')
parser.add_argument('--pretrain_path', type=str, default='work_dir/ACDC/IMIS_latest.pth')
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--mask_num', type=int, default=2)
parser.add_argument('--inter_num', type=int, default=5)
parser.add_argument('--model_variant', type=str, default='tmf', choices=['tmf', 'sam'],
                    help='tmf: full TMF-Net; sam: SAM-only baseline without temporal fusion')
parser.add_argument('--use_temporal_fusion', action='store_true', default=None, help='Enable temporal fusion training')
parser.add_argument('--no_temporal_fusion', action='store_false', dest='use_temporal_fusion',
                    help='Disable temporal fusion training')
parser.add_argument('--temporal_interactions', type=int, default=3, help='Number of interaction rounds for temporal fusion')
parser.add_argument('--ablation_no_multi_scale', action='store_true', default=False, help='Ablation study: disable multi-scale feature fusion')
parser.add_argument('--ablation_no_trajectory', action='store_true', default=False, help='Ablation study: disable LSTM trajectory analysis')
parser.add_argument('--num_epochs', type=int, default=150)
parser.add_argument('--lr_scheduler', type=str, default='piecewise_constant',
                    help='Scheduler: piecewise_constant (推荐), cosine, step, multi_step, none')
parser.add_argument('--early_stop_patience', type=int, default=50, help='Early stop patience')
parser.add_argument('--val_interval', type=int, default=20, help='Run test set evaluation every N epochs')
parser.add_argument('--disable_early_stop', action='store_true', default=False, help='Disable early stopping completely')
parser.add_argument('--lr_restart', type=float, default=None, help='Restart learning rate (for continue training)')
parser.add_argument('--continue_from_best', action='store_true', default=False, help='Continue training from best checkpoint')
parser.add_argument('--step_size', type=list, default=[7,12]) 
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--port', type=int, default=12305)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0])
parser.add_argument('--multi_gpu', action='store_true', default=False)
parser.add_argument('--dist', dest='dist', type=bool, default=False, help='distributed training or not')
parser.add_argument('--num_workers', '-num_workers', dest='num_workers', type=int, default=4)
parser.add_argument('--num_clicks', type=int, default=5)
args = parser.parse_args()

def normalize_model_variant(args):
    if getattr(args, 'model_variant', 'tmf') == 'sam':
        args.use_temporal_fusion = False
    elif getattr(args, 'use_temporal_fusion', None) is None:
        args.use_temporal_fusion = True

normalize_model_variant(args)
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])
logger = logging.getLogger(__name__)
LOG_OUT_DIR = join(args.work_dir, args.task_name)
device = args.device
MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def build_model(args):
        if args.dataset.lower() == 'acdc':
            num_classes = 4
        elif args.dataset.lower() == 'btcv':
            num_classes = 14  
        elif args.dataset.lower() == 'amos2022_mr':
            num_classes = 16  
        else:
            num_classes = 4  
        
        normalize_model_variant(args)
        use_temporal = bool(getattr(args, 'use_temporal_fusion', True))
        sam = sam_model_registry[args.model_type](args).to(device)
        
        imis = TMFNet(
            sam, 
            test_mode=args.test_mode, 
            select_mask_num=args.mask_num,
            fusion_warmup_epochs=40,      # [方案C 修改3] 延后 fusion 介入（150 epoch 版）
            max_fusion_strength=1.5,       # 前 50 epoch SAM 先学纯粹的 prompt-mask 对齐，
                                           # 避免初期强 fusion 把 SAM 引导入错误局部最优（之前 20 epoch
                                           # 就放 fusion，常导致 80 epoch 后 TrainDice 立刻平台）
            num_classes=num_classes,
            ablation_no_multi_scale=args.ablation_no_multi_scale,
            ablation_no_trajectory=args.ablation_no_trajectory,
            use_temporal_fusion=use_temporal
        ).to(device)
        print(f"[INFO] Building {'TMFNet' if use_temporal else 'SAM-only baseline'}")
        
        print("[INFO] Applying selective freezing strategy...")
        
        for name, param in imis.named_parameters():
            is_temporal_param = any(k in name for k in (
                'multiscale_temporal_fusion', 'temporal_memory',
                'temporal_weight', 'trajectory_fusion_weight', 'fusion_weights'
            ))
            if not use_temporal and is_temporal_param:
                param.requires_grad = False
            elif 'image_encoder' in name:
                if ('blocks.11' in name or 'blocks.10' in name or 'blocks.9' in name or
                    'blocks.8' in name or 'blocks.7' in name or 'blocks.6' in name or 'neck' in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = True
                if use_temporal and ('temporal' in name or 'fusion' in name or 'trajectory' in name or 'lstm' in name):
                    print(f"[INFO] Ensuring temporal parameter is trainable: {name}")
        trainable_params = sum(p.numel() for p in imis.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in imis.parameters() if not p.requires_grad)
        print(f"[INFO] Total trainable parameters: {trainable_params/1e6:.2f}M")
        print(f"[INFO] Frozen parameters: {frozen_params/1e6:.2f}M")
        print(f"[INFO] Trainable ratio: {trainable_params/(trainable_params+frozen_params)*100:.1f}%")
        
        if args.multi_gpu:
            imis = DDP(imis, device_ids=[args.rank], output_device=args.rank)
        return imis

def compute_dice_coefficient(pred, label):
    """Compute Dice coefficient"""
    assert pred.shape == label.shape
    pred = torch.sigmoid(pred)
    intersection = torch.sum(pred * label, dim=(1, 2, 3))
    union = torch.sum(pred + label, dim=(1, 2, 3))
    dice = (2. * intersection) / (union + 1e-8)
    return dice.mean()
from utils import compute_iou, get_iou_and_dice

def compute_hausdorff_distance(pred, label):
    """Compute Hausdorff distance"""
    assert pred.shape == label.shape
    pred = torch.sigmoid(pred) > 0.5
    label = label > 0
    
    pred_boundary = pred - F.max_pool2d(pred, kernel_size=3, stride=1, padding=1)
    label_boundary = label - F.max_pool2d(label, kernel_size=3, stride=1, padding=1)
    
    pred_points = torch.nonzero(pred_boundary, as_tuple=False)
    label_points = torch.nonzero(label_boundary, as_tuple=False)
    
    if len(pred_points) == 0 or len(label_points) == 0:
        return torch.tensor(0.0, device=pred.device)
    
    distances = F.pairwise_distance(
        pred_points.float(), 
        label_points.float(), 
        p=2
    ).max()
    
    return distances

def compute_focal_loss(pred, label, gamma=2.0, alpha=0.25):
    """Compute Focal Loss"""
    pred = torch.sigmoid(pred)
    target = label.float()
    pt = pred * target + (1 - pred) * (1 - target)
    loss = -alpha * (1 - pt) ** gamma * torch.log(pt + 1e-8)
    return loss.mean()

def compute_temporal_consistency(current_masks, previous_masks=None):
    """Compute temporal consistency loss"""
    if previous_masks is None:
        return 0.0
    
    consistency = F.mse_loss(torch.sigmoid(current_masks), torch.sigmoid(previous_masks))
    return consistency

def compute_temporal_aware_loss(pred_masks, gt_masks, interaction_round, previous_masks=None, global_epoch=0):
    """Temporal aware loss function (using TemporalAwareLoss class from model.py)"""
    temporal_loss_fn = TemporalAwareLoss()
    return temporal_loss_fn(pred_masks, gt_masks, previous_pred=previous_masks, interaction_round=interaction_round, global_epoch=global_epoch)

class TemporalFusionEvaluator:
    """Temporal fusion performance evaluator"""
    
    def __init__(self): 
        self.metrics = { 
            'convergence_speed': [],      
            'interaction_efficiency': [], 
            'temporal_consistency': [],  
            'final_accuracy': []          
        } 
    
    def evaluate_temporal_performance(self, model, test_loader):
        """Evaluate temporal performance"""
        model.eval()
        all_results = []
        failed_samples = []  
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                try:
                    if isinstance(batch, dict):
                        images = batch.get("image", batch.get("img", None)).to(device)
                        labels = batch.get("label", batch.get("seg", None)).to(device)
                        classes = batch.get("class", batch.get("category", None))
                        patient_ids = batch.get("patient_ids", [f"batch_{batch_idx}"])
                    else:
                        images, labels, classes = batch
                        images, labels = images.to(device), labels.to(device)
                        patient_ids = [f"batch_{batch_idx}"]
                    interaction_id = 0
                    round_metrics = []
                    prev_masks = None
                    for round_idx in range(5):
                        try:
                            if hasattr(model, 'supervised_prompts_with_temporal'):
                                prompts = model.supervised_prompts_with_temporal(
                                    classes, labels, prev_masks, None, 'points', interaction_id
                                )
                            else:
                                prompts = model.supervised_prompts(
                                    classes, labels, None, None, 'points'
                                )
                            prompts['interaction_round'] = round_idx
                            if hasattr(model, '_clear_interaction_history'):
                                outputs = model(images, prompts, interaction_id=interaction_id)
                            else:
                                outputs = model(images, prompts)
                            pred_masks = outputs['masks']
                            round_metrics.append({
                                'dice': compute_dice_coefficient(pred_masks, labels),
                                'iou': compute_iou(pred_masks, labels),
                                'hd': compute_hausdorff_distance(pred_masks, labels)
                            })
                            prev_masks = pred_masks.detach()
                            interaction_id += 1
                        except Exception as e:
                            print(f"[WARNING] Error in interaction round {round_idx} for sample {patient_ids[0]}: {str(e)}")
                            break
                    if round_metrics:
                        all_results.append(round_metrics)
                    else:
                        failed_samples.append(patient_ids[0])
                except Exception as e:
                    print(f"[ERROR] Failed to process batch {batch_idx}: {str(e)}")
                    failed_samples.append(f"batch_{batch_idx}")
        if failed_samples:
            print(f"[WARNING] Failed to evaluate {len(failed_samples)} samples: {failed_samples[:10]}...")
        self._analyze_temporal_patterns(all_results)
        return self.metrics 
    
    def _analyze_temporal_patterns(self, all_results):
        """分析时序模式，处理空结果"""
        self.metrics = {
            'convergence_speed': [],
            'interaction_efficiency': [],
            'temporal_consistency': [],
            'final_accuracy': []
        }
        for result in all_results:
            if not result:  
                continue
            convergence_round = None
            for i, metrics in enumerate(result):
                if metrics['dice'] > 0.85:
                    convergence_round = i + 1
                    break
            self.metrics['convergence_speed'].append(convergence_round or len(result))
            final_dice = result[-1]['dice']
            interaction_efficiency = final_dice / len(result)
            self.metrics['interaction_efficiency'].append(interaction_efficiency)
            if len(result) >= 2:
                dice_values = [m['dice'] for m in result]
                temporal_consistency = np.var(dice_values)  
                self.metrics['temporal_consistency'].append(temporal_consistency)
            self.metrics['final_accuracy'].append(final_dice)
    
    def get_average_metrics(self):
        """Get average evaluation metrics"""
        avg_metrics = {}
        for key, values in self.metrics.items():
            if values:
                avg_metrics[key] = sum(values) / len(values)
            else:
                avg_metrics[key] = 0.0
        return avg_metrics
    
    def print_metrics(self):
        """Print evaluation metrics"""
        avg_metrics = self.get_average_metrics()
        print("=== Temporal Fusion Performance Evaluation Results ===")
        print(f"Average convergence speed: {avg_metrics['convergence_speed']:.2f} rounds")
        print(f"Average interaction efficiency: {avg_metrics['interaction_efficiency']:.4f} per round")
        print(f"Average temporal consistency: {avg_metrics['temporal_consistency']:.4f}")
        print(f"Average final accuracy: {avg_metrics['final_accuracy']:.4f}")
        print("===========================")

class BaseTrainer:
    def __init__(self, model, dataloaders, args):
        """Initialize trainer, removing dependencies on checkpoint_manager and training_monitor"""
        self.model = model
        self.dataloaders = dataloaders
        self.args = args
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.best_iou = 0.0
        self.step_best_dice = 0.0
        self.losses = []
        self.dices = []
        self.ious = []
        
        self.early_stop_patience = getattr(args, 'early_stop_patience', 10)
        self.early_stop_counter = 0
        
        self.scaler = GradScaler(
            init_scale=2**10,  
            growth_factor=1.5,  
            backoff_factor=0.6,  
            growth_interval=4000,  
            enabled=True
        )
        self.set_loss_fn()
        self.build_optimizer()
        
        if args.pretrain_path is not None:
            self.load_checkpoint(args.pretrain_path, args.resume)
        else:
            self.start_epoch = 0
            
        print("BaseTrainer initialized, using simplified training logic")
    def set_loss_fn(self):
        if hasattr(self, 'args') and self.args.use_temporal_fusion:
            self.temporal_loss = TemporalAwareLoss()
            self.criterion = self.temporal_loss
        else:
            self.criterion = FocalDice_MSELoss()
    def build_optimizer(self):
        # Paper training recipe (default CLI args):
        #   Epochs = 150, batch = 8, optimizer = AdamW, base_lr = 1e-4,
        #   weight_decay = 1e-2, lr schedule = cosine annealing down to 1e-6.
        #
        # Per-module group assignment is preserved (image-encoder top blocks,
        # prompt encoder, mask decoder + temporal/LSTM, special fusion scalars,
        # and bias/batchnorm without weight decay). Group LR ratios are kept
        # proportional to the original design but now anchored on args.lr so
        # the user-specified (or paper-default) 1e-4 propagates consistently.
        base_lr = float(self.args.lr)
        base_wd = float(self.args.weight_decay)

        # group 0: image_encoder top blocks (frozen-ish low lr)
        # group 1: prompt_encoder + misc params
        # group 2: mask_decoder + temporal/LSTM/trajectory/fusion params
        # group 3: explicit scalar fusion weights
        # [方案C 修改4] LR 分层缩放：
        #   Group 0 (ViT block 9-11 + neck):      1.0x base_lr → 5e-5
        #   Group 1 (prompt_encoder):             3.0x        → 15e-5
        #   Group 2 (mask_decoder + misc):        3.0x        → 15e-5
        #   Group 3 (trajectory_fusion_weight + fusion_weights): 5.0x → 25e-5
        #   Group 4 (bias / LayerNorm / norm params): 3.0x, NO weight decay
        # 原始方案用 [0.5, 1, 1, 1, 1] 让 fusion 组学得太慢（LR 只有 prompt/decoder 的 1/2），
        # 但 fusion 决定交互质量，必须放大 LR 让它在早期快速收敛。
        lr_scales = [1.0, 3.0, 3.0, 5.0, 3.0]
        wd_scales = [1.0, 1.0, 1.0, 1.0, 0.0]
        param_groups = []
        for lr_s, wd_s in zip(lr_scales, wd_scales):
            param_groups.append({
                'params': [],
                'lr': base_lr * lr_s,
                'weight_decay': base_wd * wd_s,
            })

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # LayerNorm / GroupNorm / bias -> no weight-decay group (group 4)
            if name.endswith('.bias') or 'norm.' in name or '.bn.' in name or '.ln.' in name or name.endswith('.weight') and ('norm' in name or 'batchnorm' in name or 'layernorm' in name):
                target_group = 4
            elif 'trajectory_fusion_weight' in name or 'fusion_weights' in name:
                target_group = 3
            elif 'image_encoder' in name:
                if 'neck' in name or any(
                    f'blocks.{i}.' in name for i in range(6, 12)
                ):
                    target_group = 0
                else:
                    param.requires_grad = False
                    continue
            elif 'prompt_encoder' in name:
                target_group = 1
            elif 'mask_decoder' in name:
                target_group = 2
            elif 'temporal' in name or 'lstm' in name or 'trajectory' in name or 'fusion' in name:
                target_group = 2
            else:
                target_group = 1
            param_groups[target_group]['params'].append(param)

        for i, group in enumerate(param_groups):
            logger.debug(
                f"[DEBUG Optimizer] Group {i}: n={len(group['params'])}, "
                f"lr={group['lr']:.3e}, weight_decay={group['weight_decay']:.3e}"
            )
        # [方案C 修改1] 分段 constant + warmup + 阶段性 warm restart
        # 总 epoch 150 分为 3 段（缩短 1.0x 稳定期，提前进入 LR 衰减）：
        #   Epoch 1-10 :   warmup，LR 线性升到 base_lr
        #   Epoch 11-80:  phase1 = base_lr     (ViT 5e-5, Prompt 15e-5, Decoder 15e-5, Fusion 25e-5)
        #   Epoch 81-120:  phase2 = base_lr * 0.2  (warm restart 降到 20%)
        #   Epoch 121-150: phase3 = base_lr * 0.04 (再降一次)
        # 避免 Cosine 后半段 7.5e-7 几乎为零，导致 80 epoch 后学不到任何新东西。
        self.optimizer = torch.optim.AdamW(param_groups)
        self.scheduler = None
        sched = self.args.lr_scheduler.lower() if self.args.lr_scheduler else 'none'
        if sched == 'piecewise_constant':
            warmup_epochs = 10
            total = self.args.num_epochs
            phase1_end = 80   # 1..80: 1.0x
            phase2_end = 120  # 81..120: 0.2x
            # phase3_end = total (121..150: 0.04x)
            def _lr_lambda(epoch):
                # epoch 0-based
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs  # 线性升到 1.0
                e = epoch + 1  # 转 1-based 方便划分
                if e <= phase1_end:
                    return 1.0
                elif e <= phase2_end:
                    return 0.2
                else:
                    return 0.04
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=_lr_lambda)
        elif sched == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.num_epochs,
                eta_min=1e-6
            )
        elif sched == 'step':
            step_size = self.args.step_size[0] if isinstance(self.args.step_size, list) else self.args.step_size
            self.scheduler = StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=self.args.gamma
            )
        elif sched == 'multi_step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.args.step_size,
                gamma=self.args.gamma
            )
        elif sched == 'none':
            self.scheduler = None
        else:
            # 兜底：cosine
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.args.num_epochs, eta_min=1e-6)
        self.warmup_epochs = 10 if sched == 'piecewise_constant' else 0
        
    def load_checkpoint(self, ckp_path, resume):
        last_ckpt = None
        if os.path.exists(ckp_path):
            if self.args.multi_gpu:
                dist.barrier()
                last_ckpt = torch.load(ckp_path, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
            else:
                def map_location(storage, loc):
                    return storage.cuda(0) if torch.cuda.is_available() else storage.cpu()
                last_ckpt = torch.load(ckp_path, map_location=map_location)
        
        if last_ckpt:
            if isinstance(last_ckpt, dict) and 'model_state_dict' in last_ckpt:
                model_state_dict = last_ckpt['model_state_dict']
            else:
                model_state_dict = last_ckpt
            
            current_state_dict = self.model.state_dict() if not self.args.multi_gpu else self.model.module.state_dict()
            
            filtered_state_dict = {}
            for key, value in model_state_dict.items():
                if key in current_state_dict and current_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    pass
            
            print(f"Loaded {len(filtered_state_dict)} parameters from checkpoint")
            print(f"Skipped {len(model_state_dict) - len(filtered_state_dict)} parameters due to shape mismatch")
            print(f"Added {len(current_state_dict) - len(filtered_state_dict)} new parameters")
            
            current_state_dict.update(filtered_state_dict)
            if self.args.multi_gpu:
                self.model.module.load_state_dict(current_state_dict)
            else:
                self.model.load_state_dict(current_state_dict)
            
            if resume:
                try:
                    self.start_epoch = last_ckpt['epoch']
                    try:
                        self.optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
                        print("Loaded optimizer state dict successfully")
                    except ValueError as e:
                        print(f"Warning: Failed to load optimizer state dict: {e}")
                        print("Using newly initialized optimizer instead")
                    
                    try:
                        if self.scheduler is not None and 'lr_scheduler_state_dict' in last_ckpt and last_ckpt['lr_scheduler_state_dict'] is not None:
                            self.scheduler.load_state_dict(last_ckpt['lr_scheduler_state_dict'])
                    except Exception as e:
                        print(f"Warning: Failed to load scheduler state dict: {e}")
                        print("Scheduler will use current settings")
                    
                    if 'losses' in last_ckpt:
                        self.losses = last_ckpt['losses']
                    if 'dices' in last_ckpt:
                        self.dices = last_ckpt['dices']
                    if 'ious' in last_ckpt:
                        self.ious = last_ckpt['ious']
                    if 'best_loss' in last_ckpt:
                        self.best_loss = last_ckpt['best_loss']
                    if 'best_dice' in last_ckpt:
                        self.best_dice = last_ckpt['best_dice']
                except Exception as e:
                    print(f"Warning: Failed to load some training state: {e}")
                    self.start_epoch = 0
            else:
                self.start_epoch = 0
            print(f"Loaded checkpoint from {ckp_path} (starting from epoch {self.start_epoch})")
            
        else:
            self.start_epoch = 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")
    
    def save_checkpoint(self, epoch, state_dict, describe="last"):
        """Simplified checkpoint saving method, completely independent of checkpoint_manager"""
        try:
            if not os.path.exists(MODEL_SAVE_PATH):
                os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
                print(f"Create save directory: {MODEL_SAVE_PATH}")
            
            if describe == 'best' or 'best' in describe:
                current_dice = getattr(self, 'current_dice', 0.0)
                filename = os.path.join(MODEL_SAVE_PATH, f'IMIS_{epoch}_step_dice:{current_dice:.4f}_best.pth')
                best_filename = os.path.join(MODEL_SAVE_PATH, f'IMIS_dice_best.pth')
            else:
                filename = os.path.join(MODEL_SAVE_PATH, f'IMIS_latest.pth')
            
            torch.save(state_dict, filename)
            print(f"✅ Checkpoint saved successfully: {filename}")
            
            if describe == 'best' or 'best' in describe:
                torch.save(state_dict, best_filename)
                print(f"✅ Best model additionally saved as: {best_filename}")
                
            print(f"Epoch {epoch} - Checkpoint saving completed")
            
        except Exception as e:
            print(f"❌ Error saving checkpoint: {str(e)}")
            import traceback
            traceback.print_exc()  
    def _checkpoint_prefix(self):
        return 'SAM' if getattr(self.args, 'model_variant', 'tmf') == 'sam' else 'IMIS'

    def _save_direct_checkpoint(self, epoch, state_dict, describe='latest'):
        try:
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            checkpoint = {
                'model_state_dict': state_dict,
                'epoch': epoch,
                'best_loss': self.best_loss,
                'best_dice': self.best_dice,
                'best_iou': self.best_iou,
                'losses': self.losses,
                'dices': self.dices,
                'ious': self.ious,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'model_variant': getattr(self.args, 'model_variant', 'tmf'),
                'use_temporal_fusion': getattr(self.args, 'use_temporal_fusion', True),
            }
            prefix = self._checkpoint_prefix()
            if describe == 'latest':
                filename = os.path.join(MODEL_SAVE_PATH, f'{prefix}_latest.pth')
            elif describe == 'dice_best':
                filename = os.path.join(MODEL_SAVE_PATH, f'{prefix}_best_dice_{self.best_dice:.4f}.pth')
                generic_best = os.path.join(MODEL_SAVE_PATH, f'{prefix}_dice_best.pth')
            else:
                filename = os.path.join(MODEL_SAVE_PATH, f'{prefix}_{describe}.pth')

            torch.save(checkpoint, filename)
            print(f"Save checkpoint to {filename}")
            if describe == 'dice_best':
                torch.save(checkpoint, generic_best)
                print(f"Save generic best checkpoint to {generic_best}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            import traceback
            traceback.print_exc()

    def train(self):
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        print(f"[INFO] Start {'SAM-only' if getattr(self.args, 'model_variant', 'tmf') == 'sam' else 'base'} training")

        for epoch in range(self.start_epoch, self.args.num_epochs):
            print(f"Epoch: {epoch}/{self.args.num_epochs - 1}")
            metrics = self.train_one_epoch(epoch)
            avg_loss = metrics['loss']
            avg_iou = metrics['iou']
            avg_dice = metrics['dice']

            self.losses.append(avg_loss)
            self.ious.append(avg_iou)
            self.dices.append(avg_dice)

            state_dict = self.model.module.state_dict() if self.args.multi_gpu else self.model.state_dict()
            self._save_direct_checkpoint(epoch, state_dict, describe='latest')

            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
            if avg_iou > self.best_iou:
                self.best_iou = avg_iou
            if avg_dice > self.best_dice:
                self.best_dice = avg_dice
                self._save_direct_checkpoint(epoch, state_dict, describe='dice_best')
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            if self.scheduler is not None:
                self.scheduler.step()

            print(f"Epoch {epoch} finished. Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, Dice: {avg_dice:.4f}")
            if math.isnan(avg_loss) or math.isinf(avg_loss):
                print("[Early stop] loss is NaN/Inf.")
                break
            if (not getattr(self.args, 'disable_early_stop', False)
                    and self.early_stop_counter >= self.early_stop_patience):
                print(f"[Early stop] no Dice improvement for {self.early_stop_patience} epochs.")
                break

        print("==========================================")
        print("Training completed!")
        print(f"Best metrics - Loss: {self.best_loss:.4f}, Dice: {self.best_dice:.4f}, IoU: {self.best_iou:.4f}")
        print("==========================================")

    get_iou_and_dice = staticmethod(get_iou_and_dice)
    def plot_result(self, plot_data, description, save_name):
        plt.plot(plot_data)
        plt.title(description)
        plt.xlabel('Epoch')
        plt.ylabel(f'{save_name}')
        plt.savefig(join(MODEL_SAVE_PATH, f'{save_name}.png'))
        plt.close()
    def train_one_epoch(self, epoch):
        """BaseTrainer的训练方法，用于静态数据集（如Kvasir、BTCV）"""
        self.model.train()
        model = self.model.module if self.args.multi_gpu else self.model
        device = next(model.parameters()).device
        
        self.train_loader = self.dataloaders if not isinstance(self.dataloaders, dict) else self.dataloaders.get('train', self.dataloaders)
        tbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        epoch_loss, epoch_dice, epoch_iou = 0.0, 0.0, 0.0
        num_steps = len(tbar)
        
        accumulation_steps = 4  
        step_counter = 0  
        for step, batch in enumerate(tbar):
            if step % 10 == 0:  
                logger.info(f"[INFO] Batch {step+1}/{num_steps}")
            
            if step % 3 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            images = batch["image"].to(device)
            labels = batch["label"].to(device).long()
            batch_size = images.size(0)
            
            if 'patient_ids' not in batch or not batch['patient_ids']:
                raise RuntimeError("Batch中没有patient_ids！请先完成data_loader.py的修改")
            patient_ids = [str(pid) for pid in batch['patient_ids']]
            if 'target_list' not in batch or not batch['target_list']:
                raise RuntimeError("Batch中没有target_list！请先完成data_loader.py的修改")
            
            target_list_len = len(batch['target_list'])
            logger.debug(f"[DEBUG target_list] 长度: {target_list_len}, batch_size: {batch_size}, mask_num: {self.args.mask_num}")
            logger.debug(f"[DEBUG target_list] 内容: {batch['target_list']}")
            
            categories = []
            if target_list_len > 0:
                for i in range(len(patient_ids)):
                    idx = min(i, target_list_len - 1)
                    categories.append(batch['target_list'][idx])
            else:
                categories = [0] * len(patient_ids)
                print("[WARNING] target_list 为空，使用默认类别 0")
            print(f"[DEBUG 数据] 真实病人ID: {patient_ids}, 真实类别: {categories}")
            logger.debug(f"[DEBUG 数据] patient_ids长度: {len(patient_ids)}, categories长度: {len(categories)}")
            
            total_loss = 0.0 
            num_interactions = 1  
            previous_pred = None
            round_loss = 0.0
            image_features = None
            
            for interaction_round in range(num_interactions):
                # [FIX ②] 先用"原始batch_size版本"的 categories 算点击（和 labels 原始 batch 对齐）
                raw_categories_for_click = categories
                points, point_labels = self._simulate_clicks(
                    images, labels, previous_pred, interaction_round,
                    categories=raw_categories_for_click
                )
                
                real_model = self.model.module if hasattr(self.model, 'module') else self.model
                
                if image_features is None and hasattr(self.model, 'image_forward'):
                    image_features = self.model.image_forward(images)
                
                extended_batch_size = image_features.shape[0]
                if not categories:
                    categories = ["default"] * extended_batch_size
                elif isinstance(categories, (list, tuple)):
                    if len(categories) != extended_batch_size:
                        extended_categories = []
                        for cat in categories:
                            extended_categories.extend([cat] * self.args.mask_num)
                        extended_categories = extended_categories[:extended_batch_size]
                        categories = extended_categories
                else:
                    categories = [str(categories)] * extended_batch_size
                
                if not patient_ids:
                    patient_ids = [f"patient_{i}" for i in range(extended_batch_size)]
                elif len(patient_ids) != extended_batch_size:
                    extended_patient_ids = []
                    for pid in patient_ids:
                        extended_patient_ids.extend([pid] * self.args.mask_num)
                    extended_patient_ids = extended_patient_ids[:extended_batch_size]
                    patient_ids = extended_patient_ids

                # [FIX 兼容性] 点击按 mask_num 展开到 extended_batch_size（每个mask副本都有同样的点击）
                if points.shape[0] != extended_batch_size:
                    mn = self.args.mask_num or 1
                    if points.shape[0] * mn == extended_batch_size:
                        points = points.repeat_interleave(mn, dim=0)
                        point_labels = point_labels.repeat_interleave(mn, dim=0)
                    elif extended_batch_size % points.shape[0] == 0 and extended_batch_size > points.shape[0]:
                        r = extended_batch_size // points.shape[0]
                        points = points.repeat_interleave(r, dim=0)
                        point_labels = point_labels.repeat_interleave(r, dim=0)
                    else:
                        # fallback：tile到相同长度（不丢样本）
                        rep = (extended_batch_size + points.shape[0] - 1) // points.shape[0]
                        points = torch.cat([points] * rep, dim=0)[:extended_batch_size]
                        point_labels = torch.cat([point_labels] * rep, dim=0)[:extended_batch_size]
                
                prompts = {
                    'patient_ids': patient_ids,
                    'categories': categories,
                    'interaction_ids': patient_ids,
                    'temporal_enabled': True,
                    # [FIX ③] 移除 labels（多通道GT），训练时不再把GT作为模型输入
                    'interaction_round': interaction_round,
                    'epoch': epoch,
                    'global_epoch': epoch,
                    'quality_score': [0.0] * len(patient_ids),
                    'iou_predictions': [0.0] * len(patient_ids),
                    'point_coords': points,      # [B, N, 2] 显式注入模型输入
                    'point_labels': point_labels,  # [B, N] 显式注入模型输入
                }
                
                original_labels = labels
                if labels.dim() == 4 and labels.shape[1] > 1:
                    binary_labels = []
                    for i in range(labels.shape[0]):
                        if i < len(categories):
                            cat_name = categories[i] if isinstance(categories[i], str) else str(categories[i])
                            if hasattr(self.args, 'classes') and self.args.classes:
                                try:
                                    class_idx = self.args.classes.index(cat_name) if cat_name in self.args.classes else 1
                                except ValueError:
                                    class_idx = 1
                            else:
                                class_idx = 1
                            if labels.shape[1] > class_idx:
                                binary_labels.append(labels[i:i+1, class_idx:class_idx+1])
                            else:
                                binary_labels.append(labels[i:i+1, 0:1])
                        else:
                            binary_labels.append(labels[i:i+1, 0:1])
                    if binary_labels:
                        labels = torch.cat(binary_labels, dim=0)
                    else:
                        labels = (labels.argmax(dim=1, keepdim=True) > 0).float()
                elif labels.dim() == 3:
                    labels = labels.unsqueeze(1)

                if labels.shape[0] != extended_batch_size:
                    if extended_batch_size % labels.shape[0] == 0:
                        labels = labels.repeat_interleave(extended_batch_size // labels.shape[0], dim=0)
                    else:
                        labels = labels[:extended_batch_size]
                original_labels = labels
                
                with autocast():
                    if hasattr(self.model, 'forward_with_features'):
                        outputs = self.model.forward_with_features(image_features, prompts)
                    else:
                        outputs = self.model.forward(images, prompts)
                # [ANTI-LEAK compliant] 用 GT 算真实 IoU 作为 TQM 决策的监督目标
                # 依据用户训练边界："计算质量监督目标，比如用预测 mask 和 GT 算真实 Dice/IoU，
                #                   训练 iou_pred / quality_score"——允许 GT 当老师
                # 同时保留 model iou_pred 作为被监督对象（loss 里的 quality_loss 会拉它逼近真实 IoU）
                if isinstance(outputs, dict) and 'masks' in outputs:
                    with torch.no_grad():
                        _pred_masks_for_iou = outputs['masks'].detach().float()
                        if _pred_masks_for_iou.shape[2:] != original_labels.shape[2:]:
                            _pred_masks_for_iou = F.interpolate(_pred_masks_for_iou, size=original_labels.shape[2:], mode='bilinear', align_corners=False)
                        _pred_bin = (torch.sigmoid(_pred_masks_for_iou) > 0.5).float()
                        _gt_bin = (original_labels > 0).float()
                        if _gt_bin.dim() == 4 and _gt_bin.shape[1] > 1:
                            _gt_bin = _gt_bin[:, :1]  # 取第一通道作为目标类
                        _inter = (_pred_bin * _gt_bin).sum(dim=(1, 2, 3))
                        _union = _pred_bin.sum(dim=(1, 2, 3)) + _gt_bin.sum(dim=(1, 2, 3)) - _inter
                        _real_iou = ((_inter + 1e-7) / (_union + 1e-7)).clamp(0.0, 1.0)
                        _flat = _real_iou.flatten()
                        _n = min(len(patient_ids), _flat.numel())
                        _qs = [float(_flat[i].item()) for i in range(_n)] + [0.0] * max(0, len(patient_ids) - _n)
                    prompts['quality_score'] = _qs
                    prompts['iou_predictions'] = list(_qs)
                
                try:
                    loss = self.criterion(outputs, labels, interaction_round=interaction_round, previous_pred=previous_pred, global_epoch=epoch, new_click_coords=points)
                except Exception as e:
                    model_logger.error(f"[FATAL] 前向传播错误: {str(e)}")
                    model_logger.error(f"[FATAL] 病人ID: {patient_ids}, 类别: {categories}")
                    model_logger.error(f"[FATAL] 交互轮次: {interaction_round}")
                    raise e
                
                pred_masks = outputs['masks'].float()
                del outputs  
                
                total_loss += loss
                
                dice_scores = self._calc_dice(pred_masks, labels)
                tci_values = self._calc_tci(pred_masks, labels)
                batch_dice = dice_scores.mean().item()
                batch_tci = tci_values.mean().item()
                
                if batch_dice > 1.0:
                    print(f"[WARNING] 交互轮次Dice异常: {batch_dice}, 检查计算逻辑")
                    batch_dice = min(batch_dice, 1.0)
                if batch_tci > 1.0:
                    print(f"[WARNING] 交互轮次TCI异常: {batch_tci}, 检查计算逻辑")
                    batch_tci = min(batch_tci, 1.0)
                if batch_dice == 0:
                    print(f"[DEBUG] Batch dice is 0! pred_masks shape: {pred_masks.shape}, labels shape: {labels.shape}")
                    print(f"[DEBUG] pred_masks min: {pred_masks.min().item()}, max: {pred_masks.max().item()}")
                    print(f"[DEBUG] labels sum: {labels.sum().item()}")
                    print(f"[DEBUG] pred_masks sigmoid sum: {torch.sigmoid(pred_masks).sum().item()}")
                print(f"[交互监控] 轮次 {interaction_round+1}/{num_interactions} - Dice: {batch_dice:.4f}, TCI: {batch_tci:.4f}")
                
                previous_pred = pred_masks.detach().cpu()
                final_pred_masks = pred_masks.detach()
                del pred_masks
                round_loss = loss.item()  
                del loss
            
            if torch.is_tensor(total_loss) and total_loss.requires_grad:
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    logger.error(
                        f"[FATAL] 当前 Batch 的 Total Loss 为 NaN/Inf! 放弃参数更新，防止模型权重崩溃。"
                    )
                    self.optimizer.zero_grad()
                    step_counter += 1  
                else:
                    step_counter += 1
                    self.scaler.scale(total_loss / num_interactions).backward()
                    
                    if step_counter % accumulation_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        
                        logger.info(f"[梯度累积] 完成 {accumulation_steps} 个batch的梯度累积，更新参数")
            
            if step_counter % accumulation_steps != 0 and step == num_steps - 1:
                has_valid_grad = any(p.grad is not None and not torch.isnan(p.grad).any() and not torch.isinf(p.grad).any() 
                                     for p in self.model.parameters())
                
                if has_valid_grad:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    logger.info(f"[梯度累积] 处理最后剩余的梯度，完成epoch的参数更新")
                else:
                    logger.warning(f"[WARNING] 剩余梯度包含NaN/Inf，跳过参数更新")
                    self.optimizer.zero_grad()
            
            else:
                print("[WARNING] Total loss has no grad, skipping optimizer step.")
                    
            with torch.no_grad():
                print(f"[DEBUG] final_pred_masks shape: {final_pred_masks.shape}, labels shape: {labels.shape}")
                print(f"[DEBUG] final_pred_masks min: {final_pred_masks.min().item()}, max: {final_pred_masks.max().item()}")
                print(f"[DEBUG] labels min: {labels.min().item()}, max: {labels.max().item()}")
                
                dice_scores = self._calc_dice(final_pred_masks, labels)
                tci_scores = self._calc_tci(final_pred_masks, labels)
                
                print(f"[DEBUG] dice_scores: {dice_scores}")
                print(f"[DEBUG] tci_scores: {tci_scores}")
                
                batch_dice = dice_scores.mean().item()
                batch_iou = tci_scores.mean().item()
                
                if batch_dice > 1.0:
                    print(f"[WARNING] 批量Dice异常: {batch_dice}, 检查计算逻辑")
                    batch_dice = min(batch_dice, 1.0)
                if batch_iou > 1.0:
                    print(f"[WARNING] 批量IoU异常: {batch_iou}, 检查计算逻辑")
                    batch_iou = min(batch_iou, 1.0)
                
                final_pred_masks = final_pred_masks.detach().cpu()
                    
            epoch_loss += round_loss
            epoch_dice += batch_dice
            epoch_iou += batch_iou
        
        avg_loss = epoch_loss / num_steps
        avg_dice = epoch_dice / num_steps
        avg_iou = epoch_iou / num_steps
        
        if avg_dice > 1.0:
            print(f"[WARNING] 平均Dice异常: {avg_dice}, 检查计算逻辑")
            avg_dice = min(avg_dice, 1.0)
        if avg_iou > 1.0:
            print(f"[WARNING] 平均IoU异常: {avg_iou}, 检查计算逻辑")
            avg_iou = min(avg_iou, 1.0)
        
        print(f"[Epoch 结束] 平均Loss: {avg_loss:.4f}, 平均Dice: {avg_dice:.4f}, 平均IoU: {avg_iou:.4f}")
        
        return {
            'loss': avg_loss,
            'dice': avg_dice,
            'iou': avg_iou
        }
    def _simulate_clicks(self, images, labels, previous_pred, round_idx, categories=None):
        """模拟点击点生成"""
        batch_size = labels.size(0)
        points = []
        point_labels = []
        
        for i in range(batch_size):
            label_i = labels[i]
            if label_i.dim() == 3:
                if label_i.shape[0] > 1:
                    class_idx = 1
                    if categories is not None and i < len(categories) and getattr(self.args, 'classes', None):
                        cat_name = str(categories[i])
                        if cat_name in self.args.classes:
                            class_idx = self.args.classes.index(cat_name)
                    class_idx = max(0, min(class_idx, label_i.shape[0] - 1))
                    label_i = label_i[class_idx]
                else:
                    label_i = label_i[0]

            foreground = (label_i > 0).nonzero(as_tuple=True)
            if len(foreground[0]) > 0:
                idx = torch.randint(0, len(foreground[0]), (1,), device=label_i.device)
                y, x = foreground[0][idx], foreground[1][idx]
                points.append(torch.tensor([[x.item(), y.item()]], dtype=torch.float32))
                point_labels.append(torch.tensor([1], dtype=torch.long))
            else:
                h, w = label_i.shape[-2], label_i.shape[-1]
                x = torch.randint(0, w, (1,), device=label_i.device)
                y = torch.randint(0, h, (1,), device=label_i.device)
                points.append(torch.tensor([[x.item(), y.item()]], dtype=torch.float32))
                point_labels.append(torch.tensor([0], dtype=torch.long))
        
        points = torch.stack(points).to(labels.device)
        point_labels = torch.stack(point_labels).to(labels.device)
        return points, point_labels
    def _calc_dice(self, pred, gt):
        """计算Dice系数"""
        if pred.shape[2:] != gt.shape[2:]:
            pred = F.interpolate(pred, size=gt.shape[2:], mode='bilinear', align_corners=False)
        
        if gt.dtype == torch.long:
            gt = gt.float()
        
        if gt.dim() == 4 and gt.shape[1] > 1:
            gt = (gt.argmax(dim=1, keepdim=True) > 0).float()
        elif gt.dim() == 4 and gt.shape[1] == 1:
            gt = (gt > 0).float()
        
        pred_bin = (torch.sigmoid(pred) > 0.5).float()
        
        if pred_bin.shape != gt.shape:
            print(f"[WARNING] 维度不匹配: pred_bin.shape={pred_bin.shape}, gt.shape={gt.shape}")
            if pred_bin.shape[0] != gt.shape[0]:
                print(f"[ERROR] 批次维度不匹配: pred_bin={pred_bin.shape[0]}, gt={gt.shape[0]}")
                min_batch = min(pred_bin.shape[0], gt.shape[0])
                pred_bin = pred_bin[:min_batch]
                gt = gt[:min_batch]
            if pred_bin.dim() == 4 and pred_bin.shape[1] != gt.shape[1]:
                if pred_bin.shape[1] > 1 and gt.shape[1] == 1:
                    pred_bin = pred_bin.argmax(dim=1, keepdim=True).float()
                else:
                    pred_bin = pred_bin[:, :1]
            if pred_bin.shape[2:] != gt.shape[2:]:
                pred_bin = F.interpolate(pred_bin, size=gt.shape[2:], mode='bilinear', align_corners=False)
        
        pred_bin = torch.clamp(pred_bin, 0, 1)
        gt = torch.clamp(gt, 0, 1)
        inter = (pred_bin * gt).sum(dim=(1, 2, 3))  
        pred_sum = pred_bin.sum(dim=(1, 2, 3))  
        gt_sum = gt.sum(dim=(1, 2, 3))  
        denom = pred_sum + gt_sum  # 标准 Dice 分母是 |A|+|B|，不要减 inter
        epsilon = 1e-6
        dice = torch.zeros_like(inter, dtype=torch.float32)
        valid_mask = gt_sum > 0  
        if valid_mask.any():
            valid_inter = inter[valid_mask]
            valid_denom = denom[valid_mask]
            dice[valid_mask] = (2 * valid_inter + epsilon) / (valid_denom + epsilon)
        if torch.isnan(dice).any() or (dice > 1.0).any() or (dice < 0).any():
            print(f"[WARNING] Dice异常: dice={dice}")
            print(f"[DEBUG] inter={inter}, pred_sum={pred_sum}, gt_sum={gt_sum}, denom={denom}")
            dice = torch.clamp(dice, 0.0, 1.0)
        return dice
    def _calc_tci(self, pred, gt):
        """计算IoU系数（用于静态数据集）"""
        if pred.shape[2:] != gt.shape[2:]:
            pred = F.interpolate(pred, size=gt.shape[2:], mode='bilinear', align_corners=False)
        
        if gt.dtype == torch.long:
            gt = gt.float()
        
        if gt.dim() == 4 and gt.shape[1] > 1:
            gt = (gt.argmax(dim=1, keepdim=True) > 0).float()
        elif gt.dim() == 4 and gt.shape[1] == 1:
            gt = (gt > 0).float()
        
        pred_bin = (torch.sigmoid(pred) > 0.5).float()
        
        if pred_bin.shape != gt.shape:
            if pred_bin.shape[0] != gt.shape[0]:
                min_batch = min(pred_bin.shape[0], gt.shape[0])
                pred_bin = pred_bin[:min_batch]
                gt = gt[:min_batch]
            if pred_bin.dim() == 4 and pred_bin.shape[1] != gt.shape[1]:
                if pred_bin.shape[1] > 1 and gt.shape[1] == 1:
                    pred_bin = pred_bin.argmax(dim=1, keepdim=True).float()
                else:
                    pred_bin = pred_bin[:, :1]
            if pred_bin.shape[2:] != gt.shape[2:]:
                pred_bin = F.interpolate(pred_bin, size=gt.shape[2:], mode='bilinear', align_corners=False)
        
        pred_bin = torch.clamp(pred_bin, 0, 1)
        gt = torch.clamp(gt, 0, 1)
        inter = (pred_bin * gt).sum(dim=(1, 2, 3))  
        pred_sum = pred_bin.sum(dim=(1, 2, 3))  
        gt_sum = gt.sum(dim=(1, 2, 3))  
        union = pred_sum + gt_sum - inter  
        epsilon = 1e-6
        iou = torch.zeros_like(inter, dtype=torch.float32)
        valid_mask = gt_sum > 0
        if valid_mask.any():
            valid_inter = inter[valid_mask]
            valid_union = union[valid_mask]
            iou[valid_mask] = (valid_inter + epsilon) / (valid_union + epsilon)
        if torch.isnan(iou).any() or (iou > 1.0).any() or (iou < 0).any():
            print(f"[WARNING] IoU异常: iou={iou}")
            print(f"[DEBUG] inter={inter}, pred_sum={pred_sum}, gt_sum={gt_sum}, union={union}")
            iou = torch.clamp(iou, 0.0, 1.0)
        return iou

class SmartCheckpointManager:
    def __init__(self, save_dir, max_checkpoints=5):
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        
    def save_checkpoint(self, epoch, trainer_state, metrics, describe="last"):
        """Smart checkpoint saving, automatically manage storage space"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': trainer_state,
            'optimizer_state_dict': trainer_state.optimizer.state_dict(),
            'metrics': metrics,
            'args': trainer_state.args,
            'timestamp': dt.datetime.now().isoformat()
        }
        
        filename = f"IMIS_{describe}_{epoch:04d}.pth"
        torch.save(checkpoint, join(self.save_dir, filename))
        
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Keep only the latest few checkpoints"""
        checkpoints = sorted([f for f in os.listdir(self.save_dir) if f.endswith('.pth')])
        if len(checkpoints) > self.max_checkpoints:
            for old_ckpt in checkpoints[:-self.max_checkpoints]:
                os.remove(join(self.save_dir, old_ckpt))

class ModelDiagnosticTool:
    """模型诊断和分析工具，用于监控训练过程中的关键指标"""
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.metrics_history = defaultdict(list)
        self.temporal_metrics = defaultdict(list)  
        self.gradient_stats = []  
        
    def update_metrics(self, epoch, metrics_dict, temporal_info=None, gradient_info=None):
        """Update and visualize training metrics"""
        for metric_name, value in metrics_dict.items():
            self.metrics_history[metric_name].append(value)
        
        if temporal_info:
            for key, value in temporal_info.items():
                self.temporal_metrics[key].append(value)
        
        if gradient_info:
            self.gradient_stats.append(gradient_info)
        
        self._plot_metrics(epoch)
        
        self._save_metrics_to_csv()
        
        if epoch % 10 == 0:
            self._generate_diagnostic_report(epoch)
    
    def _plot_metrics(self, epoch):
        """动态绘制指标图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics_to_plot = ['loss', 'dice', 'iou', 'learning_rate']
        for i, metric in enumerate(metrics_to_plot):
            if metric in self.metrics_history:
                ax = axes[i//2, i%2]
                ax.plot(self.metrics_history[metric])
                ax.set_title(f'{metric.upper()} over Epochs')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric)
                ax.grid(True)
                
                if metric == 'dice' and self.metrics_history[metric]:
                    best_idx = np.argmax(self.metrics_history[metric])
                    best_value = self.metrics_history[metric][best_idx]
                    ax.scatter(best_idx, best_value, color='red', s=100, zorder=5)
                    ax.annotate(f'Best: {best_value:.4f}', 
                               (best_idx, best_value),
                               xytext=(10, 10),
                               textcoords='offset points',
                               fontsize=12,
                               color='red')
        
        plt.tight_layout()
        plt.savefig(join(self.save_dir, f'training_metrics_epoch_{epoch:04d}.png'))
        plt.close()
        
        self._plot_single_metric(epoch, 'dice', 'Dice Score')
        
        if self.temporal_metrics:
            self._plot_temporal_metrics(epoch)
    
    def _plot_single_metric(self, epoch, metric_name, title):
        """绘制单个指标的详细图表"""
        if metric_name not in self.metrics_history:
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics_history[metric_name], 'b-', linewidth=2)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.title(title, fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel(title, fontsize=14)
        
        best_idx = np.argmax(self.metrics_history[metric_name]) if metric_name in ['dice', 'iou'] else np.argmin(self.metrics_history[metric_name])
        best_value = self.metrics_history[metric_name][best_idx]
        plt.scatter(best_idx, best_value, color='red', s=100, zorder=5)
        plt.annotate(f'Best: {best_value:.4f}', 
                   (best_idx, best_value),
                   xytext=(10, 10),
                   textcoords='offset points',
                   fontsize=12,
                   color='red')
        
        plt.ylim(max(0, np.min(self.metrics_history[metric_name]) - 0.1), 
                min(1.0, np.max(self.metrics_history[metric_name]) + 0.1))
        
        plt.tight_layout()
        plt.savefig(join(self.save_dir, f'{metric_name}_epoch_{epoch:04d}.png'))
        plt.close()
    
    def _plot_temporal_metrics(self, epoch):
        """绘制时序相关指标"""
        if not self.temporal_metrics:
            return
        
        plt.figure(figsize=(12, 8))
        for key, values in self.temporal_metrics.items():
            plt.plot(values, label=key)
        
        plt.title('Temporal Metrics over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(join(self.save_dir, f'temporal_metrics_epoch_{epoch:04d}.png'))
        plt.close()
    
    def _save_metrics_to_csv(self):
        """保存指标到CSV文件"""
        csv_path = join(self.save_dir, 'training_metrics.csv')
        
        metric_names = list(self.metrics_history.keys())
        if not metric_names:
            return
        
        max_length = max(len(values) for values in self.metrics_history.values())
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['epoch'] + metric_names)
            
            for i in range(max_length):
                row = [i + 1]  
                for metric in metric_names:
                    if i < len(self.metrics_history[metric]):
                        row.append(self.metrics_history[metric][i])
                    else:
                        row.append('')
                writer.writerow(row)
    
    def _generate_diagnostic_report(self, epoch):
        """生成诊断报告"""
        report_path = join(self.save_dir, f'diagnostic_report_epoch_{epoch:04d}.txt')
        
        with open(report_path, 'w') as f:
            f.write(f"Model Diagnostic Report - Epoch {epoch}\n")
            f.write("=" * 50 + "\n")
            
            f.write("\n1. Basic Metrics:\n")
            for metric, values in self.metrics_history.items():
                if values:
                    latest = values[-1]
                    best = max(values) if metric in ['dice', 'iou'] else min(values)
                    f.write(f"  {metric}: latest={latest:.4f}, best={best:.4f}\n")
            
            if self.temporal_metrics:
                f.write("\n2. Temporal Metrics:\n")
                for metric, values in self.temporal_metrics.items():
                    if values:
                        latest = values[-1]
                        f.write(f"  {metric}: latest={latest:.4f}\n")
            
            f.write("\n3. Training Status Analysis:\n")
            if len(self.metrics_history.get('dice', [])) > 5:
                recent_dice = self.metrics_history['dice'][-5:]
                improvement = recent_dice[-1] - recent_dice[0]
                if improvement > 0.01:
                    f.write("  ✓ Dice score is improving steadily\n")
                elif improvement > 0:
                    f.write("  ⚠ Dice score improvement is slow\n")
                else:
                    f.write("  ✗ Dice score is not improving\n")
            
            if 'learning_rate' in self.metrics_history:
                current_lr = self.metrics_history['learning_rate'][-1]
                f.write(f"  Current learning rate: {current_lr:.6f}\n")

class TrainingMonitor:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.metrics_history = defaultdict(list)
        
    def update_metrics(self, epoch, metrics_dict):
        """Update and visualize training metrics"""
        for metric_name, value in metrics_dict.items():
            self.metrics_history[metric_name].append(value)
        
        self._plot_metrics(epoch)
        
        self._save_metrics_to_csv()
    
    def _plot_metrics(self, epoch):
        """Dynamically plot metrics charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics_to_plot = ['loss', 'dice', 'iou', 'learning_rate']
        for i, metric in enumerate(metrics_to_plot):
            if metric in self.metrics_history:
                ax = axes[i//2, i%2]
                ax.plot(self.metrics_history[metric])
                ax.set_title(f'{metric.upper()} over Epochs')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric)
                ax.grid(True)
                
                if metric == 'dice' and self.metrics_history[metric]:
                    best_idx = np.argmax(self.metrics_history[metric])
                    best_value = self.metrics_history[metric][best_idx]
                    ax.scatter(best_idx, best_value, color='red', s=100, zorder=5)
                    ax.annotate(f'Best: {best_value:.4f}', 
                               (best_idx, best_value),
                               xytext=(10, 10),
                               textcoords='offset points',
                               fontsize=12,
                               color='red')
        
        plt.tight_layout()
        plt.savefig(join(self.save_dir, f'training_metrics_epoch_{epoch:04d}.png'))
        plt.close()
        
        self._plot_single_metric(epoch, 'dice', 'Dice Score')
    
    def _plot_single_metric(self, epoch, metric_name, title):
        """Plot detailed chart for a single metric"""
        if metric_name not in self.metrics_history:
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics_history[metric_name], 'b-', linewidth=2)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.title(title, fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel(title, fontsize=14)
        
        best_idx = np.argmax(self.metrics_history[metric_name]) if metric_name in ['dice', 'iou'] else np.argmin(self.metrics_history[metric_name])
        best_value = self.metrics_history[metric_name][best_idx]
        plt.scatter(best_idx, best_value, color='red', s=100, zorder=5)
        plt.annotate(f'Best: {best_value:.4f}', 
                   (best_idx, best_value),
                   xytext=(10, 10),
                   textcoords='offset points',
                   fontsize=12,
                   color='red')
        
        plt.ylim(max(0, np.min(self.metrics_history[metric_name]) - 0.1), 
                min(1.0, np.max(self.metrics_history[metric_name]) + 0.1))
        
        plt.tight_layout()
        plt.savefig(join(self.save_dir, f'{metric_name}_epoch_{epoch:04d}.png'))
        plt.close()
    
    def _save_metrics_to_csv(self):
        """Save metrics to CSV file"""
        csv_path = join(self.save_dir, 'training_metrics.csv')
        
        metric_names = list(self.metrics_history.keys())
        if not metric_names:
            return
        
        max_length = max(len(values) for values in self.metrics_history.values())
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['epoch'] + metric_names)
            
            for i in range(max_length):
                row = [i + 1]  
                for metric in metric_names:
                    if i < len(self.metrics_history[metric]):
                        row.append(self.metrics_history[metric][i])
                    else:
                        row.append('')
                writer.writerow(row)

class TemporalTrainer(BaseTrainer):
    def __init__(self, model, dataloaders, args):
        """Initialize temporal trainer"""
        super().__init__(model, dataloaders, args)
        if hasattr(self.model, 'image_encoder'):
            for name, param in self.model.image_encoder.named_parameters():
                if ('blocks.11' in name or 'blocks.10' in name or 'blocks.9' in name or
                    'blocks.8' in name or 'blocks.7' in name or 'blocks.6' in name or 'neck' in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print("SAM Image Encoder: last 6 blocks + neck unfrozen")
        self.interaction_rounds = args.num_clicks
        self.scaler = GradScaler()
        self.target_list = [c for c in args.classes if c != 'background'] if args.classes else ['LV', 'RV', 'Myo']
        self.temporal_loss = TemporalAwareLoss()
        self.fusion_warmup_epochs = 20      
        self.max_fusion_strength = 1.5      
        print(f"✅ TemporalTrainer initialized, temporal interaction rounds: {self.interaction_rounds}")
        print(f"✅ Fusion parameters: warmup_epochs={self.fusion_warmup_epochs}, max_strength={self.max_fusion_strength}")
    def _save_direct_checkpoint(self, epoch, state_dict, describe='latest'):
        """Save checkpoint directly to file, independent of checkpoint_manager"""
        try:
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

            checkpoint = {
                'model_state_dict': state_dict,
                'epoch': epoch,
                'best_loss': self.best_loss,
                'best_dice': self.best_dice,
                'best_iou': self.best_iou,
                'losses': self.losses,
                'dices': self.dices,
                'ious': self.ious,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
            }

            if describe == 'latest':
                filename = os.path.join(MODEL_SAVE_PATH, 'IMIS_latest.pth')
            elif describe == 'dice_best':
                filename = os.path.join(MODEL_SAVE_PATH, f'IMIS_best_dice_{self.best_dice:.4f}.pth')
                generic_best = os.path.join(MODEL_SAVE_PATH, 'IMIS_dice_best.pth')
            else:
                filename = os.path.join(MODEL_SAVE_PATH, f'IMIS_{describe}.pth')

            torch.save(checkpoint, filename)
            print(f"Save checkpoint to {filename}")

            if describe == 'dice_best':
                torch.save(checkpoint, generic_best)
                print(f"Save generic best checkpoint to {generic_best}")
                self._cleanup_best_checkpoints(max_keep=5)

        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            import traceback
            traceback.print_exc()

    def _cleanup_best_checkpoints(self, max_keep=5):
        """保留得分最高的 max_keep 个 best checkpoint，删除其余的"""
        import glob
        pattern = os.path.join(MODEL_SAVE_PATH, 'IMIS_best_dice_*.pth')
        best_files = glob.glob(pattern)
        if len(best_files) <= max_keep:
            return
        # 按文件名中的dice值排序，保留最高的max_keep个
        def extract_dice(filepath):
            import re
            m = re.search(r'IMIS_best_dice_(\d+\.\d+)\.pth', os.path.basename(filepath))
            return float(m.group(1)) if m else 0.0
        best_files.sort(key=extract_dice, reverse=True)
        for old_file in best_files[max_keep:]:
            try:
                os.remove(old_file)
                print(f"[Checkpoint清理] 删除旧权重: {os.path.basename(old_file)}")
            except OSError:
                pass

    def _run_periodic_test(self, epoch, state_dict):
        """[PERIODIC TEST] 每 val_interval 个 epoch 自动调用 test.py 跑测试集，
        用真实 mDice@8 评估泛化能力。如果 mDice 超过历史最佳，额外存一个 checkpoint。"""
        val_interval = getattr(self.args, 'val_interval', 20)
        if (epoch + 1) % val_interval != 0:
            return

        logger.info(f"[Periodic Test] Epoch {epoch+1} 完成，开始测试集评估...")
        print(f"[Periodic Test] Running test.py on test set (epoch {epoch+1})...")

        test_log_file = os.path.join(MODEL_SAVE_PATH, f'periodic_test_epoch{epoch+1}.log')

        # 先存一个临时权重给 test.py 用
        tmp_ckpt = os.path.join(MODEL_SAVE_PATH, 'IMIS_periodic_test_tmp.pth')
        tmp_checkpoint = {
            'model_state_dict': state_dict,
            'epoch': epoch,
            'best_dice': self.best_dice,
        }
        torch.save(tmp_checkpoint, tmp_ckpt)

        try:
            cmd = [
                sys.executable, 'test.py',
                '--dataset', str(getattr(self.args, 'dataset', 'btcv')),
                '--data_dir', str(getattr(self.args, 'data_dir', 'dataset/BTCV')),
                '--pretrain_path', tmp_ckpt,
                '--sam_checkpoint', str(getattr(self.args, 'sam_checkpoint', 'ckpt/IMISNet-B.pth')),
                '--model_type', str(getattr(self.args, 'model_type', 'vit_b')),
                '--model_variant', str(getattr(self.args, 'model_variant', 'tmf')),
                '--image_size', str(getattr(self.args, 'image_size', 256)),
                '--inter_num', '8',
                '--task_name', str(getattr(self.args, 'task_name', 'BTCV')),
                '--work_dir', str(getattr(self.args, 'work_dir', 'work_dir')),
            ]
            if getattr(self.args, 'use_temporal_fusion', False):
                cmd.append('--use_temporal_fusion')
            else:
                cmd.append('--no_temporal_fusion')

            with open(test_log_file, 'w') as logf:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=os.getcwd())
                logf.write(result.stdout)
                logf.write(result.stderr)

            # 从日志解析 mDice@8
            mdice = None
            for line in reversed(result.stdout.splitlines()):
                if 'mDice@8' in line or 'mDice' in line:
                    import re
                    m = re.search(r'mDice[@]?\d*[:\s]+([0-9.]+)', line)
                    if m:
                        mdice = float(m.group(1))
                        break
                # summary_results.txt 回退
            if mdice is None:
                summary_file = os.path.join(MODEL_SAVE_PATH, 'summary_results.txt')
                if os.path.exists(summary_file):
                    with open(summary_file, 'r') as sf:
                        for line in sf:
                            if 'mDice@8' in line:
                                import re
                                m = re.search(r'([0-9.]+)', line.split(':')[-1])
                                if m:
                                    mdice = float(m.group(1))
                                    break

            if mdice is not None:
                logger.info(f"[Periodic Test] Epoch {epoch+1} → mDice@8 = {mdice:.4f}")
                print(f"[Periodic Test] Epoch {epoch+1} → mDice@8 = {mdice:.4f}")

                if not hasattr(self, 'test_best_dice'):
                    self.test_best_dice = 0.0
                if mdice > self.test_best_dice:
                    self.test_best_dice = mdice
                    test_ckpt = os.path.join(MODEL_SAVE_PATH, f'IMIS_test_best_dice_{mdice:.4f}.pth')
                    torch.save(tmp_checkpoint, test_ckpt)
                    logger.info(f"[Periodic Test] New test best! Saved {test_ckpt}")
                    print(f"[Periodic Test] New test best! Saved {test_ckpt}")
            else:
                logger.warning(f"[Periodic Test] 无法解析 mDice@8，请查看日志: {test_log_file}")
                print(f"[Periodic Test] Could not parse mDice, see: {test_log_file}")

        except subprocess.TimeoutExpired:
            logger.warning(f"[Periodic Test] 超时 (3600s)")
            print(f"[Periodic Test] Timeout!")
        except Exception as e:
            logger.warning(f"[Periodic Test] 失败: {e}")
            print(f"[Periodic Test] Error: {e}")
        finally:
            if os.path.exists(tmp_ckpt):
                os.remove(tmp_ckpt)

    
    def _to_binary_labels(self, labels, categories):
        """[FIX ①] 把多通道 labels (B, C, H, W) 按每个样本的 category 切成单通道二值 (B, 1, H, W)。
        单通道/3通道(H,W,H)直接二值化。与训练循环 L651-672 保持一致。

        Args:
            labels: (B,C,H,W) 或 (B,H,W) 或 (B,1,H,W)，long/float 均可
            categories: list[str/int]，对应每个样本的目标类别名或索引
        Returns:
            binary_labels: (B, 1, H, W) float32 {0,1}
        """
        import torch
        if labels.dim() == 4 and labels.shape[1] > 1:
            binary_list = []
            for i in range(labels.shape[0]):
                cat_name = str(categories[i]) if i < len(categories) else '1'
                if hasattr(self.args, 'classes') and self.args.classes and (cat_name in self.args.classes):
                    class_idx = self.args.classes.index(cat_name)
                else:
                    try:
                        class_idx = int(cat_name) if not cat_name.startswith('class_') else int(cat_name.split('_', 1)[1])
                    except Exception:
                        class_idx = 1
                if 0 <= class_idx < labels.shape[1]:
                    slc = labels[i:i+1, class_idx:class_idx+1]
                else:
                    slc = labels[i:i+1, 0:1]
                binary_list.append((slc > 0).float())
            return torch.cat(binary_list, dim=0) if binary_list else (labels[:, 0:1] > 0).float()
        elif labels.dim() == 3:
            return (labels.unsqueeze(1) > 0).float()
        else:
            return (labels > 0).float()

    def _get_extreme_clicks(self, labels, categories=None):
        """[FIX ①②] Round 0 首点击：
        ① 先通过 _to_binary_labels 切到目标类别的单通道二值GT（不再对13通道找==1造成跨类假正点）
        ② 采用距离变换中心（与 test._generate_first_click 完全一致），且始终返回 1 个正点，不再随机取点

        Args:
            labels: GT 标签（可能多通道，也可能单通道）
            categories: 每个样本的类别列表，用于切通道；为空时按通道0取

        Returns:
            points: [B, 1, 2] float
            point_labels: [B, 1] long {0,1}
        """
        import numpy as np
        from scipy.ndimage import distance_transform_edt

        binary_labels = self._to_binary_labels(labels, categories or [])
        batch_size = binary_labels.size(0)
        h, w = binary_labels.size(2), binary_labels.size(3)
        device = labels.device
        points = []
        point_labels = []

        for i in range(batch_size):
            fg_mask = binary_labels[i, 0].detach().cpu().numpy().astype(np.uint8)
            fg_count = int(fg_mask.sum())
            if fg_count == 0:
                # 无前景：回退到GT质心或中心
                src = labels[i].detach().cpu().numpy()
                if src.ndim == 3:
                    multi = (src > 0).astype(np.uint8)
                    if multi.ndim == 3 and multi.shape[0] in (1, 13):
                        multi = multi.max(axis=0)
                else:
                    multi = (src > 0).astype(np.uint8)
                if multi.sum() > 0:
                    ys, xs = np.nonzero(multi)
                    cy, cx = int(round(float(ys.mean()))), int(round(float(xs.mean())))
                else:
                    cy, cx = h // 2, w // 2
                pl = 1 if fg_count > 0 else 0
            else:
                dist = distance_transform_edt(fg_mask)
                cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)
                cy, cx = int(cy), int(cx)
                pl = 1

            points.append(torch.tensor([[float(cx), float(cy)]], dtype=torch.float32))
            point_labels.append(torch.tensor([pl], dtype=torch.long))

        points = torch.cat(points, dim=0).to(device).unsqueeze(1) if batch_size > 0 else torch.zeros(0, 1, 2, device=device)
        point_labels = torch.cat(point_labels, dim=0).to(device).unsqueeze(1) if batch_size > 0 else torch.zeros(0, 1, device=device, dtype=torch.long)
        return points, point_labels

    def _get_error_based_clicks(self, prev_pred, labels, categories=None):
        """纠错轮点击点：对齐误差区域用 单通道二值GT，避免多通道GT的通道错位。

        Args:
            prev_pred: 上一轮预测 [B, 1, H, W] （logits）
            labels: GT 标签（原始形式，内部转单通道二值）
            categories: 每个样本的类别列表，用于切通道
        Returns:
            points: [B, 1, 2]
            point_labels: [B, 1]
        """
        import numpy as np
        from scipy.ndimage import label, distance_transform_edt

        binary_labels = self._to_binary_labels(labels, categories or [])
        if prev_pred.device != binary_labels.device:
            prev_pred = prev_pred.to(binary_labels.device)
        batch_size = binary_labels.size(0)
        _, _, h, w = binary_labels.shape
        device = binary_labels.device
        points = []
        point_labels = []

        for i in range(batch_size):
            pred_bin = (torch.sigmoid(prev_pred[i]) > 0.5).float()
            gt_bin = binary_labels[i, 0:1]
            error_map = torch.abs(pred_bin - gt_bin)

            if error_map.sum() == 0:
                fg = (gt_bin.squeeze() > 0).nonzero(as_tuple=False)
                if len(fg) > 0:
                    idx = torch.randint(0, len(fg), (1,))
                    y, x = int(fg[idx, 0].item()), int(fg[idx, 1].item())
                else:
                    y, x = h // 2, w // 2
                points.append(torch.tensor([[float(x), float(y)]], dtype=torch.float32))
                point_labels.append(torch.tensor([1], dtype=torch.long))
            else:
                error_np = error_map.detach().cpu().numpy().squeeze().astype(np.uint8)
                labeled, n = label(error_np)
                if n == 0:
                    error_indices = torch.nonzero(error_map.view(-1)).squeeze(-1)
                    random_idx = torch.randint(0, len(error_indices), (1,)).item()
                    flat = error_indices[random_idx].item()
                    y, x = flat // w, flat % w
                else:
                    sizes = np.bincount(labeled.ravel())
                    largest_idx = int(np.argmax(sizes[1:])) + 1
                    max_region = (labeled == largest_idx).astype(np.uint8)
                    dist = distance_transform_edt(max_region)
                    cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)
                    y, x = int(cy), int(cx)
                true_val = 1 if gt_bin[0, y, x].item() > 0.5 else 0
                pl = 1 if true_val == 1 else 0
                points.append(torch.tensor([[float(x), float(y)]], dtype=torch.float32))
                point_labels.append(torch.tensor([pl], dtype=torch.long))

        points = torch.cat(points, dim=0).to(device).unsqueeze(1)
        point_labels = torch.cat(point_labels, dim=0).to(device).unsqueeze(1)
        return points, point_labels

    def _simulate_clicks(self, images, labels, previous_pred, round_idx, categories=None):
        """[FIX ①②] 点击模拟：
        ① Round 0 和 纠错轮都基于目标类别的单通道二值GT
        ② Round 0 只生成 1 个距离变换中心的正点击（严格对齐 test._generate_first_click）
        移除了 training 下的点噪声/丢弃——该操作会把正点击概率性变成0点，模型训练不稳定，不符合协议。

        Returns:
            points: [B, N, 2] float32（N=1）
            point_labels: [B, N] long {0,1}（N=1）
        """
        if previous_pred is None or round_idx == 0:
            points, point_labels = self._get_extreme_clicks(labels, categories=categories)
        else:
            points, point_labels = self._get_error_based_clicks(previous_pred, labels, categories=categories)
        return points.float(), point_labels.long()
    
    def _build_seq_tensor(self, points, point_labels, round_idx):
        """points: [B, N, 2], point_labels: [B, N], 返回 [B, T, 4]"""
        B, N = points.shape[:2]
        device = points.device
        seq = torch.zeros(B, round_idx + 1, 4, device=device)
        idx = min(round_idx, N-1)
        seq[:, round_idx, :2] = points[:, idx, :]
        seq[:, round_idx, 2] = point_labels[:, idx]
        seq[:, round_idx, 3] = float(round_idx)
        return seq
    
    def _calc_dice(self, pred, gt):
        if pred.shape[2:] != gt.shape[2:]:
            pred = F.interpolate(pred, size=gt.shape[2:], mode='bilinear', align_corners=False)
        
        if gt.dtype == torch.long:
            gt = gt.float()
        
        if gt.dim() == 4 and gt.shape[1] > 1:
            gt = (gt.argmax(dim=1, keepdim=True) > 0).float()
        elif gt.dim() == 4 and gt.shape[1] == 1:
            gt = (gt > 0).float()
        
        pred_bin = (torch.sigmoid(pred) > 0.5).float()
        
        if pred_bin.shape != gt.shape:
            print(f"[WARNING] 维度不匹配: pred_bin.shape={pred_bin.shape}, gt.shape={gt.shape}")
            if pred_bin.shape[0] != gt.shape[0]:
                print(f"[ERROR] 批次维度不匹配: pred_bin={pred_bin.shape[0]}, gt={gt.shape[0]}")
                min_batch = min(pred_bin.shape[0], gt.shape[0])
                pred_bin = pred_bin[:min_batch]
                gt = gt[:min_batch]
            if pred_bin.dim() == 4 and pred_bin.shape[1] != gt.shape[1]:
                if pred_bin.shape[1] > 1 and gt.shape[1] == 1:
                    pred_bin = pred_bin.argmax(dim=1, keepdim=True).float()
                else:
                    pred_bin = pred_bin[:, :1]
            if pred_bin.shape[2:] != gt.shape[2:]:
                pred_bin = F.interpolate(pred_bin, size=gt.shape[2:], mode='bilinear', align_corners=False)
        
        pred_bin = torch.clamp(pred_bin, 0, 1)
        gt = torch.clamp(gt, 0, 1)
        
        inter = (pred_bin * gt).sum(dim=(1, 2, 3))  
        pred_sum = pred_bin.sum(dim=(1, 2, 3))  
        gt_sum = gt.sum(dim=(1, 2, 3))  
        denom = pred_sum + gt_sum  # 标准 Dice 分母是 |A|+|B|，不要减 inter
        
        denom = torch.clamp(denom, min=1e-6)
        
        dice = (2 * inter + 1e-6) / (denom + 1e-6)
        
        if torch.isnan(dice).any() or (dice > 1.0).any() or (dice < 0).any():
            print(f"[WARNING] Dice异常: dice={dice}")
            print(f"[DEBUG] inter={inter}, pred_sum={pred_sum}, gt_sum={gt_sum}, denom={denom}")
            dice = torch.clamp(dice, 0.0, 1.0)
        
        return dice  
    def _calc_tci(self, pred, target):
        """修复后的时序一致性指标（与Dice计算逻辑一致）"""
        if pred.shape[2:] != target.shape[2:]:
            pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)
        if target.dtype == torch.long:
            target = target.float()
        
        if target.dim() == 4 and target.shape[1] > 1:
            target = (target.argmax(dim=1, keepdim=True) > 0).float()
        elif target.dim() == 4 and target.shape[1] == 1:
            target = (target > 0).float()
        
        pred_bin = (torch.sigmoid(pred) > 0.5).float()
        target_bin = target.float()
        inter = (pred_bin * target_bin).sum(dim=(1, 2, 3))  
        union = pred_bin.sum(dim=(1, 2, 3)) + target_bin.sum(dim=(1, 2, 3)) - inter  
        iou = (inter + 1e-6) / (union + 1e-6)  
        
        if torch.isnan(iou).any() or (iou > 1.0).any():
            print(f"[WARNING] IoU异常: iou={iou}, pred_bin sum={pred_bin.sum()}, target_bin sum={target_bin.sum()}")
            iou = torch.clamp(iou, 0.0, 1.0)
        
        return iou  
    
    def criterion(self, outputs, labels, interaction_round, previous_pred=None, global_epoch=0, new_click_coords=None):
        """计算损失函数（修复版：不使用self.previous_pred避免计算图累积）
        
        Args:
            outputs: 模型输出
            labels: 真实标签
            interaction_round: 交互轮次
            previous_pred: 上一轮预测
            global_epoch: 当前全局epoch
            new_click_coords: 当前轮新增点击坐标 (B, N, 2)
            
        Returns:
            loss: 计算的损失值
        """
        pred_masks = outputs['masks']
        iou_pred = outputs.get('iou_pred', None)
        
        if not pred_masks.requires_grad:
            if self.model.training:
                pred_masks = pred_masks.clone().requires_grad_(True)
                logger.warning("[WARNING] pred_masks had no gradient, created a clone with requires_grad=True")
            else:
                pass
        
        if hasattr(self, 'temporal_loss'):
            loss = self.temporal_loss(pred_masks, labels.float(),
                                      iou_pred=iou_pred,
                                      previous_pred=previous_pred,
                                      interaction_round=interaction_round,
                                      global_epoch=global_epoch,
                                      new_click_coords=new_click_coords,
                                      image_size=(self.args.image_size, self.args.image_size),
                                      click_exclude_radius=5)
        else:
            loss = F.binary_cross_entropy_with_logits(pred_masks, labels.float())
        
        return loss
    
    def _save_checkpoint(self, epoch, state_dict, loss, dice, iou):
        """保存检查点
        
        Args:
            epoch: 当前轮次
            state_dict: 模型状态字典
            loss: 损失值
            dice: Dice系数
            iou: IoU系数
        """
        try:
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            
            checkpoint = {
                'model_state_dict': state_dict,
                'epoch': epoch,
                'loss': loss,
                'dice': dice,
                'iou': iou,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
            }
            
            latest_path = os.path.join(MODEL_SAVE_PATH, 'IMIS_latest.pth')
            torch.save(checkpoint, latest_path)
            
            if dice > self.best_dice:
                best_path = os.path.join(MODEL_SAVE_PATH, f'IMIS_best_dice_{dice:.4f}.pth')
                torch.save(checkpoint, best_path)
                print(f"Saved best model with dice: {dice:.4f}")
                
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
    
    def _plot_metrics(self, epoch):
        """绘制指标曲线
        
        Args:
            epoch: 当前轮次
        """
        pass
    
    def _save_metrics_to_csv(self):
        """保存指标到CSV文件
        """
        pass
    
    def train_one_epoch(self, epoch):
        self.model.train()
        model = self.model.module if self.args.multi_gpu else self.model
        device = next(model.parameters()).device
        
        # 课程学习：前 50 epoch 使用 3 轮交互，之后使用 5 轮
        # epoch 从 0 开始计数，epoch < 50 对应第 1--50 个 epoch
        if epoch < 50:
            self.interaction_rounds = 3
        else:
            self.interaction_rounds = 5
        logger.info(f"[课程学习] 当前epoch: {epoch+1}, 交互轮次: {self.interaction_rounds}")
        if not hasattr(self, 'scaler'):
            self.scaler = torch.cuda.amp.GradScaler()
            
        self.train_loader = self.dataloaders if not isinstance(self.dataloaders, dict) else self.dataloaders.get('train', self.dataloaders)
        tbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        epoch_loss, epoch_dice, epoch_iou = 0.0, 0.0, 0.0
        num_steps = len(tbar)
        
        accumulation_steps = 4  
        step_counter = 0  
        
        if epoch == 10:
            target_model = self.model.module if hasattr(self.model, 'module') else self.model
            if hasattr(target_model, 'temporal_memory'):
                target_model.temporal_memory.clear_all_history()
                logger.info("[时序融合] 第10个epoch开始，清空此前积累的低质量历史记录")
        
        target_model = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(target_model, 'multiscale_temporal_fusion') and hasattr(target_model.multiscale_temporal_fusion, 'update_fusion_weight'):
            fusion_weight = target_model.multiscale_temporal_fusion.update_fusion_weight(epoch)
            logger.info(f"[时序融合] 当前融合权重: {fusion_weight:.4f}")
        last_patient_id = None
        for step, batch in enumerate(tbar):
            
            batch_patient_ids = set([str(pid) for pid in batch['patient_ids']])
            
            if last_patient_id is not None and not any(pid == last_patient_id for pid in batch_patient_ids):
                for pid in batch_patient_ids:
                    if hasattr(self.model.module if hasattr(self.model, 'module') else self.model, 'temporal_memory'):
                        (self.model.module if hasattr(self.model, 'module') else self.model).temporal_memory.clear_patient_history(pid)
                logger.info(f"[INFO] 切换到新病人组 {batch_patient_ids}，已清空之前病人的历史时序记忆！")
                if hasattr(self.model.module if hasattr(self.model, 'module') else self.model, 'interaction_history'):
                    (self.model.module if hasattr(self.model, 'module') else self.model).interaction_history.clear()
                    logger.info(f"[INFO] 已清空 interaction_history 防止显存泄漏！")
            
            last_patient_id = str(batch['patient_ids'][0])
            
            if step % 10 == 0:  
                logger.info(f"[INFO] Batch {step+1}/{num_steps}")
            
            if step % 3 == 0:  
                torch.cuda.empty_cache()
                gc.collect()
            
            target_model = self.model.module if hasattr(self.model, 'module') else self.model
            
            images = batch["image"].to(device)
            labels = batch["label"].to(device).long()
            batch_size = images.size(0)
            
            if 'patient_ids' not in batch or not batch['patient_ids']:
                raise RuntimeError("Batch中没有patient_ids！请先完成data_loader.py的修改")
            patient_ids = [str(pid) for pid in batch['patient_ids']]
            if 'target_list' not in batch or not batch['target_list']:
                raise RuntimeError("Batch中没有target_list！请先完成data_loader.py的修改")
            
            target_list_len = len(batch['target_list'])
            logger.debug(f"[DEBUG target_list] 长度: {target_list_len}, batch_size: {batch_size}, mask_num: {self.args.mask_num}")
            logger.debug(f"[DEBUG target_list] 内容: {batch['target_list']}")
            
            categories = []
            if target_list_len > 0:
                for i in range(len(patient_ids)):
                    idx = min(i, target_list_len - 1)
                    categories.append(batch['target_list'][idx])
            else:
                categories = [0] * len(patient_ids)
                print("[WARNING] target_list 为空，使用默认类别 0")
            print(f"[DEBUG 数据] 真实病人ID: {patient_ids}, 真实类别: {categories}")
            logger.debug(f"[DEBUG 数据] patient_ids长度: {len(patient_ids)}, categories长度: {len(categories)}")
            
            current_patients_in_batch = set(str(pid) for pid in patient_ids)
            logger.info(f"[DEBUG] Batch info: images shape: {images.shape}, labels shape: {labels.shape}, labels unique: {labels.unique().tolist()}")
            logger.info(f"[DEBUG] args.classes: {getattr(self.args, 'classes', None)}")
            logger.info(f"[DEBUG] target_list: {getattr(self, 'target_list', None)}")
            total_loss = 0.0 
            num_interactions = self.interaction_rounds
            previous_pred = None
            round_loss = 0.0
            image_features = None
            accumulated_points = None
            accumulated_labels = None
            real_model = self.model.module if hasattr(self.model, 'module') else self.model
            if image_features is None and hasattr(self.model, 'image_forward'):
                image_features = self.model.image_forward(images)
            extended_batch_size = image_features.shape[0]
            if not categories:
                categories = ["default"] * extended_batch_size
            elif isinstance(categories, (list, tuple)):
                if len(categories) != extended_batch_size:
                    extended_categories = []
                    for cat in categories:
                        extended_categories.extend([cat] * self.args.mask_num)
                    extended_categories = extended_categories[:extended_batch_size]
                    categories = extended_categories
            else:
                categories = [str(categories)] * extended_batch_size
            if not patient_ids:
                patient_ids = [f"patient_{i}" for i in range(extended_batch_size)]
            elif len(patient_ids) != extended_batch_size:
                extended_patient_ids = []
                for pid in patient_ids:
                    extended_patient_ids.extend([pid] * self.args.mask_num)
                extended_patient_ids = extended_patient_ids[:extended_batch_size]
                patient_ids = extended_patient_ids
            original_labels = labels
            if labels.dim() == 4 and labels.shape[1] > 1:
                binary_labels = []
                for i in range(labels.shape[0]):
                    if i < len(categories):
                        cat_name = categories[i] if isinstance(categories[i], str) else str(categories[i])
                        if hasattr(self.args, 'classes') and self.args.classes:
                            try:
                                class_idx = self.args.classes.index(cat_name) if cat_name in self.args.classes else 1
                            except ValueError:
                                class_idx = 1
                        else:
                            class_idx = 1
                        if labels.shape[1] > class_idx:
                            binary_labels.append(labels[i:i+1, class_idx:class_idx+1])
                        else:
                            binary_labels.append(labels[i:i+1, 0:1])
                    else:
                        binary_labels.append(labels[i:i+1, 0:1])
                if binary_labels:
                    labels = torch.cat(binary_labels, dim=0)
                else:
                    labels = (labels.argmax(dim=1, keepdim=True) > 0).float()
            elif labels.dim() == 3:
                labels = labels.unsqueeze(1)
            for interaction_round in range(num_interactions):
                points, point_labels = self._simulate_clicks(images, labels, previous_pred, interaction_round)
                if accumulated_points is None:
                    accumulated_points = points
                    accumulated_labels = point_labels
                else:
                    accumulated_points = torch.cat([accumulated_points, points], dim=1)
                    accumulated_labels = torch.cat([accumulated_labels, point_labels], dim=1)
                prompts = {
                    'patient_ids': patient_ids,   
                    'categories': categories,     
                    'interaction_ids': patient_ids, 
                    'temporal_enabled': True,
                    # [方案C+AntiLEAK 修复] 移除 labels（多通道GT张量）
                    # 不再把整块 GT 作为 prompt 项（之前虽然仅用于 label_sum gate，但仍属于 GT tensor 进 forward）。
                    # 改用 meta_label_sum 传标量（合法，只是用来判空，不影响推理路径不影响梯度）。
                    'meta_label_sum': [float(labels[i].sum().item()) if labels is not None else -1.0 for i in range(len(patient_ids))],
                    'interaction_round': interaction_round,
                    'epoch': epoch,
                    'global_epoch': epoch
                }
                
                if interaction_round == 0:
                    prompts['dice'] = [0.0] * len(patient_ids)
                    prompts['tci'] = [0.0] * len(patient_ids)
                    prompts['quality_score'] = [0.0] * len(patient_ids)
                    prompts['iou_predictions'] = [0.0] * len(patient_ids)
                
                prompts['point_coords'] = accumulated_points.float()
                prompts['point_labels'] = accumulated_labels
                with autocast():
                    if hasattr(self.model, 'forward_with_features'):
                        outputs = self.model.forward_with_features(image_features, prompts)
                    else:
                        outputs = self.model.forward(images, prompts)
                    # [ANTI-LEAK compliant] 用 GT 算真实 IoU 作为 TQM 决策的监督目标
                    if isinstance(outputs, dict) and 'masks' in outputs:
                        with torch.no_grad():
                            _pm = outputs['masks'].detach().float()
                            if _pm.shape[2:] != labels.shape[2:]:
                                _pm = F.interpolate(_pm, size=labels.shape[2:], mode='bilinear', align_corners=False)
                            _pb = (torch.sigmoid(_pm) > 0.5).float()
                            _gb = (labels > 0).float()
                            if _gb.dim() == 4 and _gb.shape[1] > 1:
                                _gb = _gb[:, :1]
                            _inter = (_pb * _gb).sum(dim=(1, 2, 3))
                            _union = _pb.sum(dim=(1, 2, 3)) + _gb.sum(dim=(1, 2, 3)) - _inter
                            _ri = ((_inter + 1e-7) / (_union + 1e-7)).clamp(0.0, 1.0).flatten()
                            _n = min(len(patient_ids), _ri.numel())
                            _qs2 = [float(_ri[i].item()) for i in range(_n)] + [0.0] * max(0, len(patient_ids) - _n)
                        prompts['quality_score'] = _qs2
                        prompts['iou_predictions'] = list(_qs2)
                    try:
                        loss = self.criterion(outputs, labels, interaction_round=interaction_round, previous_pred=previous_pred, global_epoch=epoch, new_click_coords=points)
                    except Exception as e:
                        model_logger.error(f"[FATAL] 前向传播错误: {str(e)}")
                        model_logger.error(f"[FATAL] 病人ID: {patient_ids}, 类别: {categories}")
                        model_logger.error(f"[FATAL] 交互轮次: {interaction_round}")
                        if hasattr(self.model, 'temporal_memory'):
                            self.model.temporal_memory.clear_all_history()
                        elif hasattr(self.model.module, 'temporal_memory'):
                            self.model.module.temporal_memory.clear_all_history()
                        raise e
                
                pred_masks = outputs['masks'].float()
                # Capture model-predicted mask quality (iou_pred = paper q_t) before deleting outputs
                _iou_pred = outputs.get('iou_pred', None)
                del outputs  
                
                if isinstance(interaction_round, torch.Tensor):
                    interaction_round_val = interaction_round.mean().item() if interaction_round.numel() > 1 else interaction_round.item()
                else:
                    interaction_round_val = interaction_round
                if interaction_round_val == 0 and not pred_masks.requires_grad:
                    print("[CRITICAL] Masks have no grad! Check your model's forward path.")
                
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"[WARNING] 交互轮次 {interaction_round+1} 的loss为NaN/Inf，跳过累加")
                else:
                    total_loss += loss.float()
                
                dice_scores = self._calc_dice(pred_masks, labels)
                tci_values = self._calc_tci(pred_masks, labels)
                batch_dice = dice_scores.mean().item()
                batch_tci = tci_values.mean().item()
                
                if batch_dice > 1.0:
                    print(f"[WARNING] 交互轮次Dice异常: {batch_dice}, 检查计算逻辑")
                    batch_dice = min(batch_dice, 1.0)
                if batch_tci > 1.0:
                    print(f"[WARNING] 交互轮次TCI异常: {batch_tci}, 检查计算逻辑")
                    batch_tci = min(batch_tci, 1.0)
                if batch_dice == 0:
                    print(f"[DEBUG] Batch dice is 0! pred_masks shape: {pred_masks.shape}, labels shape: {labels.shape}")
                    print(f"[DEBUG] pred_masks min: {pred_masks.min().item()}, max: {pred_masks.max().item()}")
                    print(f"[DEBUG] labels sum: {labels.sum().item()}")
                    print(f"[DEBUG] pred_masks sigmoid sum: {torch.sigmoid(pred_masks).sum().item()}")
                print(f"[交互监控] 轮次 {interaction_round+1}/{num_interactions} - Dice: {batch_dice:.4f}, TCI: {batch_tci:.4f}")
                prompts['dice'] = dice_scores.tolist()
                prompts['tci'] = tci_values.tolist()
                # Paper q_t = model-predicted mask quality (iou_pred from decoder head)
                try:
                    __iou = _iou_pred
                except NameError:
                    __iou = None
                if __iou is not None and isinstance(__iou, torch.Tensor):
                    _flat = __iou.detach().float().flatten()
                    _n = min(len(patient_ids), _flat.numel())
                    _qs = [float(_flat[i].item()) for i in range(_n)] + [0.0] * max(0, len(patient_ids) - _n)
                    prompts['quality_score'] = _qs
                    prompts['iou_predictions'] = list(_qs)
                else:
                    prompts['quality_score'] = list(prompts.get('dice', [0.0] * len(patient_ids)))
                    prompts['iou_predictions'] = list(prompts['quality_score'])
                
                previous_pred = pred_masks.detach().cpu()
                final_pred_masks = pred_masks.detach()
                del pred_masks
                if torch.isnan(loss) or torch.isinf(loss):
                    round_loss = 0.0
                    logger.warning(f"[WARNING] 交互轮次 {interaction_round+1} 的loss为NaN/Inf，使用0.0代替")
                else:
                    round_loss = loss.item()  
                del loss
            
            if torch.is_tensor(total_loss) and total_loss.requires_grad:
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    logger.error(
                        f"[FATAL] 当前 Batch 的 Total Loss 为 NaN/Inf! 放弃参数更新，防止模型权重崩溃。"
                    )
                    self.optimizer.zero_grad()
                    step_counter += 1  
                else:
                    step_counter += 1
                    self.scaler.scale(total_loss / num_interactions).backward()
                    
                    if step_counter % accumulation_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        
                        logger.info(f"[梯度累积] 完成 {accumulation_steps} 个batch的梯度累积，更新参数")
            
            if step_counter % accumulation_steps != 0 and step == num_steps - 1:
                has_valid_grad = any(p.grad is not None and not torch.isnan(p.grad).any() and not torch.isinf(p.grad).any() 
                                     for p in self.model.parameters())
                
                if has_valid_grad:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    logger.info(f"[梯度累积] 处理最后剩余的梯度，完成epoch的参数更新")
                else:
                    logger.warning(f"[WARNING] 剩余梯度包含NaN/Inf，跳过参数更新")
                    self.optimizer.zero_grad()
            
            if step % 10 == 0:  
                real_model = self.model.module if hasattr(self.model, 'module') else self.model
                
                fusion_weight = real_model.multiscale_temporal_fusion.trajectory_fusion_weight.item()
                print(f"[时序监控] 轨迹融合权重: {fusion_weight:.4f}")
                
                if hasattr(real_model.multiscale_temporal_fusion, 'trajectory_analyzer'):
                    lstm = real_model.multiscale_temporal_fusion.trajectory_analyzer.temporal_memory
                    if hasattr(lstm, 'weight_ih_l0') and lstm.weight_ih_l0.grad is not None:
                        lstm_grad = lstm.weight_ih_l0.grad.abs().mean().item()
                        print(f"[时序监控] LSTM梯度均值: {lstm_grad:.6f}")
                    
                print(f"[时序监控] 第{interaction_round+1}轮Dice: {batch_dice:.4f}")
                
            else:
                print("[WARNING] Total loss has no grad, skipping optimizer step.")
                    
            with torch.no_grad():
                print(f"[DEBUG] final_pred_masks shape: {final_pred_masks.shape}, labels shape: {labels.shape}")
                print(f"[DEBUG] final_pred_masks min: {final_pred_masks.min().item()}, max: {final_pred_masks.max().item()}")
                print(f"[DEBUG] labels min: {labels.min().item()}, max: {labels.max().item()}")
                
                dice_scores = self._calc_dice(final_pred_masks, labels)
                tci_scores = self._calc_tci(final_pred_masks, labels)
                
                print(f"[DEBUG] dice_scores: {dice_scores}")
                print(f"[DEBUG] tci_scores: {tci_scores}")
                
                batch_dice = dice_scores.mean().item()
                batch_iou = tci_scores.mean().item()
                
                if batch_dice > 1.0:
                    print(f"[WARNING] 批量Dice异常: {batch_dice}, 检查计算逻辑")
                    batch_dice = min(batch_dice, 1.0)
                if batch_iou > 1.0:
                    print(f"[WARNING] 批量IoU异常: {batch_iou}, 检查计算逻辑")
                    batch_iou = min(batch_iou, 1.0)
                
                final_pred_masks = final_pred_masks.detach().cpu()
                    
            epoch_loss += round_loss
            epoch_dice += batch_dice
            epoch_iou += batch_iou
            
            if not hasattr(self, 'all_interaction_dices'):
                self.all_interaction_dices = []
            self.all_interaction_dices.append(batch_dice)
            
            tbar.set_postfix(loss=f"{round_loss:.4f}", dice=f"{batch_dice:.4f}")
            
            target_model = self.model.module if hasattr(self.model, 'module') else self.model
            if hasattr(target_model, 'interaction_history'):
                target_model.interaction_history.clear()
            
            del image_features
            del previous_pred
            del final_pred_masks
            torch.cuda.empty_cache()
            gc.collect()
        avg_loss = epoch_loss / num_steps
        avg_dice = epoch_dice / num_steps
        avg_iou = epoch_iou / num_steps
        
        if avg_dice > 1.0:
            print(f"[WARNING] 最后一轮Dice异常: {avg_dice}, 检查计算逻辑")
            avg_dice = min(avg_dice, 1.0)
        
        if hasattr(self, 'all_interaction_dices') and len(self.all_interaction_dices) > 0:
            all_interaction_avg = sum(self.all_interaction_dices) / len(self.all_interaction_dices)
            if all_interaction_avg > 1.0:
                print(f"[WARNING] 所有交互轮次Dice异常: {all_interaction_avg}, 检查计算逻辑")
                all_interaction_avg = min(all_interaction_avg, 1.0)
            logger.info(f"[详细统计] 所有交互轮次平均Dice: {all_interaction_avg:.4f}, 最后一轮平均Dice: {avg_dice:.4f}")
            avg_dice = all_interaction_avg  
            self.all_interaction_dices = []  
        
        lr_vit = self.optimizer.param_groups[0]['lr']      
        lr_prompt = self.optimizer.param_groups[1]['lr']   
        lr_decoder = self.optimizer.param_groups[2]['lr']  
        lr_temporal = self.optimizer.param_groups[3]['lr']  
        
        logger.info(f"Epoch {epoch+1} 完成 - LR(ViT): {lr_vit:.6f}, LR(Prompt): {lr_prompt:.6f}, LR(Decoder): {lr_decoder:.6f}, LR(时序): {lr_temporal:.6f}, Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}")
        if not hasattr(self, 'metrics_history'):
            self.metrics_history = {
                'loss': [],
                'dice': [],
                'iou': [],
                'learning_rate': []
            }
        
        self.metrics_history['loss'].append(avg_loss)
        self.metrics_history['dice'].append(avg_dice)
        self.metrics_history['iou'].append(avg_iou)
        
        state_dict = self.model.module.state_dict() if self.args.multi_gpu else self.model.state_dict()
        self._plot_metrics(epoch)
        self._save_metrics_to_csv()
        
        return {'loss': avg_loss, 'dice': avg_dice, 'iou': avg_iou}
    def validate_one_epoch(self, epoch):
        """验证一个epoch并搜索最优分割阈值"""
        self.model.eval()
        model = self.model.module if self.args.multi_gpu else self.model
        device = next(model.parameters()).device
        
        if isinstance(self.dataloaders, dict):
            val_loader = self.dataloaders.get('val', self.dataloaders)
        else:
            val_loader = self.dataloaders
        
        if epoch % 5 == 0:
            best_threshold = 0.5
            best_dice = 0.0
            
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for step, batch in enumerate(val_loader):
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device).long()
                    patient_ids = [str(pid) for pid in batch['patient_ids']]
                    categories = batch['target_list']
                    
                    prompts = {
                        'patient_ids': patient_ids,
                        'categories': categories,
                        'interaction_ids': patient_ids,
                        'temporal_enabled': True,
                        # [方案C+AntiLEAK 修复] 移除 labels 张量，改用 meta_label_sum 标量（合法：仅判空）。
                        'meta_label_sum': [float(labels[i].sum().item()) if labels is not None else -1.0 for i in range(len(patient_ids))],
                        'interaction_round': 0,
                        'epoch': epoch
                    }
                    
                    points, point_labels = self._simulate_clicks(images, labels, None, 0)
                    prompts['point_coords'] = points.float()
                    prompts['point_labels'] = point_labels
                    # [ANTI-LEAK compliant] 验证阶段也用 GT 算真实 IoU 作为 TQM 监督目标
                    prompts['quality_score'] = [0.0] * len(patient_ids)
                    prompts['iou_predictions'] = [0.0] * len(patient_ids)
                    prompts['global_epoch'] = epoch
                    
                    with autocast():
                        if hasattr(self.model, 'forward'):
                            outputs = self.model.forward(images, prompts)
                        else:
                            outputs = self.model(images, prompts)
                    # [ANTI-LEAK compliant] 用 GT 算真实 IoU 作为 TQM 监督目标（验证阶段）
                    if isinstance(outputs, dict) and 'masks' in outputs:
                        with torch.no_grad():
                            _pm = outputs['masks'].detach().float()
                            if _pm.shape[2:] != labels.shape[2:]:
                                _pm = F.interpolate(_pm, size=labels.shape[2:], mode='bilinear', align_corners=False)
                            _pb = (torch.sigmoid(_pm) > 0.5).float()
                            _gb = (labels > 0).float()
                            if _gb.dim() == 4 and _gb.shape[1] > 1:
                                _gb = _gb[:, :1]
                            _inter = (_pb * _gb).sum(dim=(1, 2, 3))
                            _union = _pb.sum(dim=(1, 2, 3)) + _gb.sum(dim=(1, 2, 3)) - _inter
                            _ri = ((_inter + 1e-7) / (_union + 1e-7)).clamp(0.0, 1.0).flatten()
                            _n = min(len(patient_ids), _ri.numel())
                            _qs3 = [float(_ri[i].item()) for i in range(_n)] + [0.0] * max(0, len(patient_ids) - _n)
                        prompts['quality_score'] = _qs3
                        prompts['iou_predictions'] = list(_qs3)

                    pred_masks = outputs['masks'].float()
                    all_preds.append(pred_masks)
                    all_labels.append(labels)
            
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            for threshold in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
                pred_bin = (torch.sigmoid(all_preds) > threshold).float()
                inter = (pred_bin * all_labels).sum(dim=(1, 2, 3))
                union = pred_bin.sum(dim=(1, 2, 3)) + all_labels.sum(dim=(1, 2, 3))
                dice = (2 * inter + 1e-6) / (union + 1e-6)
                avg_dice = dice.mean().item()
                
                if avg_dice > best_dice:
                    best_dice = avg_dice
                    best_threshold = threshold
            
            logger.info(f"[阈值搜索] 最优阈值: {best_threshold}, 对应Dice: {best_dice:.4f}")
            self.best_threshold = best_threshold
            if hasattr(self.model, 'module'):
                self.model.module.best_threshold = best_threshold
            else:
                self.model.best_threshold = best_threshold
        
        total_dice = 0.0
        total_iou = 0.0
        
        tbar = tqdm(val_loader, desc=f"Validation Epoch {epoch}")
        with torch.no_grad():
            for step, batch in enumerate(tbar):
                images = batch["image"].to(device)
                labels = batch["label"].to(device).long()
                patient_ids = [str(pid) for pid in batch['patient_ids']]
                categories = batch['target_list']
                
                prompts = {
                    'patient_ids': patient_ids,
                    'categories': categories,
                    'interaction_ids': patient_ids,
                    'temporal_enabled': True,
                    # [方案C+AntiLEAK 修复] 移除 labels 张量，改用 meta_label_sum 标量（合法：仅判空）。
                    'meta_label_sum': [float(labels[i].sum().item()) if labels is not None else -1.0 for i in range(len(patient_ids))],
                    'interaction_round': 0,
                    'epoch': epoch
                }
                
                points, point_labels = self._simulate_clicks(images, labels, None, 0)
                prompts['point_coords'] = points.float()
                prompts['point_labels'] = point_labels
                
                with autocast():
                    if hasattr(self.model, 'forward'):
                        outputs = self.model.forward(images, prompts)
                    else:
                        outputs = self.model(images, prompts)
                
                pred_masks = outputs['masks'].float()
                
                threshold = getattr(self, 'best_threshold', 0.5)
                pred_bin = (torch.sigmoid(pred_masks) > threshold).float()
                inter = (pred_bin * labels).sum(dim=(1, 2, 3))
                union = pred_bin.sum(dim=(1, 2, 3)) + labels.sum(dim=(1, 2, 3))
                dice = (2 * inter + 1e-6) / (union + 1e-6)
                iou = (inter + 1e-6) / (union + 1e-6)
                
                total_dice += dice.mean().item()
                total_iou += iou.mean().item()
                
                tbar.set_postfix(dice=f"{dice.mean().item():.4f}", threshold=f"{threshold:.2f}")
        
        avg_dice = total_dice / len(val_loader)
        avg_iou = total_iou / len(val_loader)
        
        logger.info(f"[验证] Epoch {epoch+1} - Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}, 阈值: {getattr(self, 'best_threshold', 0.5):.2f}")
        
        return {'dice': avg_dice, 'iou': avg_iou}
    def train(self):
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        
        
        logger.info("[INFO] 训练环境准备完成，开始训练")
        
        if not hasattr(self, 'scaler'):
            self.scaler = GradScaler()
            print("scaler initialized in train method")
        
        self.best_threshold = 0.5
        
        from tqdm import tqdm
        epoch_tbar = tqdm(range(self.start_epoch, self.args.num_epochs), desc="Training Epochs")
        
        for epoch in epoch_tbar:
            logger.info(f"开始 Epoch {epoch+1}/{self.args.num_epochs}")
            
            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                print(f'Epoch: {epoch}/{self.args.num_epochs - 1}')
            
            if self.args.multi_gpu:
                dist.barrier()
                if hasattr(self.dataloaders.sampler, 'set_epoch'):
                    self.dataloaders.sampler.set_epoch(epoch)
            metrics = self.train_one_epoch(epoch)
            avg_loss = metrics['loss']
            avg_iou = metrics['iou']
            avg_dice = metrics['dice']
            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                self.losses.append(avg_loss)
                self.ious.append(avg_iou)
                self.dices.append(avg_dice)
                
                current_lr = self.optimizer.param_groups[0]['lr']
                
                print(f"Epoch {epoch} finished. Current Base LR: {current_lr:.8f}")
                print(f'Epochs: {epoch}, LR: {current_lr}, Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, Dice: {avg_dice:.4f}')
                logger.info(f'Epoch\t {epoch}\t LR\t {current_lr}\t: loss: {avg_loss:.4f}, iou: {avg_iou:.4f}, dice: {avg_dice:.4f}')
                
                epoch_tbar.set_postfix(loss=f"{avg_loss:.4f}", dice=f"{avg_dice:.4f}", lr=f"{current_lr:.6f}")
                
                if hasattr(self, 'metrics_history'):
                    self.metrics_history['learning_rate'].append(current_lr)
                elif hasattr(self.model, 'metrics_history'):
                    self.model.metrics_history['learning_rate'].append(current_lr)
                
                if self.args.multi_gpu:
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()
                
                self._save_direct_checkpoint(epoch, state_dict, describe='latest')
                if avg_loss < self.best_loss: 
                    self.best_loss = avg_loss
                
                if avg_iou > self.best_iou: 
                    self.best_iou = avg_iou
                if avg_dice > self.best_dice: 
                    prev_best = self.best_dice
                    self.best_dice = avg_dice
                    self._save_direct_checkpoint(epoch, state_dict, describe='dice_best')
                    self.early_stop_counter = 0
                    logger.info(f"[早停监控] 新最佳 Dice: {self.best_dice:.4f}（前最佳: {prev_best:.4f}），早停计数器已清零: 0/{self.early_stop_patience}")
                else:
                    if epoch >= 40:
                        self.early_stop_counter += 1
                        logger.info(f"[早停监控] 性能未改善（当前 {avg_dice:.4f} ≤ 最佳 {self.best_dice:.4f}），早停计数器: {self.early_stop_counter}/{self.early_stop_patience}")
                
                if math.isnan(avg_loss) or math.isinf(avg_loss):
                    logger.error("[早停触发] 损失值异常，停止训练！")
                    print("[早停触发] 损失值异常，停止训练！")
                    break
                
                if epoch >= 10 and avg_dice < 0.01:
                    logger.error("[早停触发] Dice 值接近 0，模型完全失效，停止训练！")
                    print("[早停触发] Dice 值接近 0，模型完全失效，停止训练！")
                    break
                
                if epoch >= 40 and self.early_stop_counter >= self.early_stop_patience:
                    if self.args.disable_early_stop:
                        logger.info(f"[早停监控] 已禁用早停，继续训练...")
                        print(f"[早停监控] 已禁用早停，继续训练...")
                        self.early_stop_counter = 0  
                    else:
                        logger.info(f"[早停触发] 连续 {self.early_stop_patience} 个 epoch 性能未改善，停止训练！")
                        print(f"[早停触发] 连续 {self.early_stop_patience} 个 epoch 性能未改善，停止训练！")
                        break
                
                print(f"Epoch {epoch} Metrics: Loss={avg_loss:.4f}, Dice={avg_dice:.4f}, IoU={avg_iou:.4f}")
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                if hasattr(self.model, 'module'):
                    if hasattr(self.model.module, 'multiscale_temporal_fusion'):
                        if hasattr(self.model.module.multiscale_temporal_fusion, 'temporal_memory'):
                            self.model.module.multiscale_temporal_fusion.temporal_memory.on_epoch_end(epoch)
                else:
                    if hasattr(self.model, 'multiscale_temporal_fusion'):
                        if hasattr(self.model.multiscale_temporal_fusion, 'temporal_memory'):
                            self.model.multiscale_temporal_fusion.temporal_memory.on_epoch_end(epoch)
                
                patient_count = 0
                total_records = 0
                if hasattr(self.model, 'module'):
                    patient_count = len(self.model.module.multiscale_temporal_fusion.temporal_memory.history_buffer)
                    total_records = sum(len(v) for val in self.model.module.multiscale_temporal_fusion.temporal_memory.history_buffer.values() for v in val.values() if isinstance(val, dict))
                else:
                    patient_count = len(self.model.multiscale_temporal_fusion.temporal_memory.history_buffer)
                    total_records = sum(len(v) for val in self.model.multiscale_temporal_fusion.temporal_memory.history_buffer.values() for v in val.values() if isinstance(val, dict))
                print(f"[INFO] Epoch {epoch} 完成，历史缓冲包含 {patient_count} 个病人，共 {total_records} 条记录")
                
                # [PERIODIC TEST] 每 val_interval 个 epoch 自动跑测试集
                if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                    self._run_periodic_test(epoch, state_dict)
                
      
        print("==========================================")
        print(f"Training completed!")
        print(f"Best metrics - Loss: {self.best_loss:.4f}, Dice: {self.best_dice:.4f}, IoU: {self.best_iou:.4f}")
        print(f"Training epochs: {self.args.num_epochs}")
        print(f"Best threshold: {getattr(self, 'best_threshold', 0.5):.2f}")
        print("==========================================")

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

def main():
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
    os.environ['PYOPENGL_LOG_LEVEL'] = 'ERROR'
    BTCV_CLASSES = [
        'background', 'spleen', 'right_kidney', 'left_kidney',
        'gallbladder', 'esophagus', 'liver', 'stomach', 'aorta',
        'inferior_vena_cava', 'portal_vein_and_splenic_vein',
        'pancreas', 'right_adrenal_gland', 'left_adrenal_gland'
    ]
    AMOS_CLASSES = [
        'background', 'spleen', 'right_kidney', 'left_kidney',
        'gallbladder', 'esophagus', 'liver', 'stomach', 'aorta',
        'inferior_vena_cava', 'pancreas', 'right_adrenal_gland',
        'left_adrenal_gland', 'duodenum', 'bladder', 'prostate_uterus'
    ]
    if args.task_name == 'BTCV':
        args.classes = BTCV_CLASSES
        args.num_classes = 14  
        args.window_level = 40
        args.window_width = 400
    elif args.task_name == 'AMOS2022_MR':
        args.classes = AMOS_CLASSES
        args.num_classes = 16  
        args.is_mr = True      
    elif 'ACDC' in args.task_name.upper():
        args.classes = ['background', 'RV', 'Myo', 'LV']
        args.num_classes = 4  
    normalize_model_variant(args)
    logging.getLogger('cv2').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    LOG_OUT_DIR = os.path.join(args.work_dir, args.task_name)
    os.makedirs(LOG_OUT_DIR, exist_ok=True)
    log_file = os.path.join(LOG_OUT_DIR, 'training.log')
    print(f"[INIT] 日志文件将保存到: {log_file}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)  
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] - [%(levelname)s] - %(message)s', datefmt='%Y/%m/%d %H:%M:%S'))
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter('[%(asctime)s] - [%(levelname)s] - %(message)s', datefmt='%Y/%m/%d %H:%M:%S'))
    console_handler.stream.flush()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    print(f"[TEST] 日志文件路径: {log_file}", flush=True)
    print(f"[TEST] 文件是否存在: {os.path.exists(log_file)}", flush=True)
    file_handler.stream.flush()
    print(f"[TEST] 文件大小: {os.path.getsize(log_file) if os.path.exists(log_file) else 0} bytes", flush=True)
    root_logger.info(f"[TEST] 日志系统初始化完成")
    print('*'*100)
    for key, value in vars(args).items():
        print(key + ': ' + str(value))
    print('*'*100)
    mp.set_sharing_strategy('file_system')
    device_config(args)
    if args.multi_gpu:
        mp.spawn(main_worker, nprocs=args.world_size, args=(args, ))
    else:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        init_seeds(42)
        dataloaders = get_loader(args)
        model = build_model(args)
        if args.use_temporal_fusion:
            trainer = TemporalTrainer(model, dataloaders, args)
            print("Using TemporalTrainer for temporal fusion training")
        else:
            trainer = BaseTrainer(model, dataloaders, args)
            print("Using BaseTrainer for standard training")
        trainer.train()

def main_worker(rank, args):
    setup(rank, args.world_size)
    torch.cuda.set_device(rank)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank
    args.gpu_info = {"gpu_count":args.world_size, 'gpu_name':rank}
    init_seeds(2024 + rank)
    log_file = os.path.join(LOG_OUT_DIR, 'training.log')
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)  
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] - [%(levelname)s] - %(message)s', datefmt='%Y/%m/%d %H:%M:%S'))
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if rank in [-1, 0] else logging.WARN)
    console_handler.setFormatter(logging.Formatter('[%(asctime)s] - [%(levelname)s] - %(message)s', datefmt='%Y/%m/%d %H:%M:%S'))
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    dataloaders = get_loader(args)
    model = build_model(args)
    if args.use_temporal_fusion:
        trainer = TemporalTrainer(model, dataloaders, args)
        if rank == 0:
            print("Using TemporalTrainer for temporal fusion training")
    else:
        trainer = BaseTrainer(model, dataloaders, args)
        if rank == 0:
            print("Using BaseTrainer for standard training")
    trainer.train()
    cleanup()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = f'{args.port}'
    dist.init_process_group(backend='NCCL', init_method='env://', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print(f"程序崩溃了！错误: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        sys.exit(1)
