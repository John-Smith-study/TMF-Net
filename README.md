# TMF-Net

## Checkpoints

We use the publicly released IMISNet-B checkpoint provided by the original IMISNet-B authors.

- **Baidu Netdisk:** [Download Link](https://pan.baidu.com/s/1eCuHs3qhd1lyVGqUOdaeFw?pwd=r1pg)
- **Password:** `r1pg`

Please download the checkpoint and place the extracted files under the `ckpt/` directory:

```text
ckpt/
...

Dataset Splits
All experiments in the paper follow a 2D slice-level interactive segmentation protocol. Although ACDC, BTCV, and AMOS2022_MR originate from volumetric medical images, training and evaluation are performed on 2D slices. The reported DSC, HD95, ASSD, NoC, and AUC values are computed directly on 2D slice predictions and macro-averaged across valid target-containing slices. No 3D volume reconstruction is used for metric computation.
We use the preprocessed ACDC, BTCV, and AMOS2022_MR datasets released in IMed-361M on HuggingFace. The train/test splits are defined by the training and test entries in the dataset.json file included in each corresponding IMed-361M dataset archive.
The data should be organized following the IMed-361M format:

dataset/
├── ACDC/
│   ├── image/
│   ├── label/
│   ├── imask/
│   └── dataset.json
├── BTCV/
│   ├── image/
│   ├── label/
│   ├── imask/
│   └── dataset.json
└── AMOS2022_MR/
    ├── image/
    ├── label/
    ├── imask/
    └── dataset.json
Train TMF-Net
To train TMF-Net with the paper setting, run:
Bash

python train.py --data_dir dataset/BTCV --sam_checkpoint ckpt/IMISNet-B.pth --inter_num 5 --num_epochs 150 --batch_size 8 --lr_scheduler cosine
Main arguments:
• --work_dir: Working directory for training outputs. Default: work_dir.
• --image_size: Input image size. Default: 256.
• --data_dir: Dataset directory, e.g., dataset/BTCV.
• --sam_checkpoint: Path to the downloaded IMISNet-B checkpoint under ckpt/.
• --inter_num: Number of simulated interaction rounds used during training. The paper uses 5 for training.
• --num_epochs: Number of training epochs. The paper uses 150.
• --batch_size: Training batch size. The paper uses 8.
• --lr_scheduler: Learning-rate scheduler. Use cosine for the paper setting.
Evaluate TMF-Net
To evaluate TMF-Net with the paper setting, run:

python test.py --test_mode True --data_dir dataset/BTCV --sam_checkpoint ckpt/IMISNet-B.pth --pretrain_path work_dir/BTCV_traj/TMF_latest.pth --inter_num 8
Main arguments:
• --test_mode: Set to True for evaluation.
• --image_size: Input image size. Default: 256.
• --data_dir: Dataset directory, e.g., dataset/BTCV.
• --sam_checkpoint: Path to the downloaded IMISNet-B checkpoint under ckpt/.
• --pretrain_path: Path to the trained TMF-Net checkpoint.
• --inter_num: Number of simulated interaction rounds. The paper reports results with K=8.
Evaluation follows the same 2D slice-level protocol as described in the paper. A valid slice contains at least one foreground pixel of the target structure. Metrics are computed on 2D slices without reconstructing 3D volumes. Empty-mask slices are retained during preprocessing and inference but excluded from the reported metric denominators. NoC is capped at 8 clicks, so click-8 successes and within-budget failures both receive a NoC value of 8.
