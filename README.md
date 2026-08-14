# Checkpoints

We use the publicly released IMISNet-B checkpoint provided by the original IMISNet-B authors.

Original checkpoint link:

- Baidu Netdisk: https://pan.baidu.com/s/1eCuHs3qhd1lyVGqUOdaeFw?pwd=r1pg
- Password: `r1pg`

Please download the checkpoint from the original link and place the downloaded files under:

```text
ckpt/


# Dataset Splits

We use the preprocessed datasets released in IMed-361M on HuggingFace:

https://huggingface.co/datasets/General-Medical-AI/IMed-361M/tree/main

Specifically, ACDC, BTCV, and AMOS2022_MR are downloaded from the corresponding IMed-361M archives. We follow the train/test partitions provided in the `dataset.json` file of each dataset archive.

The data are organized following the IMed-361M format:

```text
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


# Train TMF-Net

To train TMF-Net, run:

```bash
python train.py
Main arguments:
work_dir: working directory for training outputs. Default: work_dir.
image_size: input image size. Default: 256.
data_path: dataset directory, for example dataset/BTCV.
sam_checkpoint: path to the downloaded IMISNet-B checkpoint under ckpt/.
inter_num: number of simulated interaction rounds used during training.
epochs: number of training epochs. Default: 150.
batch_size: training batch size. Default: 8.


# Evaluate TMF-Net
To evaluate TMF-Net, run:
python test.py
Main arguments:
test_mode: set to True for evaluation.
image_size: input image size. Default: 256.
data_path: dataset directory, for example dataset/BTCV.
checkpoint: path to the trained TMF-Net checkpoint.
inter_num: number of simulated interaction rounds. The paper reports results with K = 8.
prompt_mode: interaction mode. The paper uses point-based simulated corrective clicks.


# Evaluation Protocol
All reported results use the same 2D slice-level interactive protocol:
Interaction budget: K = 8
ACDC: NoC@90%-DSC
BTCV and AMOS2022_MR: NoC@85%-DSC
Spatial metrics: mDice@8, HD95, and ASSD are computed after 3D reconstruction from the final prediction at the 8th interaction.
Failed cases within the 8-click budget are capped at 8 clicks.
