# TMF-Net

## Checkpoints

We use the publicly released IMISNet-B checkpoint provided by the original IMISNet-B authors.

- **Baidu Netdisk:** [Download Link](https://pan.baidu.com/s/1eCuHs3qhd1lyVGqUOdaeFw?pwd=r1pg)
- **Password:** `r1pg`

Please download the checkpoint and place the extracted files under the `ckpt/` directory:

```text
ckpt/

Dataset Splits
We use the preprocessed datasets released in IMed-361M on HuggingFace.

Specifically, ACDC, BTCV, and AMOS2022_MR are downloaded from the corresponding IMed-361M archives. We follow the train/test partitions provided in the dataset.json file of each dataset archive.

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
To train TMF-Net, run the following command:

python train.py

Main Arguments:
--work_dir: Working directory for training outputs (Default: work_dir).
--image_size: Input image size (Default: 256).
--data_path: Dataset directory (e.g., dataset/BTCV).
--sam_checkpoint: Path to the downloaded IMISNet-B checkpoint under ckpt/.
--inter_num: Number of simulated interaction rounds used during training.
--epochs: Number of training epochs (Default: 150).
--batch_size: Training batch size (Default: 8).

Evaluate TMF-Net
To evaluate TMF-Net, run:
python test.py
Main Arguments:
--test_mode: Set to True for evaluation.
--image_size: Input image size (Default: 256).
--data_path: Dataset directory (e.g., dataset/BTCV).
--checkpoint: Path to the trained TMF-Net checkpoint.
--inter_num: Number of simulated interaction rounds (The paper reports results with K = 8).
--prompt_mode: Interaction mode (The paper uses point-based simulated corrective clicks).
Evaluation Protocol
All reported results use the same 2D slice-level interactive protocol:
Interaction budget: K = 8
ACDC: NoC@90%-DSC
BTCV and AMOS2022_MR: NoC@85%-DSC
Spatial metrics: mDice@8, HD95, and ASSD are computed after 3D reconstruction from the final prediction at the 8th interaction.
Note: Failed cases within the 8-click budget are capped at 8 clicks.
