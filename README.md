# TMF-Net

## Checkpoints

We use the publicly released IMISNet-B checkpoint provided by the original IMISNet-B authors.

- **Baidu Netdisk:** [Download Link](https://pan.baidu.com/s/1eCuHs3qhd1lyVGqUOdaeFw?pwd=r1pg)
- **Password:** `r1pg`

Please download the checkpoint and place the extracted files under the `ckpt/` directory:

```text
ckpt/
Dataset Splits
All experiments in the paper follow a 2D slice-level interactive segmentation protocol. Although ACDC, BTCV, and AMOS2022_MR originate from volumetric medical images, training and evaluation are performed on 2D slices. The reported DSC, HD95, ASSD, NoC, and AUC values are computed directly on 2D slice predictions and macro-averaged across valid target-containing slices. No 3D volume reconstruction is used for metric computation.
We use the preprocessed datasets released in IMed-361M on HuggingFace. Specifically, ACDC, BTCV, and AMOS2022_MR are downloaded from the corresponding IMed-361M archives. We follow the train/test partitions provided in the dataset.json file of each dataset archive.
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
Main arguments:
--work_dir: Working directory for training outputs. Default: work_dir.
--image_size: Input image size. Default: 256.
--data_path: Dataset directory, e.g., dataset/BTCV.
--sam_checkpoint: Path to the downloaded IMISNet-B checkpoint under ckpt/.
--inter_num: Number of simulated interaction rounds used during training.
--epochs: Number of training epochs. Default: 150.
--batch_size: Training batch size. Default: 8.


Evaluate TMF-Net
To evaluate TMF-Net, run:
python test.py
Main arguments:
--test_mode: Set to True for evaluation.
--image_size: Input image size. Default: 256.
--data_path: Dataset directory, e.g., dataset/BTCV.
--checkpoint: Path to the trained TMF-Net checkpoint.
--inter_num: Number of simulated interaction rounds. The paper reports results with K=8.
Evaluation follows the same 2D slice-level protocol as described in the paper. A valid slice contains at least one foreground pixel of the target structure. Metrics are computed on 2D slices without reconstructing 3D volumes. NoC is capped at 8 clicks, so both click-8 successes and within-budget failures are assigned a NoC value of 8.
