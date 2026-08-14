# Checkpoints

We use the publicly released IMISNet-B checkpoint provided by the original IMISNet-B authors.

Original checkpoint link:

- Baidu Netdisk: https://pan.baidu.com/s/1eCuHs3qhd1lyVGqUOdaeFw?pwd=r1pg
- Password: `r1pg`

Please download the checkpoint from the original link and place the downloaded files under:

```text
ckpt/


## Dataset Splits

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

