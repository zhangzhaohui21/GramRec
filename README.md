# GramRec: Global Cross-modal Alignment with Alignment and Uniformity for Multimodal Recommendation

## Introduction

This is the Pytorch implementation for our GramRec paper:

>GramRec: Global Cross-modal Alignment with Alignment and Uniformity for Multimodal Recommendation

## Project Structure

```text
GramRec/
├── data/
│   ├── baby/
│   ├── clothing/
│   ├── sports/
│   └── README.md
├── src/
│   ├── common/                 # Recommender base classes and trainer
│   ├── configs/
│   │   ├── dataset/            # Dataset configurations
│   │   ├── model/              # Model and hyperparameter configurations
│   │   └── overall.yaml        # General training and evaluation settings
│   ├── models/
│   │   └── gramrec.py          # GramRec implementation
│   └── utils_package/          # Data loading, evaluation, logging, and utilities
├── main.py                     # Entry point
└── README.md
```

## Requirements

- Python 3.9
- Pytorch 2.1.0
- NumPy
- pandas

## Data Preparation

The project uses the processed Amazon Baby, Clothing, and Sports datasets. See [data/README.md](data/README.md) for the download link. After downloading the data, place it under `data/` using lowercase directory names that match the configuration files:

```text
data/
├── baby/
│   ├── baby.inter
│   ├── image_feat.npy
│   └── text_feat.npy
├── clothing/
│   ├── clothing.inter
│   ├── image_feat.npy
│   └── text_feat.npy
└── sports/
    ├── sports.inter
    ├── image_feat.npy
    └── text_feat.npy
```

## Training

```
cd ./src
python main.py --model GRAMREC --dataset sports
```

## Acknowledgements

This project uses the Amazon Review datasets for multimodal recommendation experiments. We thank the authors and maintainers of the related open-source .

The structure of this code is based on [MMRec](https://github.com/enoche/MMRec). Thank for their work.
