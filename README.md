# Identifying Cell Cycle Phase Using DNA Staining in Time-Lapse Microscopy

## Overview

This repository contains a deep learning pipeline for cell segmentation and cell cycle phase identification from single-channel microscopy images.

## Project Highlights

- Built a U-Net-based semantic segmentation pipeline for microscopy image analysis.
- Used EfficientNet-based encoders for cell segmentation and cell cycle phase classification.
- Included training, testing, inference, data augmentation, encoder comparison, and treatment analysis scripts.
- Supported post-processing, visualization, and time-lapse output generation.

## Main Files

- `train.py`: trains the U-Net segmentation model and saves the best checkpoint.
- `test.py`: evaluates the trained model and generates segmentation visualizations.
- `inference.py`: runs prediction on new `.tif` images and saves colored masks, plots, and timelapse videos.
- `inference.spec`: PyInstaller configuration for building a standalone executable.
- `requirements.txt`: lists all required Python packages.

## Utility Files

- `utils/augment_dataset.py`: creates augmented image-mask pairs.
- `utils/test_encoders.py`: compares U-Net encoders such as EfficientNet, ResNet, VGG, and Xception.
- `utils/optimize_model_hyperparam.py`: runs Optuna-based hyperparameter optimization.
- `utils/organize-train-data.py`: reorganizes training data by location and imaging zone.
- `utils/compare-treatments.py`: compares cell cycle phase changes under different drug treatments.
- `utils/*.ipynb`: experimental notebooks for augmentation and utility testing.

## Dataset Format

```text
dataset/
├── train/
│   ├── input/
│   └── gt/
├── valid/
│   ├── input/
│   └── gt/
└── test/
    ├── input/
    └── gt/
```

## Setup

### Create Conda Environment
To create a Conda environment with Python 3.10.13, follow these steps:

```bash
conda create --name myenv python=3.10.13
conda activate myenv
```

### Install Dependencies
After activating the environment, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Scripts

### 1. Training the Model
The `train.py` script is used to train the UNet model.

**Usage:**
```bash
python train.py
```

**Description:**
- This script trains a UNet model using the training data provided in the dataset directory. It includes data augmentation, logging with TensorBoard, and saving the best model based on validation dice score.

### 2. Testing the Model
The `test.py` script is used to test the trained UNet model.

**Usage:**
```bash
python test.py
```

**Description:**
- This script evaluates the model's performance on the test dataset. It computes various metrics, including IoU and dice scores, and visualizes the results.

### 3. Running Inference
The `inference.py` script is used to perform inference on new images using the trained model.

**Usage:**
```bash
python inference.py <input_folder> <model_path>
```

The inference results are saved to:

```text
<input_folder>_predicted/
```

**Description:**
- This script loads a trained model and performs segmentation on new images. It includes preprocessing, prediction, post-processing, and visualization of the results.

## Notes

- Make sure the dataset is organized with input images and ground truth masks in separate directories.
- Modify the paths in the scripts as needed to point to your dataset and model directories.
