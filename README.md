# Matrix Multiplication Image Classification Model

A PyTorch implementation of matrix multiplication-free convolutional neural networks for CIFAR-10 image classification. This project replaces expensive matrix multiplications with addition/sustraction operations, exploring the model accuracy trade-off for potential computational efficiency when running the MMF on dedicated hardware.


---

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
  - [1. Clone Repository](#1-clone-repository)
  - [2. Install Required Packages](#2-install-required-packages)
  - [3. Download CIFAR-10 Dataset](#3-download-cifar-10-dataset)
- [Repository Structure](#repository-structure)
- [Citation](#citation)

---

## Overview

Deep learning models for computer vision are computationally dominated by floating-point matrix multiplications (MatMul), which consume significant energy and memory bandwidth. This project implements **MatMul-Free (MMF) networks** that eliminate these operations entirely, replacing them with simple additions and subtractions through ternary weight quantization.

See the 10 minute video presentation [here](https://youtu.be/-jWEjfmtxMs)

### Key Innovation

Instead of standard convolution and linear layers, this implementation uses:
- **Custom `MMFConv2d` and `MMFLinear` layers** with fused RMSNorm
- **Ternary weight quantization** ({-1, 0, +1}) to eliminate floating-point multiplications
- **Activation quantization** for efficient inference
- **Custom forward and backward passes** optimized for low-bit operations

### Performance

Two architecturally equivalent models are trained under identical conditions:
- **Baseline Model** (standard Conv2d): **85.6%** validation accuracy
- **MMF Model** (ternary quantized): **67.1%** validation accuracy

The 18.5 percentage point gap reflects the fundamental trade-off between computational efficiency and representational capacity. While the MMF model achieves non-trivial accuracy with dramatically reduced operations, closing this gap remains an open research challenge.

This repository provides a complete experimental framework for exploring MatMul-free architectures, including hyperparameter tuning, data augmentation strategies, and quantization-aware training techniques.


---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/jonurce/MatMulFree_Image_Classification_Model.git
cd MatMulFree_Image_Classification_Model
```

### 2. Install Required Packages

#### 2.1. Create New Virtual Environment
```bash
python -m venv venv
```

#### 2.2. Activate Virtual Environment
**Windows:**
```powershell
.\venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

#### 2.3. Install All Packages from requirements.txt
```bash
pip install -r requirements.txt
```

### 3. Download CIFAR-10 Dataset

CIFAR-10 dataset is available at: https://www.cs.toronto.edu/~kriz/cifar.html

Follow these instructions to download and extract the dataset:

#### 3.1. Navigate to Your Project Folder
```bash
cd ~/path/to/workspace/MatMulFree_Image_Classification_Model
```

#### 3.2. Create Download and Dataset Folders
```bash
mkdir -p _downloads _dataset
```

#### 3.3. Download CIFAR-10 Dataset
```bash
cd _downloads
wget "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
```

*Note: If `wget` is not available, use `curl`:*
```bash
curl -O "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
```

#### 3.4. Extract Dataset
```bash
cd ..
mkdir -p _dataset/cifar10
tar -xzvf _downloads/cifar-10-python.tar.gz -C _dataset/cifar10
```

#### 3.5. (Optional) Clean Up Downloads Folder
```bash
rm -rf _downloads
```

---

## Repository Structure

```
MatMulFree_Image_Classification_Model/
│
├── _dataset/                      # Dataset folder (created during setup)
│   └── cifar10/                   # CIFAR-10 extracted files
│       └── cifar-10-batches-py/   # Training and test batches
│
├── classification/                # Main folder
│   │
│   ├── _1_dataset/                # Dataset folder
│   │   ├── samples/               # Samples from dataset
│   │   ├── dataset.py             # Defines dataset splits and augmentations
│   │   └── test_dataset.py        # Script to test if dataset is correct (creates samples)
│   │
│   ├── _2_train/                  # Train folder
│   │   ├── runs/                  # Runs tested during the project
│   │   ├── layers_mmf.py          # MMF Layer definition for different experiments
│   │   ├── model.py               # Definition of used models during the project
│   │   ├── test.py                # Script to test chosen model
│   │   ├── train_mmf.py           # Script to train MMF model
│   │   └── train.py               # Script to train baseline model
│   │
│   └── _3_plot/                   # Plot folder
│       └── ...                    # Scripts used for plotting results
│
├── Master Thesis Poster - Jon Urcelay.pdf          # Project Poster
├── Master Thesis Final Report - Jon Urcelay.pdf    # Project Report
├── requirements.txt                                # Python dependencies
└── README.md                                       # Project documentation
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{matmulfree2026,
  author = {Your Name},
  title = {Matrix Multiplication Free Image Classification Model},
  year = {2026},
  url = {https://github.com/jonurce/MatMulFree_Image_Classification_Model}
}
```

---

**Questions or Issues?** Feel free to open an issue on [GitHub](https://github.com/jonurce/MatMulFree_Image_Classification_Model/issues).



