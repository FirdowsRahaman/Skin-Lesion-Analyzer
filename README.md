# Skin Lesion Analyzer & U-Net Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Architecture: U-Net](https://img.shields.io/badge/Architecture-U--Net-orange.svg)](#)
[![Dermatology AI](https://img.shields.io/badge/Domain-Dermatology%20AI-teal.svg)](#)

Automated skin lesion segmentation and classification for clinical dermatology using U-Net and Deep Learning.

## Overview
Dermatological AI application designed to segment lesion boundaries from dermoscopic images and assess malignant vs. benign lesion characteristics. Built with modular Python modules and U-Net encoder-decoder architectures with skip connections for high-resolution edge detection.

## Architecture & Modules
- `unet.py`: Core U-Net architecture implementation with contracting encoder, bottleneck, and expanding decoder paths with skip connections.
- `model.py`: Classifier and segmentation head configurations.
- `data_loader.py`: Dermoscopic image batching, augmentation, and ground-truth mask alignment.
- `main.py`: Modular training loop CLI supporting hyperparameter sweeps, Dice loss, and IoU metric logging.
- `Skin Lesion Analyzer Notebook.ipynb`: Interactive analysis, lesion boundary segmentation visualization, and Grad-CAM overlays.

## Quickstart
```bash
git clone https://github.com/FirdowsRahaman/Skin-Lesion-Analyzer.git
cd Skin-Lesion-Analyzer
pip install torch torchvision opencv-python matplotlib scikit-learn
python main.py
```

## License
This project is licensed under the MIT License.
