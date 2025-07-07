# Smart Cities Semantic Segmentation

This project implements a deep learning-based semantic segmentation system for urban street scenes using the Cityscapes dataset. It compares DeepLabV3 and HRNet architectures, trains multiple variants with different hyperparameters, and delivers a user-friendly interface for testing and deployment.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Training and Validation](#training-and-validation)
- [Testing and Evaluation](#testing-and-evaluation)
- [Results Analysis](#results-analysis)
- [Interface (GUI)](#interface-gui)
- [Effectiveness and Critical Evaluation](#effectiveness-and-critical-evaluation)
- [References](#references)

---

## Problem Statement

Smart cities rely on accurate understanding of street scenes for traffic analysis, planning, and safety monitoring. This project tackles semantic segmentation, assigning a class label to each pixel in street-level imagery.

We use the Cityscapes dataset with 19 annotated classes to train and evaluate two modern architectures:

- DeepLabV3 with ResNet-50 and ResNet-101 backbones
- HRNetV2-w48

We compare these models in terms of accuracy (mIoU), size, and training time, and deliver a GUI that allows users to:

- Choose among six trained model variants
- Upload images or videos
- View and download segmentation outputs with overlay masks

---

## Dataset

- Cityscapes dataset: 5000 high-resolution images with 19 semantic classes.
- Splits:
  - 2975 training
  - 500 validation
  - 1525 test

**Preprocessing Steps**:
- Resize to 512x256 pixels for memory and speed
- Normalize using ImageNet statistics
- Remap unused labels to 255 (ignored during training)

A custom PyTorch Dataset class ensures consistent loading and preprocessing.

---

## Training and Validation

All training was conducted on Google Colab with A100 GPUs (40GB VRAM).

**Key Steps**:
- Data preprocessing
- Forward pass
- Cross-entropy loss computation
- Backpropagation and optimization
- Learning rate scheduling
- Per-epoch validation on the Cityscapes validation split

**Hyperparameter Configurations (Examples):**

| Model Variant | Optimizer | LR | Scheduler | Regularization |
| --- | --- | --- | --- | --- |
| DeepLabV3-1 | Adam | 1e-4 | — | — |
| DeepLabV3-2 | SGD (momentum=0.9) | 1e-3 | StepLR(5, 0.5) | Weight Decay |
| DeepLabV3-3 | AdamW | 1e-4 | StepLR(5, 0.5) | Dropout(0.3) |
| HRNet Variants | AdamW / SGD | 1e-4 / 1e-3 | StepLR | Dropout, Weight Decay |

Training epochs ranged from 10 to 20 depending on convergence behavior.

---

## Testing and Evaluation

All models were tested under consistent conditions with the Cityscapes validation set. 

**Metrics Used**:
- Cross Entropy Loss: pixel-wise classification error
- Mean Intersection over Union (mIoU): measures overlap between predicted and ground truth masks

**Best Models Results**:

| Architecture | Best Hyperparameters | mIoU |
| --- | --- | --- |
| DeepLabV3 | ResNet-101, SGD | 0.6475 |
| HRNet | AdamW, StepLR | 0.741 |

---

## Results Analysis

**DeepLabV3 Observations**:
- Adam-based variants plateaued early.
- SGD variant generalized better, achieving the highest DeepLabV3 mIoU of 0.6475.

**HRNet Observations**:
- All three HRNet variants achieved mIoU between 0.7322 and 0.7406.
- Showed stable training and less overfitting.
- Outperformed DeepLabV3 in accurately segmenting small, detailed objects in urban scenes.

**Performance vs Efficiency Trade-off**:

- HRNet models had higher memory and computation requirements but delivered superior accuracy.
- DeepLabV3 was lighter, faster to train, and better suited for resource-constrained deployment.

**Memory Usage**:

| Model | Parameters (M) | Disk Size (MB) |
| --- | --- | --- |
| DeepLabV3-1/2 | ~42M | ~164MB |
| DeepLabV3-3 | ~60M | ~239MB |
| HRNet-1/2 | ~65M | ~256MB |
| HRNet-3 | ~38M | ~263MB |

**Training Times**:

| Model | Minutes |
| --- | --- |
| DeepLabV3 | ~116–125 |
| HRNet | ~144–174 |

---

## Interface (GUI)

A core component of this project is the user interface, designed to bridge research and real-world application.

**Features**:
- Choose between 6 trained model variants (3 DeepLabV3, 3 HRNet)
- Upload images or videos for segmentation
- Control overlay transparency
- Select/hide any of the 19 classes in results
- Download segmented outputs

![gui pic](https://github.com/user-attachments/assets/ebcb599f-e385-449c-8227-9352f6129b14)



                        ![image](https://github.com/user-attachments/assets/4152c2c9-e5cf-46ba-b90c-ceffafb611fd)



```markdown
![Main Page](path/to/main_page.png)
![Upload Section](path/to/upload_section.png)
![Results](path/to/results_section.png)
```

---

## Effectiveness and Critical Evaluation

**Strengths**:
- High-quality semantic segmentation results for urban scenes.
- HRNet models achieved mIoU up to 0.741, exceeding DeepLabV3’s best at 0.6475.
- User-friendly GUI for non-technical users.

**Limitations**:
- Slower inference time on HRNet, especially for video.
- Static input resolution (512x256).
- High GPU memory requirements for HRNet.
- Limited testing beyond European city imagery.
- No augmentations in final training runs, potentially reducing robustness.

**Future Improvements**:
- Implement model pruning for faster, real-time video inference.
- Support dynamic input sizes.
- Expand training data to cover diverse global cities.
- Add data augmentation for improved real-world variability.

---

## References

- Cordts, M. (2016). The Cityscapes Dataset for Semantic Urban Scene Understanding.
- GeeksForGeeks. (2023). [IoU for Evaluating Image Segmentation](https://www.geeksforgeeks.org/java/calculation-intersection-over-union-iou-for-evaluating-an-image-segmentation-model-using-java/)
- Mehta, S. (2018). ESPNet: Efficient Spatial Pyramid of Dilated.
- Paszke, A. (2016). ENet: A Deep Neural Network Architecture.
- Pohlen, T. (2016). Full-Resolution Residual Networks.
- Wang, Y. (2019). ESNet: An Efficient Symmetric Network.
