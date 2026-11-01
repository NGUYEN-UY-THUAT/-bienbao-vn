# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Overview

Vietnamese traffic sign **classification** (PyTorch), input 32x32x3, solved two ways and compared (for a course report/slides):
- **Hướng 1**: ResNet-18 pretrained (ImageNet, torchvision, architecture unchanged, `fc` → `nn.Identity()` + new head), transfer learning: freeze → train new head 5 epochs → unfreeze, Adam lr 1e-4, 35 epochs.
- **Hướng 2**: custom CNN (3 `MyConvBlock`s, first conv 7x7 required by the course), trained from scratch 40 epochs.

No YOLO/detection anymore (the teacher wants classification only). Style follows the user's course (NVIDIA DLI "Getting Started with Deep Learning", notebooks in `~/PX4-ROS2-Gazebo-YOLOv8/DL/`): `MyDataset` loading everything to GPU, plain `train()`/`validate()` loops, `transforms.v2` augmentation, `requires_grad_(False/True)` freeze/unfreeze (user chose ResNet-18 over the course's VGG16). Keep code that simple; the user asked not to over-engineer. README and comments are in Vietnamese.

## Classes

Dataset ids: 0 P (cấm), 1 W (nguy hiểm), 2 R (hiệu lệnh), 3 I (chỉ dẫn), 4 IE, 5 S (phụ). Meanings inferred from QCVN 41. IE is dropped (17 train crops, 0 val/test), so folders are `0_P 1_W 2_R 3_I 5_S` and model output index 0-4 ≠ dataset id.

## Workflow

Data lives in the **parent** of the repo: `../Data_Processed/`.
1. `Data_Pipeline/ndjson_to_yolo.py`: downloads images + boxes from the ndjson (YOLO label format, only used to crop). **ndjson URLs expired 2026-10-28**; images are already downloaded in `../Data_Processed/yolo_dataset`.
2. `Data_Pipeline/build_cnn_crops.py`: crops boxes into 32x32 padded PNGs → `../Data_Processed/cnn_dataset_32` (skips IE). Zip it for Colab (`cnn_dataset_32.zip`, ~74 MB).
3. Training runs on **Google Colab** (user's choice): `Training/Huong1_ResNet18_pretrained.ipynb`, `Training/Huong2_CNN_tu_xay.ipynb`, then `Training/So_sanh_2_huong.ipynb` (table + figures to `ket_qua/hinh_slide/`). Drive layout: `MyDrive/bienbao/cnn_dataset_32.zip`, outputs in `MyDrive/bienbao/ket_qua/`. The notebooks also run locally (fall back to `../../Data_Processed/cnn_dataset_32/`, output `../Inference/`).
4. `Inference/predict.py`: classifies pre-cropped sign images with both models.

## Model files

Models are saved whole (`torch.save(model)`, load with `weights_only=False`), both are plain `nn.Sequential` (MyConvBlock is a function returning `nn.Sequential`, ResNet-18 model starts with `transforms.v2.Normalize`) so they load without class definitions. Both take RGB in [0, 1], shape (N, 3, 32, 32). No horizontal flip in augmentation (changes sign meaning).

## Leftovers

The old YOLO + 64x64 pipeline files were deleted on purpose (recoverable from git history before this change). In `Utils/helper_functions.py` only `resize_with_padding` is used; the rest are old-project leftovers.
