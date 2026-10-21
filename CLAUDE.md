# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Overview

Vietnamese traffic sign recognition with a two-stage hybrid pipeline (all PyTorch): **YOLOv8n** detects signs, then a **CNN + ResNet-18** ensemble classifies each crop into one of 5 sign groups. Data comes from `classification-of-traffic-signs-vn.ndjson` (Ultralytics Platform export: labels + signed image URLs, no images). README and comments are in Vietnamese.

## Classes

Dataset ids: 0 P (cấm/prohibitory), 1 W (nguy hiểm/warning), 2 R (hiệu lệnh/regulatory), 3 I (chỉ dẫn/information), 4 IE, 5 S (phụ/supplementary). Meanings are inferred from QCVN 41 codes (IE is a guess), the ndjson only has the letters.
YOLO is trained on all 6 ids. The CNN dataset **drops IE** (17 train crops, 0 val/test), so CNN folders are `0_P 1_W 2_R 3_I 5_S` and CNN output index 0-4 does NOT equal the dataset id. Inference takes class names from the checkpoint (`class_names`), never from a hard-coded table.

## Workflow

Run from the repo root with `.venv/bin/python` (or any env with `requirements.txt`). Data lives in the **parent** of the repo: `../Data_Processed/{yolo_dataset,cnn_dataset}`.

1. `Data_Pipeline/ndjson_to_yolo.py`: downloads ~24.5k images (~1.8 GB, ~10 min) from the ndjson and writes the YOLO dataset + `data.yaml`. **The URLs in the ndjson expire (2026-10-28)**; re-export the ndjson if they do. Re-running skips existing files.
2. `Data_Pipeline/build_cnn_crops.py`: crops GT boxes into 64x64 padded PNGs (`train/val/test/<id>_<name>/`), keeping the YOLO split. Drop `*_IE` folders afterwards (the script itself still creates them).
3. `Training/train_yolo.py`: YOLOv8n, imgsz 512, `fliplr=0` (flipping changes sign meaning), outputs `runs/traffic_sign/weights/best.pt`. ~7 min/epoch on a GTX 1650.
4. `Training/cnn-detectsign.ipynb`: trains CustomCNN (baseline) and ResNet-18, evaluates both + ensemble. Uses `DATASET_PATH` env var. Run headless: `cd Training && MPLBACKEND=Agg ../.venv/bin/python -m nbconvert --to notebook --execute cnn-detectsign.ipynb`. Writes `best_cnn_traffic_sign.pt`, `best_resnet18_traffic_sign.pt`.
5. Copy `best.pt` and the two `.pt` classifiers into `Inference/`.
6. `Inference/Model_hybrid.py`: images/videos/folders/webcam. Default input `test_inputs/`, default output `results/pred_<name>`.

## Shared code

- `Utils/models.py`: `CustomCNN` (first Conv 7x7 required by the course; `_conv_block(k=7)`), `ResNet18` (hand-written, from scratch, 64x64, 7x7 stride-2 stem, no MaxPool), `EfficientNetB0Classifier` (kept, currently unused), `build_model`, `save_checkpoint`, `load_classifier`. Checkpoints are dicts `{kind, class_names, state_dict}`. All models take `(N,3,64,64)` RGB in [0,1] and return logits.
- `Utils/helper_functions.py`: path helpers, `resize_with_padding`, drawing, bbox utils, YAML I/O. Inference must use `resize_with_padding` (matches the padded training crops), not `preprocess_for_cnn` (plain resize). `TRAFFIC_SIGN_CLASSES`, `YOLO_CLASS_MAPPING`, `get_class_name/color` are leftovers from the old 7-class Zalo project and are unused.

## Results (for reference)

YOLOv8n 10 epochs, test mAP50 0.940 (mAP50-95 0.776). Classifiers on GT crops, test acc: CNN 99.78%, ResNet-18 99.73%, ensemble 99.82% (first conv 7x7 in both). Crop accuracy is optimistic: it ignores detector misses and possible near-duplicate train/test images (ndjson looks like a Roboflow export).

## Notes

- Python 3.10, torch 2.6 + CUDA 12.4 verified; GTX 1650 (4 GB) is enough.
- The old Zalo/7-class notebooks and the 52-class `archive/` dataset were deleted on purpose.
- `runs/` (YOLO training output) and `results/` are generated; don't treat them as source.
