"""
Huấn luyện YOLOv8 phát hiện biển báo.

    python train_yolo.py [--data DIR/data.yaml] [--epochs 50] [--batch 16] [--model yolov8n.pt]

Kết quả: runs/detect/<name>/weights/best.pt -> copy vào Inference/best.pt
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(ROOT.parent / 'Data_Processed' / 'yolo_dataset' / 'data.yaml'))
    ap.add_argument('--model', default='yolov8n.pt')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--imgsz', type=int, default=512)
    ap.add_argument('--name', default='traffic_sign')
    ap.add_argument('--device', default='0')
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
                device=args.device, name=args.name, project=str(ROOT / 'runs'), patience=15,
                fliplr=0.0,  # không lật ngang: làm sai nghĩa biển (rẽ trái/phải)
                workers=4, exist_ok=True)
    metrics = model.val(split='test')
    print(f"mAP50 (test): {metrics.box.map50:.3f} | mAP50-95: {metrics.box.map:.3f}")


if __name__ == '__main__':
    main()
