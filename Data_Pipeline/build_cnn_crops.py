"""
Cắt từng biển (theo bounding box ground-truth) từ dataset YOLO thành dataset phân loại cho CNN.

    python build_cnn_crops.py [--src DIR] [--out DIR] [--margin 0.1]

Output: <out>/{train,val,test}/<id>_<tên>/*.png, ảnh 64x64 (resize giữ tỉ lệ + padding đen,
đúng với resize_with_padding lúc inference). Giữ nguyên split train/val/test của YOLO để tránh rò rỉ dữ liệu.
"""
import argparse
import sys
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from Utils.helper_functions import resize_with_padding  # noqa: E402

SIZE = 64
MIN_SIDE = 8  # bỏ crop quá nhỏ (nhiễu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=str(ROOT.parent / 'Data_Processed' / 'yolo_dataset'))
    ap.add_argument('--out', default=str(ROOT.parent / 'Data_Processed' / 'cnn_dataset'))
    ap.add_argument('--margin', type=float, default=0.1, help='nới box mỗi phía theo tỉ lệ')
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)

    names = yaml.safe_load((src / 'data.yaml').read_text())['names']
    folder = {int(k): f"{int(k)}_{v}" for k, v in names.items()}
    counts = {}
    for sp in ('train', 'val', 'test'):
        for d in folder.values():
            (out / sp / d).mkdir(parents=True, exist_ok=True)
        for img_path in tqdm(sorted((src / 'images' / sp).glob('*.jpg')), desc=sp):
            lab = src / 'labels' / sp / f'{img_path.stem}.txt'
            if not lab.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            for i, line in enumerate(l for l in lab.read_text().splitlines() if l.strip()):
                c, cx, cy, bw, bh = line.split()
                c, cx, cy, bw, bh = int(c), float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                mx, my = bw * args.margin, bh * args.margin
                x1, y1 = max(0, int(cx - bw / 2 - mx)), max(0, int(cy - bh / 2 - my))
                x2, y2 = min(w, int(cx + bw / 2 + mx)), min(h, int(cy + bh / 2 + my))
                if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
                    continue
                crop = resize_with_padding(img[y1:y2, x1:x2], (SIZE, SIZE))
                cv2.imwrite(str(out / sp / folder[c] / f'{img_path.stem}_{i}.png'), crop)
                counts[(sp, c)] = counts.get((sp, c), 0) + 1

    print("\nSố crop (train/val/test):")
    for c, d in folder.items():
        print(f"  {d:8s}", *(counts.get((sp, c), 0) for sp in ('train', 'val', 'test')))
    print(f"✅ {out}")


if __name__ == '__main__':
    main()
