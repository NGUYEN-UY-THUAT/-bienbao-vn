"""
Hybrid inference: YOLOv8 phát hiện biển báo -> CNN (và/hoặc ResNet-18, PyTorch) phân loại.

Cách dùng nhanh: bỏ ảnh/video vào test_inputs/ rồi chạy `python Model_hybrid.py`;
kết quả (ảnh/video đã vẽ khung + nhãn) nằm trong results/pred_<tên file gốc>.

Ví dụ:
    python Model_hybrid.py                                  # xử lý tất cả file trong test_inputs/
    python Model_hybrid.py --source anh.jpg                 # 1 ảnh -> results/pred_anh.jpg
    python Model_hybrid.py --source video.mp4 --out kq.mp4  # video, chọn tên file kết quả
    python Model_hybrid.py --source 0 --show                # webcam

Nếu có nhiều file model phân loại, xác suất được trung bình (ensemble).
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from Utils.models import load_classifier  # noqa: E402
from Utils.helper_functions import (  # noqa: E402
    resize_with_padding,
    draw_detection,
    draw_fps,
    clamp_bbox,
)

IMG_SIZE = (64, 64)
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
DEFAULT_INPUT_DIR = HERE.parent / 'test_inputs'  # bỏ ảnh/video cần test vào đây
DEFAULT_OUT_DIR = HERE.parent / 'results'        # kết quả được lưu ở đây
DEFAULT_YOLO = HERE / 'best.pt'
DEFAULT_CLASSIFIERS = [HERE / 'traffic_sign_cnn.pt',
                       HERE / 'best_cnn_traffic_sign.pt',
                       HERE / 'best_resnet18_traffic_sign.pt']


class HybridDetector:
    def __init__(self, yolo_path, cnn_paths, conf=0.4, cnn_conf=0.0):
        from ultralytics import YOLO

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo = YOLO(str(yolo_path))
        loaded = [load_classifier(p, self.device) for p in cnn_paths]
        self.classifiers = [m for m, _ in loaded]
        # '0_P' -> 'P' (tên lớp lấy từ tên thư mục lúc train)
        self.class_names = [n.split('_', 1)[-1] for n in loaded[0][1]]
        if any(c != loaded[0][1] for _, c in loaded):
            sys.exit('❌ Các model phân loại có danh sách lớp khác nhau.')
        self.conf = conf
        self.cnn_conf = cnn_conf
        print(f"✅ YOLO: {yolo_path}")
        for p in cnn_paths:
            print(f"✅ Classifier: {p}")

    @staticmethod
    def _preprocess(crop_bgr):
        # Giống tiền xử lý lúc train: resize giữ tỉ lệ + padding, BGR->RGB, [0, 1]
        padded = resize_with_padding(crop_bgr, IMG_SIZE)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32) / 255.0

    def classify(self, crops):
        batch = np.stack([self._preprocess(c) for c in crops])  # (N, H, W, 3)
        x = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous().to(self.device)
        with torch.no_grad():
            probs = [torch.softmax(m(x), dim=1).cpu().numpy() for m in self.classifiers]
        return np.mean(probs, axis=0)

    def process(self, frame):
        """Trả về frame đã vẽ và danh sách (bbox, class_id, score)."""
        h, w = frame.shape[:2]
        result = self.yolo.predict(frame, conf=self.conf, verbose=False)[0]
        boxes, crops = [], []
        for b in result.boxes.xyxy.cpu().numpy().astype(int):
            x1, y1, x2, y2 = clamp_bbox(*b, w, h)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            boxes.append((x1, y1, x2, y2))
            crops.append(frame[y1:y2, x1:x2])

        detections = []
        if crops:
            probs = self.classify(crops)
            for (x1, y1, x2, y2), p in zip(boxes, probs):
                cid = int(np.argmax(p))
                score = float(p[cid])
                if score < self.cnn_conf:
                    continue
                draw_detection(frame, x1, y1, x2, y2, self.class_names[cid], score)
                detections.append(((x1, y1, x2, y2), cid, score))
        return frame, detections


def pick_classifiers(paths):
    chosen = [Path(p) for p in paths] if paths else [p for p in DEFAULT_CLASSIFIERS if p.exists()]
    missing = [p for p in chosen if not p.exists()]
    if missing or not chosen:
        sys.exit("❌ Không tìm thấy model phân loại (.pt): "
                 f"{missing or [str(p) for p in DEFAULT_CLASSIFIERS]}\n"
                 "   Huấn luyện bằng Training/cnn-detectsign.ipynb rồi copy file .pt vào Inference/, "
                 "hoặc dùng --cnn <đường dẫn>.")
    return chosen


def run_image(det, src, out, show=False):
    frame = cv2.imread(str(src))
    if frame is None:
        print(f"❌ Không đọc được ảnh: {src}")
        return
    frame, dets = det.process(frame)
    summary = ', '.join(f"{det.class_names[c]} {s:.2f}" for _, c, s in dets) or 'không phát hiện biển nào'
    print(f"{Path(src).name}: {summary}")
    if out:
        cv2.imwrite(str(out), frame)
    if show:
        cv2.imshow('Hybrid', frame)
        cv2.waitKey(0)


def run_video(det, source, out, show=False):
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not cap.isOpened():
        print(f"❌ Không mở được nguồn: {source}")
        return
    writer = None
    if out:
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), fps_in, size)
    n = 0
    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                break
            frame, _ = det.process(frame)
            n += 1
            draw_fps(frame, 1.0 / max(time.time() - t0, 1e-6))
            if writer:
                writer.write(frame)
            if show:
                cv2.imshow('Hybrid (q để thoát)', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    print(f"{Path(str(source)).name}: {n} khung hình" + (f" -> {out}" if out else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', default=str(DEFAULT_INPUT_DIR),
                    help='ảnh, video, THƯ MỤC chứa ảnh/video, hoặc id webcam (0). Mặc định: test_inputs/')
    ap.add_argument('--yolo', default=str(DEFAULT_YOLO))
    ap.add_argument('--cnn', nargs='*', help='một hoặc nhiều file .pt (nhiều file = ensemble)')
    ap.add_argument('--conf', type=float, default=0.4, help='ngưỡng YOLO')
    ap.add_argument('--cnn-conf', type=float, default=0.0, help='bỏ kết quả phân loại dưới ngưỡng này')
    ap.add_argument('--out', default=str(DEFAULT_OUT_DIR),
                    help='thư mục lưu kết quả (mặc định results/); với 1 file có thể là đường dẫn file')
    ap.add_argument('--show', action='store_true', help='hiển thị cửa sổ (cần GUI)')
    args = ap.parse_args(argv)

    src = Path(args.source)
    out = Path(args.out)

    if args.source.isdigit():  # webcam: chỉ lưu nếu --out là file
        det = HybridDetector(args.yolo, pick_classifiers(args.cnn), args.conf, args.cnn_conf)
        run_video(det, args.source, out if out.suffix else None, args.show)
        return

    if src.is_dir():
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS)
        if not files:
            sys.exit(f"❌ Thư mục {src} chưa có ảnh/video nào ({', '.join(sorted(IMAGE_EXTS | VIDEO_EXTS))}).")
    elif src.is_file():
        files = [src]
    else:
        sys.exit(f"❌ Không tìm thấy: {src}")

    det = HybridDetector(args.yolo, pick_classifiers(args.cnn), args.conf, args.cnn_conf)
    single_file_out = len(files) == 1 and out.suffix  # --out là đường dẫn file
    if not single_file_out:
        out.mkdir(parents=True, exist_ok=True)
    for f in files:
        is_img = f.suffix.lower() in IMAGE_EXTS
        dest = out if single_file_out else out / (f"pred_{f.name}" if is_img else f"pred_{f.stem}.mp4")
        (run_image if is_img else run_video)(det, f, dest, args.show)
    print(f"✅ Kết quả lưu tại: {out.resolve()}")


if __name__ == '__main__':
    main()
