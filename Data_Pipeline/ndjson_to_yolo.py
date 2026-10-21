"""
Tải ảnh từ file .ndjson (Ultralytics Platform export) và ghi ra dataset YOLO.

    python ndjson_to_yolo.py [--ndjson FILE] [--out DIR] [--workers 16]

Output (mặc định ../../Data_Processed/yolo_dataset, theo quy ước data nằm ngoài repo):
    images/{train,val,test}/*.jpg   labels/{train,val,test}/*.txt   data.yaml
Chạy lại được: ảnh đã tải sẽ bị bỏ qua. Link trong ndjson có hạn, nên tải sớm.
"""
import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def fetch(url: str, dest: Path, retries: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            data = urllib.request.urlopen(req, timeout=30).read()
            dest.write_bytes(data)
            return True
        except Exception:
            continue
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ndjson', default=str(ROOT / 'classification-of-traffic-signs-vn.ndjson'))
    ap.add_argument('--out', default=str(ROOT.parent / 'Data_Processed' / 'yolo_dataset'))
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out)
    with open(args.ndjson, encoding='utf-8') as f:
        rows = [json.loads(l) for l in f if l.strip()]
    meta, images = rows[0], [r for r in rows[1:] if r.get('type') == 'image']
    names = {int(k): v for k, v in meta['class_names'].items()}

    jobs = []
    for r in images:
        sp = r['split']
        (out / 'images' / sp).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / sp).mkdir(parents=True, exist_ok=True)
        stem = Path(r['file']).stem
        lines = [f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}"
                 for b in r['annotations'].get('boxes', [])]
        (out / 'labels' / sp / f'{stem}.txt').write_text('\n'.join(lines) + '\n')
        jobs.append((r['url'], out / 'images' / sp / f'{stem}.jpg'))

    with ThreadPoolExecutor(args.workers) as ex:
        ok = list(tqdm(ex.map(lambda j: fetch(*j), jobs), total=len(jobs), desc='Tải ảnh'))

    failed = [j[1] for j, o in zip(jobs, ok) if not o]
    # Bỏ nhãn của ảnh tải lỗi để dataset nhất quán
    for p in failed:
        (out / 'labels' / p.parent.name / f'{p.stem}.txt').unlink(missing_ok=True)

    yaml = [f"path: {out}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    yaml += [f"  {k}: {v}" for k, v in sorted(names.items())]
    (out / 'data.yaml').write_text('\n'.join(yaml) + '\n')
    print(f"✅ {len(jobs) - len(failed)}/{len(jobs)} ảnh -> {out}")
    if failed:
        print(f"⚠️ {len(failed)} ảnh lỗi (link hết hạn/mạng). Chạy lại để thử tiếp.")


if __name__ == '__main__':
    main()
