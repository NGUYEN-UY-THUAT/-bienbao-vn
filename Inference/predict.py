"""
Dự đoán nhóm biển báo cho ảnh đã cắt sẵn (mỗi ảnh là 1 biển) bằng cả 2 model:
- Hướng 1: ResNet-18 pretrained (resnet18_pretrained.pth)
- Hướng 2: CNN tự xây           (custom_cnn.pth)

    python Inference/predict.py [ảnh hoặc thư mục, mặc định test_inputs/]

Ảnh được resize giữ tỉ lệ + padding đen về 32x32 (giống lúc tạo dataset).
"""
import sys
from pathlib import Path

import cv2
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from Utils.helper_functions import resize_with_padding  # noqa: E402

CLASS_NAMES = ['P (cấm)', 'W (nguy hiểm)', 'R (hiệu lệnh)', 'I (chỉ dẫn)', 'S (phụ)']
MODELS = {'ResNet-18 pretrained': HERE / 'resnet18_pretrained.pth', 'CNN tự xây': HERE / 'custom_cnn.pth'}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(path):
    img = cv2.imread(str(path))  # BGR
    img = cv2.cvtColor(resize_with_padding(img, (32, 32)), cv2.COLOR_BGR2RGB)
    x = torch.tensor(img).permute(2, 0, 1).float() / 255  # 3 x 32 x 32, [0, 1]
    return x.unsqueeze(0).to(device)


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / 'test_inputs'
    paths = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS) if src.is_dir() else [src]
    if not paths:
        sys.exit(f'Không có ảnh trong {src}')

    models = {}
    for title, path in MODELS.items():
        if not path.exists():
            sys.exit(f'Thiếu {path}: train trên Colab rồi chép file .pth vào Inference/')
        models[title] = torch.load(path, weights_only=False, map_location=device).eval()

    print(f"{'Ảnh':30s}" + ''.join(f'{t:>30s}' for t in models))
    with torch.no_grad():
        for p in paths:
            x = load_image(p)
            row = f'{p.name[:30]:30s}'
            for model in models.values():
                prob = torch.softmax(model(x), dim=1)[0]
                i = prob.argmax().item()
                row += f'{CLASS_NAMES[i] + f" {prob[i] * 100:.1f}%":>30s}'
            print(row)


if __name__ == '__main__':
    main()
