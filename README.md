# 🚦 Nhận diện biển báo giao thông Việt Nam (YOLOv8 + CNN)

## Giới thiệu

Dự án nhận diện biển báo giao thông Việt Nam từ **ảnh** hoặc **video** bằng pipeline hai bước, viết bằng PyTorch:

```
Ảnh / Video ──▶ YOLOv8n ──▶ cắt từng biển ──▶ CNN + ResNet-18 ──▶ ảnh / video có khung và nhãn
               (phát hiện)                      (phân loại)
```

1. **YOLOv8n** tìm vị trí các biển báo trong khung hình.
2. Mỗi biển được cắt ra và đưa vào **CNN tự xây** (lớp Conv đầu tiên kernel 7×7) và **ResNet-18**. Xác suất của hai mô hình được lấy trung bình (ensemble) để ra nhóm biển cuối cùng.

Dữ liệu huấn luyện lấy từ file `classification-of-traffic-signs-vn.ndjson` (khoảng 24,5 nghìn ảnh, 39 nghìn biển báo).

## Phân loại biển báo

Mô hình phân loại biển báo thành **5 nhóm**:

| Ký hiệu | Nhóm biển | Đặc điểm nhận dạng |
|---|---|---|
| **P** | Biển báo cấm | Hình tròn, viền đỏ |
| **W** | Biển báo nguy hiểm | Hình tam giác, viền đỏ, nền vàng |
| **R** | Biển hiệu lệnh | Hình tròn, nền xanh dương |
| **I** | Biển chỉ dẫn | Hình chữ nhật hoặc vuông, nền xanh dương |
| **S** | Biển phụ | Hình chữ nhật nhỏ, đi kèm biển chính |

> Ý nghĩa các ký hiệu được suy ra từ quy chuẩn QCVN 41, dataset gốc chỉ ghi chữ cái.

## Cấu trúc

```
vietnam-traffic-sign-recognition-main/
├── Data_Pipeline/
│   ├── ndjson_to_yolo.py            # tải ảnh từ file .ndjson, tạo dataset YOLO
│   └── build_cnn_crops.py           # cắt từng biển thành ảnh 64x64 cho CNN
├── Training/
│   ├── train_yolo.py                # huấn luyện YOLOv8n
│   └── cnn-detectsign.ipynb         # huấn luyện CNN + ResNet-18, đánh giá, ensemble
├── Inference/
│   ├── Model_hybrid.py              # chạy nhận diện ảnh / video / webcam
│   ├── best.pt                      # YOLOv8n đã train
│   ├── best_cnn_traffic_sign.pt     # CNN tự xây đã train
│   └── best_resnet18_traffic_sign.pt  # ResNet-18 đã train
├── Utils/
│   ├── models.py                    # định nghĩa CustomCNN, ResNet18 (PyTorch)
│   └── helper_functions.py          # resize có padding, vẽ khung, xử lý bbox
├── test_inputs/                     # ← bỏ ảnh / video cần nhận diện vào đây
├── results/                         # ← kết quả nhận diện nằm ở đây
├── requirements.txt
└── README.md
```

## Cài đặt

Yêu cầu Python 3.10 trở lên. Có GPU NVIDIA thì chạy nhanh hơn, nhưng CPU vẫn chạy được.

```bash
# 1. Tạo môi trường ảo
python3 -m venv .venv
source .venv/bin/activate

# 2. Nếu có GPU NVIDIA: cài PyTorch bản CUDA trước (bỏ qua nếu chạy bằng CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Cài các thư viện còn lại
pip install -r requirements.txt
```

Các file model đã có sẵn trong thư mục `Inference/` (`best.pt`, `best_cnn_traffic_sign.pt`, `best_resnet18_traffic_sign.pt`), nên không cần huấn luyện lại để chạy nhận diện.

## Cách chạy

**1.** Bỏ ảnh hoặc video cần nhận diện vào thư mục `test_inputs/`
(ảnh: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp` · video: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`).

**2.** Chạy:

```bash
python Inference/Model_hybrid.py
```

**3.** Xem kết quả trong thư mục `results/`: mỗi file có một bản kết quả tên `pred_<tên file gốc>`, đã vẽ khung và nhãn. Terminal cũng in danh sách biển được phát hiện kèm độ tin cậy.

### Tuỳ chọn

```bash
# Một file bất kỳ
python Inference/Model_hybrid.py --source anh.jpg

# Chọn tên file kết quả
python Inference/Model_hybrid.py --source video.mp4 --out ketqua.mp4

# Cả một thư mục, lưu kết quả vào thư mục khác
python Inference/Model_hybrid.py --source thu_muc/ --out ketqua/

# Video nhiều khung hình (vd 120 fps): chỉ xử lý 1 trong mỗi 4 khung cho nhanh gấp 4 lần
python Inference/Model_hybrid.py --stride 4

# Bắt thêm biển nhỏ / mờ (hạ ngưỡng phát hiện, mặc định 0.4)
python Inference/Model_hybrid.py --conf 0.25

# Chỉ dùng một mô hình phân loại (mặc định là ensemble CNN + ResNet-18)
python Inference/Model_hybrid.py --cnn Inference/best_resnet18_traffic_sign.pt

# Webcam (nhấn q để thoát)
python Inference/Model_hybrid.py --source 0 --show
```

Video kết quả được nén H.264 nên mở được trên hầu hết trình phát.
