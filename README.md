# 🚦 Nhận diện biển báo giao thông Việt Nam (YOLOv8 + CNN)

Pipeline hai bước viết bằng PyTorch:

```
Ảnh / Video ──▶ YOLOv8n (phát hiện vị trí biển) ──▶ cắt crop ──▶ CNN + ResNet-18 (phân loại) ──▶ ảnh/video có khung + nhãn
```

Phân loại **5 nhóm biển**: `P` (cấm), `W` (nguy hiểm), `R` (hiệu lệnh), `I` (chỉ dẫn), `S` (phụ).
Dữ liệu: `classification-of-traffic-signs-vn.ndjson` (~24.5 nghìn ảnh 512×512, ~39 nghìn biển, đã chia train/val/test).

## 📊 Kết quả

| Thành phần | Chỉ số (tập test) |
|---|---|
| YOLOv8n (10 epoch) | mAP50 **0,940** · mAP50-95 0,776 |
| Custom CNN (VGG-style, 2,5 triệu tham số) | accuracy **99,65%** |
| ResNet-18 (tự cài, 11,2 triệu tham số) | accuracy **99,56%** |
| Ensemble CNN + ResNet-18 | accuracy **99,73%** |

> Accuracy của CNN đo trên crop cắt theo nhãn gốc, nên cao hơn chất lượng thực của cả pipeline (còn phụ thuộc YOLO phát hiện đủ biển). YOLO có recall thấp nhất ở nhóm `R` (0,82).

## 📁 Cấu trúc

```
vietnam-traffic-sign-recognition-main/
├── classification-of-traffic-signs-vn.ndjson   # nhãn + link ảnh (link hết hạn 28/10/2026)
├── Data_Pipeline/
│   ├── ndjson_to_yolo.py        # tải ảnh, tạo dataset YOLO
│   └── build_cnn_crops.py       # cắt biển thành dataset phân loại 64x64
├── Training/
│   ├── train_yolo.py            # huấn luyện YOLOv8n
│   └── cnn-detectsign.ipynb     # huấn luyện CNN + ResNet-18, đánh giá, ensemble
├── Inference/
│   ├── Model_hybrid.py          # chạy nhận diện ảnh/video/webcam
│   ├── best.pt                  # YOLOv8n
│   ├── best_cnn_traffic_sign.pt
│   └── best_resnet18_traffic_sign.pt
├── Utils/
│   ├── models.py                # CustomCNN, ResNet18, EfficientNetB0 (PyTorch)
│   └── helper_functions.py      # resize có padding, vẽ khung, bbox...
├── test_inputs/                 # BỎ ẢNH/VIDEO CẦN TEST VÀO ĐÂY
└── results/                     # KẾT QUẢ NHẬN DIỆN NẰM Ở ĐÂY
```

Dữ liệu đã xử lý nằm **ngoài repo**, tại `../Data_Processed/{yolo_dataset,cnn_dataset}`.

## ⚙️ Cài đặt

```bash
python3 -m venv .venv && source .venv/bin/activate
# Có GPU NVIDIA: cài PyTorch bản CUDA trước (đã thử CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## 🚀 Chạy nhận diện (đã có sẵn model)

1. Bỏ ảnh hoặc video vào `test_inputs/`.
2. Chạy:
   ```bash
   python Inference/Model_hybrid.py
   ```
3. Xem kết quả trong `results/pred_<tên file gốc>` (ảnh `.jpg/.png`, video `.mp4`). Terminal cũng in danh sách biển và độ tin cậy.

Tuỳ chọn:
```bash
python Inference/Model_hybrid.py --source anh.jpg                   # một file -> results/pred_anh.jpg
python Inference/Model_hybrid.py --source video.mp4 --out kq.mp4    # chọn tên file kết quả
python Inference/Model_hybrid.py --source thu_muc/ --out ketqua/    # cả thư mục
python Inference/Model_hybrid.py --conf 0.25                        # nhạy hơn với biển nhỏ
python Inference/Model_hybrid.py --source 0 --show                  # webcam, nhấn q để thoát
python Inference/Model_hybrid.py --cnn Inference/best_resnet18_traffic_sign.pt   # chỉ dùng 1 model (mặc định: ensemble)
```

## 🏋️ Huấn luyện lại từ đầu

```bash
# 1. Tải ảnh (~1,8 GB, ~10 phút) và tạo dataset YOLO
python Data_Pipeline/ndjson_to_yolo.py
# 2. Cắt crop cho CNN, rồi xoá thư mục lớp IE (chỉ 17 mẫu, không có val/test)
python Data_Pipeline/build_cnn_crops.py
rm -r ../Data_Processed/cnn_dataset/*/4_IE
# 3. YOLOv8n (~7 phút/epoch trên GTX 1650)
python Training/train_yolo.py --epochs 10
# 4. CNN + ResNet-18 (~1 giờ trên GTX 1650)
cd Training && MPLBACKEND=Agg python -m nbconvert --to notebook --execute cnn-detectsign.ipynb
# 5. Chép model vào Inference/
cp ../runs/traffic_sign/weights/best.pt ../Inference/best.pt
cp best_cnn_traffic_sign.pt best_resnet18_traffic_sign.pt ../Inference/
```

Notebook cũng chạy được trên Kaggle/Colab: đặt biến môi trường `DATASET_PATH` hoặc sửa ở cell cấu hình.

## 🧠 Kiến trúc

- **YOLOv8n:** phát hiện biển, train 512×512, tắt lật ngang (lật làm đổi nghĩa biển rẽ trái/phải).
- **Custom CNN:** 4 khối Conv3×3 + BatchNorm + MaxPool, Dropout, Linear 512 → số lớp.
- **ResNet-18:** tự cài từng BasicBlock (shortcut), chỉnh cho ảnh 64×64, train từ đầu.
- Tiền xử lý crop: resize giữ tỉ lệ + padding đen về 64×64, RGB, chuẩn hoá [0, 1]. Inference dùng đúng cách này.
- Loss có class weights (nghịch đảo căn bậc hai tần suất) để bù lớp ít mẫu (`S`, `I`).

## ⚠️ Hạn chế

- Mới train YOLO 10 epoch; biển rất nhỏ/xa hoặc ảnh mờ có thể bị bỏ sót (nhất là nhóm `R`).
- Lớp `IE` (22 mẫu) không được phân loại.
- Nhãn hiển thị là mã nhóm (`P`, `W`...), chưa phải tên biển cụ thể.
- Ký hiệu P/W/R/I/S được suy từ quy chuẩn QCVN 41; dataset gốc không giải thích nghĩa.
