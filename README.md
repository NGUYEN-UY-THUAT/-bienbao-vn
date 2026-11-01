# 🚦 Phân loại biển báo giao thông Việt Nam: so sánh 2 hướng

## Giới thiệu

Dự án phân loại ảnh biển báo giao thông Việt Nam (ảnh đầu vào **32x32x3**) thành 5 nhóm. Bài toán được giải bằng **2 cách** rồi so sánh kết quả:

| | Hướng 1: Kiến trúc có sẵn | Hướng 2: CNN tự xây |
|---|---|---|
| Model | **ResNet-18** pretrained trên ImageNet (transfer learning) | CNN gồm 3 khối Conv–BatchNorm–ReLU–Dropout–MaxPool |
| Cách train | Freeze ResNet-18 rồi train lớp phân loại mới (5 epoch), sau đó unfreeze và fine-tune với lr = 1e-4 (35 epoch) | Train từ đầu với Adam (40 epoch) |
| Notebook | `Training/Huong1_ResNet18_pretrained.ipynb` | `Training/Huong2_CNN_tu_xay.ipynb` |

Hai hướng dùng chung dữ liệu, augmentation, batch size và tổng số epoch nên so sánh được với nhau. Code viết theo phong cách các bài trong khoá Deep Learning trên lớp: bài 03/04a cho CNN và augmentation, bài 05b/07 cho các bước transfer learning (freeze → train lớp mới → unfreeze → fine-tune).

Dữ liệu lấy từ file `classification-of-traffic-signs-vn.ndjson`. Từ mỗi ảnh gốc, từng biển báo được cắt ra theo bounding box có sẵn để tạo **35.550 ảnh 32x32**.

## Phân loại biển báo

| Ký hiệu | Nhóm biển | Đặc điểm nhận dạng | Số ảnh train / val / test |
|---|---|---|---|
| **P** | Biển báo cấm | Hình tròn, viền đỏ | 14886 / 1997 / 1115 |
| **W** | Biển báo nguy hiểm | Hình tam giác, viền đỏ, nền vàng | 9564 / 1432 / 810 |
| **R** | Biển hiệu lệnh | Hình tròn, nền xanh dương | 2716 / 297 / 199 |
| **I** | Biển chỉ dẫn | Hình chữ nhật hoặc vuông, nền xanh dương | 1514 / 182 / 96 |
| **S** | Biển phụ | Hình chữ nhật nhỏ, đi kèm biển chính | 637 / 66 / 39 |

> Ý nghĩa các ký hiệu được suy ra từ quy chuẩn QCVN 41, vì dataset gốc chỉ ghi chữ cái. Lớp IE bị bỏ vì chỉ có 17 ảnh train và không có ảnh nào trong val/test.

## Cấu trúc

```
vietnam-traffic-sign-recognition-main/
├── Data_Pipeline/
│   ├── ndjson_to_yolo.py              # tải ảnh gốc + nhãn (bounding box) từ file .ndjson
│   └── build_cnn_crops.py             # cắt từng biển thành ảnh 32x32
├── Training/                          # chạy trên Google Colab
│   ├── Huong1_ResNet18_pretrained.ipynb  # Hướng 1: transfer learning ResNet-18
│   ├── Huong2_CNN_tu_xay.ipynb        # Hướng 2: CNN tự xây
│   └── So_sanh_2_huong.ipynb          # so sánh, xuất bảng + hình cho slide
├── Inference/
│   └── predict.py                     # dự đoán ảnh mới bằng cả 2 model
├── Utils/helper_functions.py          # resize giữ tỉ lệ + padding
├── test_inputs/                       # ← bỏ ảnh biển báo (đã cắt) cần dự đoán vào đây
└── requirements.txt
```

## Các bước thực hiện

### 1. Chuẩn bị dữ liệu (trên máy)

```bash
pip install -r requirements.txt
python Data_Pipeline/ndjson_to_yolo.py     # tải ~24.5k ảnh gốc (~1.8 GB) → ../Data_Processed/yolo_dataset
python Data_Pipeline/build_cnn_crops.py    # cắt biển 32x32 → ../Data_Processed/cnn_dataset_32
cd ../Data_Processed && zip -qr cnn_dataset_32.zip cnn_dataset_32
```

> Link ảnh trong file `.ndjson` có hạn sử dụng. Khi link hết hạn, cần export lại file `.ndjson` từ Ultralytics Platform.

### 2. Train trên Google Colab

1. Upload `cnn_dataset_32.zip` (~74 MB) lên Google Drive, vào thư mục `MyDrive/bienbao/`.
2. Mở notebook trên Colab và chọn **Runtime → Change runtime type → T4 GPU**.
3. Chạy lần lượt `Huong1_ResNet18_pretrained.ipynb`, `Huong2_CNN_tu_xay.ipynb` và cuối cùng `So_sanh_2_huong.ipynb`.

Kết quả được lưu ở `MyDrive/bienbao/ket_qua/`:
- `resnet18_pretrained.pth`, `custom_cnn.pth`: model đã train
- `*_history.json`: loss/accuracy từng epoch, thời gian train, test accuracy
- `hinh_slide/`: biểu đồ quá trình train, biểu đồ so sánh các chỉ số, confusion matrix và các ảnh bị đoán sai

Notebook `So_sanh_2_huong.ipynb` còn in sẵn một **bảng markdown** để chép vào slide, gồm số tham số, dung lượng, thời gian train, thời gian dự đoán, test accuracy và macro F1.

### 3. Dự đoán ảnh mới (trên máy)

Chép 2 file `.pth` vào `Inference/`, bỏ ảnh biển báo đã cắt vào `test_inputs/`, rồi chạy:

```bash
python Inference/predict.py              # hoặc: python Inference/predict.py anh.png
```

Mỗi ảnh được in ra kết quả dự đoán và độ tin cậy của cả 2 model.
