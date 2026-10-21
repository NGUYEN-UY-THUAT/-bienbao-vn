"""
=============================================================================
HELPER FUNCTIONS - CÁC HÀM PHỤ TRỢ DÙNG CHUNG
=============================================================================
Mục đích: Tập trung các hàm tiện ích được sử dụng xuyên suốt dự án
         để dễ bảo trì và tái sử dụng

Các nhóm chức năng:
    1. Image Processing: Resize, Padding, Preprocessing
    2. Bounding Box: IoU, Clamp, Convert formats
    3. Visualization: Draw boxes, labels, FPS
    4. File Utilities: Path handling, Data loading
=============================================================================
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

# =============================================================================
# PROJECT PATHS - Đường dẫn dự án (Relative từ thư mục gốc Traffic_Sign_Project)
# =============================================================================

def get_project_root() -> Path:
    """
    Lấy đường dẫn gốc của dự án Traffic_Sign_Project.
    
    Returns:
        Path: Đường dẫn đến thư mục gốc dự án
    """
    # Utils/helper_functions.py -> Traffic_Sign_Project
    return Path(__file__).parent.parent


def get_paths() -> Dict[str, Path]:
    """
    Lấy tất cả đường dẫn quan trọng trong dự án.
    
    Returns:
        Dict với các key chính:
            - project_root: Thư mục gốc dự án Traffic_Sign_Project
            - parent: Thư mục cha (Project_DL)
            
        Data paths:
            - data_raw_root: Thư mục gốc chứa dữ liệu thô
            - yolo_raw_data: Dữ liệu YOLO gốc
            - cnn_raw_data: Dữ liệu CNN gốc
            
        Processed data paths:
            - data_processed_root: Thư mục gốc chứa dữ liệu đã xử lý
            - yolo_processed: Dữ liệu YOLO đã xử lý
            - cnn_processed: Dữ liệu CNN đã xử lý
            
        Pictures (visualization):
            - pictures_root: Thư mục gốc lưu ảnh visualization
            - yolo_pictures: Ảnh quá trình xử lý YOLO
            - cnn_pictures: Ảnh quá trình xử lý CNN
            
        Models:
            - yolo_model: File model YOLOv8
            - cnn_model: File model CNN
    """
    project_root = get_project_root()
    parent = project_root.parent  # Thư mục cha chứa Data, output_*
    
    # === DỮ LIỆU GỐC (RAW DATA) ===
    data_raw_root = parent / 'Data_Raw'
    
    # === DỮ LIỆU ĐÃ XỬ LÝ (PROCESSED DATA) ===
    data_processed_root = parent / 'Data_Processed'
    
    # === ẢNH VISUALIZATION (PICTURES) ===
    pictures_root = parent / 'Pictures'
    
    return {
        # === PATHS GỐC ===
        'project_root': project_root,
        'parent': parent,
        
        # === DỮ LIỆU GỐC (RAW) ===
        'data_raw_root': data_raw_root,
        'yolo_raw_data': data_raw_root / 'YOLO_Dataset',
        'cnn_raw_data': data_raw_root / 'CNN_Dataset',
        
        # === DỮ LIỆU ĐÃ XỬ LÝ (PROCESSED) ===
        'data_processed_root': data_processed_root,
        'yolo_processed': data_processed_root / 'YOLO_Processed',
        'cnn_processed': data_processed_root / 'CNN_Processed',
        
        # === ẢNH VISUALIZATION ===
        'pictures_root': pictures_root,
        'yolo_pictures': pictures_root / 'YOLO_Processing',
        'cnn_pictures': pictures_root / 'CNN_Processing',
        
        # === MODELS ===
        'yolo_model': project_root / 'Inference' / 'best.pt',
        'cnn_model': project_root / 'Inference' / 'traffic_sign_cnn.keras',
        
        # === LEGACY PATHS (Tương thích ngược với cấu trúc cũ) ===
        'yolo_raw_data_legacy': parent / 'Data' / 'yolo_dataset',
        'cnn_raw_data_legacy': parent / 'Data' / 'cnn_dataset',
        'yolo_processed_legacy': parent / 'output_Yolodataset',
        'cnn_processed_legacy': parent / 'output_CNNdataset',
        'yolo_model_legacy': parent / 'Model' / 'Yolo' / 'best.pt',
        'cnn_model_legacy': parent / 'Model' / 'CNN' / 'best_cnn_traffic_sign.keras',
    }


def get_yolo_picture_paths() -> Dict[str, Path]:
    """
    Lấy đường dẫn các thư mục lưu ảnh trong quá trình xử lý YOLO.
    
    Returns:
        Dict chứa các đường dẫn cho từng bước xử lý
    """
    paths = get_paths()
    yolo_pics = paths['yolo_pictures']
    
    return {
        'root': yolo_pics,
        '01_raw_samples': yolo_pics / '01_Raw_Samples',
        '02_class_distribution': yolo_pics / '02_Class_Distribution',
        '03_blur_detection': yolo_pics / '03_Blur_Detection',
        '04_duplicate_detection': yolo_pics / '04_Duplicate_Detection',
        '05_augmentation': yolo_pics / '05_Augmentation',
        '06_final_distribution': yolo_pics / '06_Final_Distribution',
        '07_sample_annotations': yolo_pics / '07_Sample_Annotations',
    }


def get_cnn_picture_paths() -> Dict[str, Path]:
    """
    Lấy đường dẫn các thư mục lưu ảnh trong quá trình xử lý CNN.
    
    Returns:
        Dict chứa các đường dẫn cho từng bước xử lý
    """
    paths = get_paths()
    cnn_pics = paths['cnn_pictures']
    
    return {
        'root': cnn_pics,
        '01_raw_samples': cnn_pics / '01_Raw_Samples',
        '02_class_distribution': cnn_pics / '02_Class_Distribution',
        '03_resize_padding': cnn_pics / '03_Resize_Padding',
        '04_duplicate_detection': cnn_pics / '04_Duplicate_Detection',
        '05_augmentation': cnn_pics / '05_Augmentation',
        '06_final_distribution': cnn_pics / '06_Final_Distribution',
        '07_sample_per_class': cnn_pics / '07_Sample_Per_Class',
    }


def setup_directories(create_all: bool = True) -> Dict[str, Any]:
    """
    Tạo tất cả thư mục cần thiết cho dự án.
    
    Args:
        create_all: Nếu True, tạo tất cả thư mục. Nếu False, chỉ trả về paths.
        
    Returns:
        Dict chứa tất cả đường dẫn đã tạo
    """
    paths = get_paths()
    yolo_pics = get_yolo_picture_paths()
    cnn_pics = get_cnn_picture_paths()
    
    all_dirs = []
    
    # Data directories
    all_dirs.extend([
        paths['data_raw_root'],
        paths['yolo_raw_data'],
        paths['cnn_raw_data'],
        paths['data_processed_root'],
        paths['yolo_processed'],
        paths['cnn_processed'],
        paths['pictures_root'],
        paths['yolo_pictures'],
        paths['cnn_pictures'],
    ])
    
    # YOLO picture directories
    all_dirs.extend(yolo_pics.values())
    
    # CNN picture directories
    all_dirs.extend(cnn_pics.values())
    
    # YOLO processed subdirectories
    yolo_proc = paths['yolo_processed']
    for subdir in ['train/images', 'train/labels', 'val/images', 'val/labels', 'test/images', 'test/labels']:
        all_dirs.append(yolo_proc / subdir)
    
    # CNN processed subdirectories (7 classes)
    cnn_proc = paths['cnn_processed']
    class_names = ['0_Cam_nguoc_chieu', '1_Cam_dung_va_do', '2_Cam_re', 
                   '3_Gioi_han_toc_do', '4_Cam_con_lai', '5_Nguy_hiem', '6_Hieu_lenh']
    for split in ['train', 'val', 'test']:
        for class_name in class_names:
            all_dirs.append(cnn_proc / split / class_name)
    
    if create_all:
        for dir_path in all_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Đã tạo {len(all_dirs)} thư mục")
    
    return {
        'main_paths': paths,
        'yolo_pictures': yolo_pics,
        'cnn_pictures': cnn_pics,
        'all_directories': all_dirs,
    }


# =============================================================================
# 1. IMAGE PROCESSING FUNCTIONS
# =============================================================================

def resize_with_padding(
    image: np.ndarray, 
    target_size: Tuple[int, int],
    pad_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Resize ảnh về kích thước target nhưng GIỮ NGUYÊN TỈ LỆ (Aspect Ratio).
    
    TẠI SAO CẦN PADDING?
    - Biển báo giao thông có hình dạng đặc trưng (tròn, tam giác, chữ nhật)
    - Nếu resize trực tiếp sẽ làm MÉMO hình dạng → Model học sai feature
    - Padding giữ nguyên tỉ lệ, thêm viền đen để đạt kích thước mong muốn
    
    Thuật toán:
    1. Tính tỉ lệ scale để fit vào target_size
    2. Resize ảnh theo tỉ lệ đó
    3. Thêm padding (viền đen) để đạt đúng target_size
    
    Args:
        image: Ảnh đầu vào (numpy array, BGR format)
        target_size: Tuple (width, height) mong muốn
        pad_color: Màu padding (mặc định đen)
        
    Returns:
        Ảnh đã resize với padding (numpy array)
    """
    if image is None or image.size == 0:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Tính tỉ lệ scale
    scale = min(target_w / w, target_h / h)
    
    # Resize ảnh
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Tạo canvas với padding
    canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    
    # Đặt ảnh vào giữa canvas
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return canvas


def preprocess_for_cnn(
    roi_bgr: np.ndarray,
    target_size: Tuple[int, int] = (64, 64),
    normalize: bool = True
) -> Optional[np.ndarray]:
    """
    Tiền xử lý vùng ảnh ROI để đưa vào CNN classifier.
    
    QUAN TRỌNG:
    - OpenCV đọc ảnh theo format BGR
    - CNN (Keras/TensorFlow) thường train với ảnh RGB
    - Phải chuyển đổi BGR → RGB trước khi predict
    
    Args:
        roi_bgr: Vùng ảnh cắt từ frame (BGR format, OpenCV)
        target_size: Kích thước output (width, height)
        normalize: Có normalize về [0, 1] hay không
    
    Returns:
        np.ndarray: Ảnh đã xử lý với shape (1, H, W, 3), hoặc None nếu lỗi
    """
    try:
        if roi_bgr is None or roi_bgr.size == 0:
            return None
        
        # Bước 1: Chuyển đổi BGR → RGB
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        
        # Bước 2: Resize về kích thước CNN yêu cầu
        roi_resized = cv2.resize(roi_rgb, target_size, interpolation=cv2.INTER_AREA)
        
        # Bước 3: Normalize pixel values từ [0, 255] → [0, 1]
        if normalize:
            roi_normalized = roi_resized.astype(np.float32) / 255.0
        else:
            roi_normalized = roi_resized.astype(np.float32)
        
        # Bước 4: Thêm batch dimension
        roi_batch = np.expand_dims(roi_normalized, axis=0)
        
        return roi_batch
    
    except Exception as e:
        print(f"⚠️ Lỗi preprocessing ROI: {e}")
        return None


def calculate_blur_score(image: np.ndarray) -> float:
    """
    Tính điểm độ mờ của ảnh bằng Laplacian variance.
    
    Ảnh càng mờ → variance càng thấp
    
    Args:
        image: Ảnh đầu vào (BGR hoặc Grayscale)
        
    Returns:
        float: Điểm Laplacian variance (cao = sắc nét, thấp = mờ)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


# =============================================================================
# 2. BOUNDING BOX UTILITIES
# =============================================================================

def clamp_bbox(
    x1: int, y1: int, x2: int, y2: int, 
    img_width: int, img_height: int
) -> Tuple[int, int, int, int]:
    """
    Giới hạn bounding box trong phạm vi ảnh.
    Xử lý trường hợp box bị tràn ra ngoài viền.
    
    Args:
        x1, y1, x2, y2: Tọa độ bounding box
        img_width, img_height: Kích thước ảnh
    
    Returns:
        tuple: (x1, y1, x2, y2) đã được clamp
    """
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(0, min(x2, img_width))
    y2 = max(0, min(y2, img_height))
    return x1, y1, x2, y2


def calculate_iou(box1: Tuple, box2: Tuple) -> float:
    """
    Tính Intersection over Union (IoU) giữa 2 bounding boxes.
    
    IoU = Area of Intersection / Area of Union
    
    Args:
        box1: (x1, y1, x2, y2) - Box thứ nhất
        box2: (x1, y1, x2, y2) - Box thứ hai
        
    Returns:
        float: Giá trị IoU từ 0 đến 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Tính intersection
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height
    
    # Tính union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


def convert_yolo_to_xyxy(
    yolo_box: Tuple[float, float, float, float],
    img_width: int, img_height: int
) -> Tuple[int, int, int, int]:
    """
    Chuyển đổi YOLO format (x_center, y_center, w, h) normalized
    sang (x1, y1, x2, y2) pixel coordinates.
    
    Args:
        yolo_box: (x_center, y_center, width, height) normalized [0,1]
        img_width, img_height: Kích thước ảnh
        
    Returns:
        (x1, y1, x2, y2) tọa độ pixel
    """
    x_center, y_center, w, h = yolo_box
    
    # Convert normalized to pixel
    x_center *= img_width
    y_center *= img_height
    w *= img_width
    h *= img_height
    
    # Convert center to corner format
    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = int(x_center + w / 2)
    y2 = int(y_center + h / 2)
    
    return x1, y1, x2, y2


def convert_xyxy_to_yolo(
    x1: int, y1: int, x2: int, y2: int,
    img_width: int, img_height: int
) -> Tuple[float, float, float, float]:
    """
    Chuyển đổi (x1, y1, x2, y2) pixel coordinates
    sang YOLO format (x_center, y_center, w, h) normalized.
    
    Args:
        x1, y1, x2, y2: Tọa độ pixel
        img_width, img_height: Kích thước ảnh
        
    Returns:
        (x_center, y_center, width, height) normalized [0,1]
    """
    w = x2 - x1
    h = y2 - y1
    x_center = x1 + w / 2
    y_center = y1 + h / 2
    
    # Normalize
    x_center /= img_width
    y_center /= img_height
    w /= img_width
    h /= img_height
    
    return x_center, y_center, w, h


# =============================================================================
# 3. VISUALIZATION FUNCTIONS
# =============================================================================

# Cấu hình mặc định cho visualization
DEFAULT_VIS_CONFIG = {
    'bbox_color': (0, 255, 0),      # Màu xanh lá cho bounding box (BGR)
    'bbox_thickness': 2,             # Độ dày viền box
    'text_color': (255, 255, 255),   # Màu trắng cho text
    'text_bg_color': (0, 0, 0),      # Màu đen cho nền text
    'font': cv2.FONT_HERSHEY_SIMPLEX,
    'font_scale': 0.6,
    'font_thickness': 2,
}


def draw_detection(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    confidence: float,
    config: Optional[Dict] = None
) -> None:
    """
    Vẽ bounding box và label lên frame.
    
    Args:
        frame: Frame ảnh để vẽ (sẽ được modify in-place)
        x1, y1, x2, y2: Tọa độ bounding box
        label: Tên class
        confidence: Độ tin cậy (0-1)
        config: Dict cấu hình visualization (optional)
    """
    cfg = {**DEFAULT_VIS_CONFIG, **(config or {})}
    
    # Vẽ bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), cfg['bbox_color'], cfg['bbox_thickness'])
    
    # Chuẩn bị text
    conf_percent = int(confidence * 100)
    text = f"{label} {conf_percent}%"
    
    # Tính kích thước text
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cfg['font'], cfg['font_scale'], cfg['font_thickness']
    )
    
    # Vị trí text (phía trên box)
    text_x = x1
    text_y = y1 - 10
    
    # Nếu text bị tràn lên trên ảnh, đặt xuống dưới box
    if text_y - text_height < 0:
        text_y = y2 + text_height + 10
    
    # Vẽ background cho text
    bg_x1 = text_x
    bg_y1 = text_y - text_height - 5
    bg_x2 = text_x + text_width + 10
    bg_y2 = text_y + 5
    
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), cfg['text_bg_color'], -1)
    
    # Vẽ text
    cv2.putText(
        frame, text, (text_x + 5, text_y),
        cfg['font'], cfg['font_scale'], cfg['text_color'], cfg['font_thickness']
    )


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """
    Vẽ FPS lên góc trên bên trái của frame.
    
    Args:
        frame: Frame ảnh
        fps: Giá trị FPS
    """
    fps_text = f"FPS: {fps:.1f}"
    
    # Vẽ background
    cv2.rectangle(frame, (10, 10), (150, 50), (0, 0, 0), -1)
    
    # Vẽ text FPS màu xanh lá
    cv2.putText(
        frame, fps_text, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )


# =============================================================================
# 4. FILE UTILITIES
# =============================================================================

def ensure_dir(path: Path) -> Path:
    """
    Đảm bảo thư mục tồn tại, tạo mới nếu chưa có.
    
    Args:
        path: Đường dẫn thư mục
        
    Returns:
        Path: Đường dẫn đã được đảm bảo tồn tại
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_image_files(directory: Path, extensions: Optional[List[str]] = None) -> List[Path]:
    """
    Lấy danh sách file ảnh trong thư mục.
    
    Args:
        directory: Đường dẫn thư mục
        extensions: List các extension cho phép
        
    Returns:
        List[Path]: Danh sách đường dẫn file ảnh
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    directory = Path(directory)
    image_files = []
    
    for ext in extensions:
        image_files.extend(directory.glob(f'*{ext}'))
        image_files.extend(directory.glob(f'*{ext.upper()}'))
    
    return sorted(image_files)


def load_yaml(path: Path) -> Dict:
    """
    Load file YAML.
    
    Args:
        path: Đường dẫn file YAML
        
    Returns:
        Dict: Nội dung file
    """
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict, path: Path) -> None:
    """
    Lưu data vào file YAML.
    
    Args:
        data: Dict cần lưu
        path: Đường dẫn file
    """
    import yaml
    ensure_dir(Path(path).parent)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


# =============================================================================
# 5. TRAFFIC SIGN SPECIFIC
# =============================================================================

# Định nghĩa các class biển báo
TRAFFIC_SIGN_CLASSES = {
    0: "Cam_nguoc_chieu",      # Cấm ngược chiều
    1: "Cam_dung_va_do",       # Cấm dừng và đỗ
    2: "Cam_re",               # Cấm rẽ
    3: "Gioi_han_toc_do",      # Giới hạn tốc độ
    4: "Cam_con_lai",          # Cấm còn lại
    5: "Nguy_hiem",            # Nguy hiểm
    6: "Hieu_lenh",            # Hiệu lệnh
}

# Mapping class ID sang Super-class (cho YOLO grouping)
YOLO_CLASS_MAPPING = {
    0: 0,  # Cam nguoc chieu -> Prohibitory
    1: 0,  # Cam dung va do -> Prohibitory  
    2: 0,  # Cam re -> Prohibitory
    3: 0,  # Gioi han toc do -> Prohibitory
    4: 0,  # Cam con lai -> Prohibitory
    5: 1,  # Nguy hiem -> Danger
    6: 2,  # Hieu lenh -> Mandatory
}

YOLO_SUPERCLASS_NAMES = ['Prohibitory', 'Danger', 'Mandatory', 'Other']


def get_class_name(class_id: int) -> str:
    """
    Lấy tên class từ ID.
    
    Args:
        class_id: ID của class (0-6)
        
    Returns:
        str: Tên class
    """
    return TRAFFIC_SIGN_CLASSES.get(class_id, f"Unknown_{class_id}")


def get_class_color(class_id: int) -> Tuple[int, int, int]:
    """
    Lấy màu hiển thị cho từng class (BGR format).
    
    Args:
        class_id: ID của class
        
    Returns:
        Tuple BGR color
    """
    colors = [
        (0, 0, 255),    # 0: Đỏ - Cấm ngược chiều
        (0, 128, 255),  # 1: Cam - Cấm dừng đỗ
        (0, 255, 255),  # 2: Vàng - Cấm rẽ
        (255, 0, 0),    # 3: Xanh dương - Giới hạn tốc độ
        (128, 0, 128),  # 4: Tím - Cấm còn lại
        (0, 255, 0),    # 5: Xanh lá - Nguy hiểm
        (255, 255, 0),  # 6: Cyan - Hiệu lệnh
    ]
    return colors[class_id % len(colors)]


# =============================================================================
# 6. SAVE FIGURES & VISUALIZATION
# =============================================================================

def save_figure(
    fig,
    filename: str,
    save_dir: Path,
    dpi: int = 150,
    close_fig: bool = True
) -> Path:
    """
    Lưu matplotlib figure vào thư mục với tên rõ ràng.
    
    Args:
        fig: matplotlib figure object
        filename: Tên file (không cần extension)
        save_dir: Thư mục lưu
        dpi: Độ phân giải
        close_fig: Đóng figure sau khi lưu
        
    Returns:
        Path: Đường dẫn file đã lưu
    """
    import matplotlib.pyplot as plt
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Đảm bảo filename có extension
    if not filename.endswith('.png'):
        filename = f"{filename}.png"
    
    filepath = save_dir / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    
    if close_fig:
        plt.close(fig)
    
    print(f"📊 Saved: {filepath}")
    return filepath


def save_image(
    image: np.ndarray,
    filename: str,
    save_dir: Path,
    is_bgr: bool = True
) -> Path:
    """
    Lưu ảnh numpy array vào thư mục.
    
    Args:
        image: Ảnh numpy array
        filename: Tên file (không cần extension)
        save_dir: Thư mục lưu
        is_bgr: True nếu ảnh đang ở format BGR (OpenCV default)
        
    Returns:
        Path: Đường dẫn file đã lưu
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if not filename.endswith(('.png', '.jpg', '.jpeg')):
        filename = f"{filename}.png"
    
    filepath = save_dir / filename
    
    if is_bgr:
        cv2.imwrite(str(filepath), image)
    else:
        # Convert RGB to BGR for saving
        cv2.imwrite(str(filepath), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    print(f"🖼️ Saved: {filepath}")
    return filepath


def save_comparison_image(
    images: List[np.ndarray],
    titles: List[str],
    filename: str,
    save_dir: Path,
    figsize: Tuple[int, int] = (15, 5)
) -> Path:
    """
    Lưu ảnh so sánh nhiều ảnh cạnh nhau.
    
    Args:
        images: List các ảnh (BGR format)
        titles: List tiêu đề cho từng ảnh
        filename: Tên file
        save_dir: Thư mục lưu
        figsize: Kích thước figure
        
    Returns:
        Path: Đường dẫn file đã lưu
    """
    import matplotlib.pyplot as plt
    
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    
    if n == 1:
        axes = [axes]
    
    for ax, img, title in zip(axes, images, titles):
        # Convert BGR to RGB for display
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        ax.imshow(img_rgb)
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    return save_figure(fig, filename, save_dir)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("✅ Helper Functions Module Loaded Successfully!")
    print("=" * 60)
    
    # Test get_paths
    paths = get_paths()
    print("\n📁 Main Project Paths:")
    for name in ['project_root', 'parent', 'data_raw_root', 'data_processed_root', 'pictures_root']:
        path = paths.get(name)
        if path:
            exists = "✅" if path.exists() else "❌"
            print(f"   {exists} {name}: {path}")
    
    print("\n📁 YOLO Picture Paths:")
    yolo_pics = get_yolo_picture_paths()
    for name, path in yolo_pics.items():
        exists = "✅" if path.exists() else "❌"
        print(f"   {exists} {name}: {path.name}")
    
    print("\n📁 CNN Picture Paths:")
    cnn_pics = get_cnn_picture_paths()
    for name, path in cnn_pics.items():
        exists = "✅" if path.exists() else "❌"
        print(f"   {exists} {name}: {path.name}")
    
    print("\n📋 Traffic Sign Classes:")
    for class_id, class_name in TRAFFIC_SIGN_CLASSES.items():
        print(f"   {class_id}: {class_name}")
    
    # Tạo thư mục nếu chưa có
    print("\n" + "=" * 60)
    user_input = input("Bạn có muốn tạo tất cả thư mục? (y/n): ")
    if user_input.lower() == 'y':
        setup_directories(create_all=True)
