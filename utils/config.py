"""Cấu hình dùng chung cho hệ thống."""
from typing import Dict, Tuple

# Bảng màu cao cấp, dịu mắt và chuyên nghiệp (BGR cho OpenCV)
CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "car": (248, 189, 56),        # Cyan #38bdf8
    "motorcycle": (129, 185, 16),  # Emerald #10b981
    "bus": (11, 158, 245),         # Amber #f59e0b
    "truck": (252, 132, 192),      # Purple #c084fc
    "bicycle": (53, 230, 163),     # Lime #a3e635
}

# Cấu hình mặc định
DEFAULT_MODEL_PATH = "models/best.pt"
DEFAULT_SAMPLE_VIDEO = "videos/DJI_20250516071323_0341_D.MP4"

# Danh mục video UAV mẫu có sẵn
SAMPLE_VIDEOS = {
    "Ngã tư giao lộ đô thị 4K (DJI UAV)": "videos/DJI_20250516071323_0341_D.MP4",
    "Đại lộ giao thông mật độ cao (DJI UAV)": "videos/DJI_20250516070629_0331_D.MP4",
    "Giao lộ phân luồng phức tạp (DJI UAV)": "videos/DJI_20250516075621_0362_D.MP4"
}
