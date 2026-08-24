"""
Module LLM Traffic Profile: Tích hợp Groq API với mô hình openai/gpt-oss-20b.
Chức năng:
- Tiếp nhận số liệu giám sát nút giao thời gian thực (Lưu lượng, Mật độ OCR, Vận tốc, Điểm CI).
- Sử dụng openai/gpt-oss-20b để suy luận và đưa ra khuyến nghị điều khiển pha đèn giao thông thích ứng.
- Trả về bản Traffic Control Profile có cấu trúc rõ ràng dạng Markdown.
"""

import os
from typing import Dict, Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class TrafficAdvisor:
    """Tạo khuyến nghị điều khiển đèn tín hiệu giao thông thông minh từ LLM (openai/gpt-oss-20b)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "openai/gpt-oss-20b"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None

        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                print(f"⚠️ Không thể khởi tạo Groq Client: {e}")

    def is_available(self) -> bool:
        """Kiểm tra xem LLM API client đã sẵn sàng chưa."""
        return self.client is not None

    def generate_control_profile(
        self,
        density_stats: Dict[str, Any],
        vehicle_counts: Dict[str, int]
    ) -> str:
        """Sinh Traffic Control Profile từ dữ liệu giám sát thời gian thực."""
        if not self.is_available():
            return self._generate_fallback_template(density_stats, vehicle_counts)

        ci = density_stats.get("congestion_index", 0.0)
        ocr = density_stats.get("occupancy_rate", 0.0)
        avg_speed = density_stats.get("avg_speed", density_stats.get("avg_speed_kmh", 0.0))
        stopped_ratio = density_stats.get("stopped_ratio", 0.0)
        vehicle_count = density_stats.get("vehicle_count", 0)

        prompt = f"""
Bạn là chuyên gia hàng đầu về phân tích và điều khiển hệ thống đèn tín hiệu giao thông thông minh (Adaptive Traffic Signal Control).
Dưới đây là số liệu giám sát thời gian thực thu thập từ camera UAV tại nút giao:

=== THỐNG KÊ GIÁM SÁT DÒNG XE TỪ UAV ===
- Số phương tiện hiện diện trong vùng nút giao: {vehicle_count} xe
- Chi tiết phương tiện đã đếm: {vehicle_counts}
- Tỷ lệ chiếm dụng mặt đường (OCR): {ocr}%
- Vận tốc trung bình dòng xe: {avg_speed} km/h
- Tỷ lệ xe dừng chờ: {stopped_ratio}%
- Chỉ số tắc nghẽn tổng hợp (Congestion Index - CI): {ci}/100 ({density_stats.get('congestion_level', 'N/A')})

HÃY TẠO BẢN 'TRAFFIC CONTROL PROFILE' BẰNG TIẾNG VIỆT GỒM 3 MỤC:
1. Đánh giá tình trạng giao thông (ngắn gọn, định lượng).
2. Khuyến nghị điều chỉnh thời gian đèn tín hiệu (thêm/bớt bao nhiêu giây cho từng hướng Bắc-Nam vs Đông-Tây).
3. Bảng chu kỳ đèn đề xuất (Thời gian Xanh/Vàng/Đỏ cụ thể cho từng hướng Hướng Chính (Bắc - Nam) và Hướng Phụ (Đông - Tây), tổng chu kỳ 60s - 90s).
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content.strip()
            return self._generate_fallback_template(density_stats, vehicle_counts)
        except Exception as e:
            return (
                f"⚠️ Lỗi kết nối LLM API ({e}). Đang hiển thị khuyến nghị từ luật suy diễn cục bộ:\n\n"
                + self._generate_fallback_template(density_stats, vehicle_counts)
            )

    def _generate_fallback_template(self, stats: Dict[str, Any], counts: Dict[str, int]) -> str:
        """Sinh bản khuyến nghị dựa trên luật chuyên gia cục bộ khi không có kết nối API."""
        ci = stats.get("congestion_index", 20.0)
        ocr = stats.get("occupancy_rate", 15.0)

        if ci < 30:
            main_green, sub_green, total_cycle = 35, 20, 60
            advice = "Dòng xe thông thoáng, duy trì chu kỳ đèn cơ bản."
        elif ci < 60:
            main_green, sub_green, total_cycle = 40, 22, 68
            advice = "Mật độ trung bình, tăng nhẹ +5s đèn xanh hướng chính."
        elif ci < 80:
            main_green, sub_green, total_cycle = 50, 18, 75
            advice = "Dòng xe đông đúc, ưu tiên kéo dài +15s đèn xanh hướng chính để giải tỏa hàng đợi."
        else:
            main_green, sub_green, total_cycle = 60, 15, 85
            advice = "Ùn tắc nghiêm trọng! Ưu tiên tối đa đèn xanh hướng chính (+25s) và kích hoạt cảnh báo điều phối."

        return f"""### 📊 TRAFFIC CONTROL PROFILE — KHUYẾN NGHỊ ĐIỀU KHIỂN NÚT GIAO

#### 1. 🔍 Đánh Giá Tình Trạng
* **Chỉ số tắc nghẽn (CI):** `{ci:.0f}/100` ({stats.get('congestion_level', 'Bình thường')})
* **Tỷ lệ chiếm dụng mặt đường:** `{ocr:.1f}%`
* **Nhận định:** {advice}

#### 2. ⚙️ Khuyến Nghị Điều Chỉnh Pha Đèn (Tổng chu kỳ: {total_cycle}s)
* **Pha đèn Hướng Chính (Bắc - Nam):** 🟢 **Xanh {main_green}s** | 🟡 Vàng 3s | 🔴 Đỏ {total_cycle - main_green - 3}s
* **Pha đèn Hướng Phụ (Đông - Tây):** 🟢 **Xanh {sub_green}s** | 🟡 Vàng 3s | 🔴 Đỏ {total_cycle - sub_green - 3}s
"""


# Alias để tương thích ngược với các file import cũ
GroqTrafficAdvisor = TrafficAdvisor


def generate_traffic_profile(metrics: Dict[str, Any]) -> str:
    """Hàm tiện ích sinh Traffic Control Profile từ dict metrics tổng hợp."""
    advisor = TrafficAdvisor()
    counts = metrics.get("counts", {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0})
    density_stats = {
        "vehicle_count": metrics.get("vehicles_roi", metrics.get("vehicle_count", sum(counts.values()))),
        "occupancy_rate": metrics.get("occupancy_rate", 0.0),
        "avg_speed": metrics.get("avg_speed", 0.0),
        "avg_speed_kmh": metrics.get("avg_speed", 0.0),
        "stopped_ratio": metrics.get("stopped_ratio", 0.0),
        "congestion_index": metrics.get("congestion_index", 0.0),
        "congestion_level": "Đông đúc" if metrics.get("congestion_index", 0.0) >= 60 else "Thông thoáng"
    }
    return advisor.generate_control_profile(density_stats, counts)
