"""
Module Alert: Động cơ kiểm tra ngưỡng và phát cảnh báo giao thông.
"""

from typing import Dict, List, Any


class TrafficAlertEngine:
    """Động cơ phát cảnh báo trạng thái bất thường của dòng xe."""

    def __init__(
        self,
        congestion_threshold: float = 60.0,
        stop_ratio_threshold: float = 50.0,
        occupancy_threshold: float = 65.0
    ):
        self.congestion_threshold = congestion_threshold
        self.stop_ratio_threshold = stop_ratio_threshold
        self.occupancy_threshold = occupancy_threshold

    def check_alerts(self, stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Kiểm tra các chỉ số thống kê và sinh cảnh báo tương ứng."""
        alerts = []
        ci = stats.get("congestion_index", 0.0)
        r_stop = stats.get("stopped_ratio", 0.0)
        ocr = stats.get("occupancy_rate", 0.0)
        num_v = stats.get("vehicle_count", 0)

        if ci >= 80.0:
            alerts.append({
                "type": "SEVERE_CONGESTION",
                "message": f"ÙN TẮC NGHIÊM TRỌNG: Chỉ số tắc nghẽn đạt {ci:.0f}/100!",
                "severity": "critical"
            })
        elif ci >= self.congestion_threshold:
            alerts.append({
                "type": "MODERATE_CONGESTION",
                "message": f"Dòng xe đông đúc, nguy cơ ùn tắc tại giao lộ (CI: {ci:.0f}/100).",
                "severity": "warning"
            })

        if r_stop >= self.stop_ratio_threshold and num_v >= 5:
            alerts.append({
                "type": "GRIDLOCK",
                "message": f"Phát hiện {r_stop:.0f}% phương tiện đang phải dừng chờ kéo dài.",
                "severity": "critical"
            })

        if ocr >= self.occupancy_threshold:
            alerts.append({
                "type": "HIGH_OCCUPANCY",
                "message": f"Lòng đường bị chiếm dụng {ocr:.1f}% diện tích.",
                "severity": "warning"
            })

        return alerts
