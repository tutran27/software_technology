"""
Module Analyzer: Phân tích mật độ, vận tốc dòng xe (km/h) và Chỉ số Tắc Nghẽn (CI).
Công thức tính:
- OCR = (Tổng diện tích BBox xe trong ROI / Diện tích ROI) * 100%
- Vận tốc: v_i (km/h) = (khoảng cách_px / dt / pixels_per_meter) * 3.6
- R_stop = (Số xe có v_i < ngưỡng dừng / Tổng số xe) * 100%
- CI = 0.4 * OCR + 0.3 * (1 - v_avg / v_max) * 100 + 0.3 * R_stop
"""

import time
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import numpy as np
import cv2


class TrafficDensityAnalyzer:
    """Phân tích mật độ và trạng thái tắc nghẽn giao thông từ dữ liệu video UAV."""

    def __init__(
        self,
        stop_speed_threshold_kmh: float = 5.0,
        max_speed_reference_kmh: float = 60.0,
        pixels_per_meter: float = 10.0,
        roi_polygon: Optional[List[Tuple[int, int]]] = None
    ):
        self.stop_threshold_kmh = stop_speed_threshold_kmh
        self.max_speed_ref_kmh = max_speed_reference_kmh
        self.pixels_per_meter = max(1.0, pixels_per_meter)
        self.roi_polygon = roi_polygon

        self.position_history: Dict[int, deque] = {}
        self.entry_times: Dict[int, float] = {}

    def set_roi_polygon(self, polygon: Optional[List[Tuple[int, int]]]):
        """Cập nhật tọa độ vùng ROI."""
        self.roi_polygon = polygon

    def analyze_frame(
        self,
        tracked_objects: List[Dict[str, Any]],
        frame_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Tính toán toàn bộ các chỉ số mật độ trên frame hiện tại."""
        now = time.time()
        frame_h, frame_w = frame_shape[:2]

        if self.roi_polygon is not None and len(self.roi_polygon) >= 3:
            roi_pts = np.array(self.roi_polygon, dtype=np.int32)
            roi_area = float(cv2.contourArea(roi_pts))
        else:
            roi_pts = None
            roi_area = float(frame_h * frame_w)

        if roi_area <= 0:
            roi_area = float(frame_h * frame_w)

        total_bbox_area = 0.0
        speeds_kmh = []
        stopped_count = 0
        dwell_times = []
        current_ids = set()
        vehicles_in_roi = 0

        for obj in tracked_objects:
            tid = obj["id"]
            x1, y1, x2, y2 = obj["bbox"]
            cx, cy = obj["center"]

            if roi_pts is not None:
                if cv2.pointPolygonTest(roi_pts, (float(cx), float(cy)), False) < 0:
                    continue

            vehicles_in_roi += 1
            current_ids.add(tid)
            total_bbox_area += max(0, x2 - x1) * max(0, y2 - y1)

            if tid not in self.entry_times:
                self.entry_times[tid] = now
            dwell_times.append(now - self.entry_times[tid])

            if tid not in self.position_history:
                self.position_history[tid] = deque(maxlen=10)
            self.position_history[tid].append((now, cx, cy))

            if len(self.position_history[tid]) >= 2:
                t_old, x_old, y_old = self.position_history[tid][0]
                dt = now - t_old
                if dt > 0.05:
                    dist_px = np.sqrt((cx - x_old)**2 + (cy - y_old)**2)
                    speed_kmh = min(120.0, (dist_px / dt / self.pixels_per_meter) * 3.6)
                    speeds_kmh.append(speed_kmh)
                    if speed_kmh < self.stop_threshold_kmh:
                        stopped_count += 1

        # Xóa xe đã rời khỏi frame
        for eid in set(self.entry_times.keys()) - current_ids:
            self.entry_times.pop(eid, None)
            self.position_history.pop(eid, None)

        ocr = min(100.0, (total_bbox_area / roi_area) * 100.0) if roi_area > 0 else 0.0
        avg_speed_kmh = float(np.mean(speeds_kmh)) if speeds_kmh else 0.0
        stopped_ratio = (stopped_count / vehicles_in_roi * 100.0) if vehicles_in_roi > 0 else 0.0
        avg_dwell = float(np.mean(dwell_times)) if dwell_times else 0.0

        speed_factor = max(0.0, 1.0 - (avg_speed_kmh / self.max_speed_ref_kmh)) * 100.0
        ci = min(100.0, max(0.0, 0.4 * ocr + 0.3 * speed_factor + 0.3 * stopped_ratio))

        if ci < 30:
            level = "Thông thoáng"
        elif ci < 60:
            level = "Bình thường"
        elif ci < 80:
            level = "Đông đúc"
        else:
            level = "Ùn tắc nghiêm trọng"

        return {
            "vehicle_count": vehicles_in_roi,
            "occupancy_rate": round(ocr, 1),
            "avg_speed": round(avg_speed_kmh, 1),
            "avg_speed_kmh": round(avg_speed_kmh, 1),
            "stopped_ratio": round(stopped_ratio, 1),
            "avg_dwell_time": round(avg_dwell, 1),
            "congestion_index": round(ci, 1),
            "congestion_level": level,
            "roi_area_px": roi_area,
            "timestamp": now
        }

    def reset(self):
        """Xóa trạng thái tính toán."""
        self.position_history.clear()
        self.entry_times.clear()
