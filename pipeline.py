"""
Pipeline chính tích hợp toàn bộ các module:
Video -> VehicleTracker -> VehicleCounter -> TrafficDensityAnalyzer -> TrafficAlertEngine -> Overlay Frame
"""

import time
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import cv2

from core.tracker import VehicleTracker
from core.counter import VehicleCounter
from core.analyzer import TrafficDensityAnalyzer
from core.alert import TrafficAlertEngine
from utils.drawing import draw_vehicle_tracking_overlay


class UAVTrafficPipeline:
    """Pipeline tổng xử lý video UAV theo từng frame."""

    def __init__(
        self,
        model_path: str = "models/best.pt",
        tracker_name: str = "BoT-SORT",
        conf: float = 0.25,
        iou: float = 0.45,
        line_y: int = 700,
        device: str = "",
        pixels_per_meter: float = 10.0,
        roi_polygon: Optional[List[Tuple[int, int]]] = None
    ):
        self.tracker = VehicleTracker(
            model_path=model_path,
            tracker_name=tracker_name,
            conf=conf,
            iou=iou,
            device=device
        )
        self.counter = VehicleCounter(line_y=line_y)
        self.analyzer = TrafficDensityAnalyzer(
            pixels_per_meter=pixels_per_meter,
            roi_polygon=roi_polygon
        )
        self.alert_engine = TrafficAlertEngine()
        self.line_y = line_y
        self.roi_polygon = roi_polygon

    def process_frame(self, frame: np.ndarray, imgsz: int = 640) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Xử lý 1 frame hình ảnh và trả về frame vẽ overlay + thống kê chi tiết.

        Args:
            frame: Ảnh numpy BGR từ OpenCV.
            imgsz: Kích thước ảnh truyền vào YOLO.

        Returns:
            Tuple: (annotated_frame, payload_stats)
        """
        h, w = frame.shape[:2]

        # Đồng bộ tọa độ vạch đếm vào bộ đếm
        self.counter.set_line_y(self.line_y)

        # 1. Detection + Tracking
        t0 = time.perf_counter()
        tracked_objects, _ = self.tracker.track_frame(frame, imgsz=imgsz)
        infer_latency_ms = (time.perf_counter() - t0) * 1000

        # 2. Đếm phương tiện (Hiện diện tức thời + Qua vạch ảo)
        active_counts = self.counter.get_active_counts(tracked_objects)
        line_counts = self.counter.update(tracked_objects)
        total_line_crossed = self.counter.get_total_count()
        total_unique_seen = self.counter.get_total_unique_seen()

        # 3. Phân tích mật độ và chỉ số tắc nghẽn (Vận tốc km/h, ROI full màn)
        density_stats = self.analyzer.analyze_frame(tracked_objects, (h, w))
        density_stats["total_counted"] = total_line_crossed
        density_stats["total_unique"] = total_unique_seen
        density_stats["latency_ms"] = infer_latency_ms
        density_stats["fps"] = 1000.0 / infer_latency_ms if infer_latency_ms > 0 else 0.0

        # 4. Kiểm tra cảnh báo
        alerts = self.alert_engine.check_alerts(density_stats)

        # 5. Vẽ trực quan hóa lên frame
        annotated_frame = draw_vehicle_tracking_overlay(
            frame=frame,
            tracked_objects=tracked_objects,
            line_y=self.line_y,
            show_trail=True,
            show_labels=True,
            roi_polygon=self.roi_polygon
        )

        payload = {
            "tracked_objects": tracked_objects,
            "counts": active_counts,
            "line_counts": line_counts,
            "density_stats": density_stats,
            "alerts": alerts
        }

        return annotated_frame, payload

    def reset(self):
        """Reset toàn bộ pipeline."""
        self.tracker.reset()
        self.counter.reset()
        self.analyzer.reset()

