"""
Module Detector: Wrapper phát hiện phương tiện bằng mô hình YOLO (Ultralytics).
"""

from typing import Any
import numpy as np
from ultralytics import YOLO


class VehicleDetector:
    """Class wrapper cho YOLO phục vụ phát hiện phương tiện giao thông."""

    def __init__(self, model_path: str = "models/best.pt", conf: float = 0.25, iou: float = 0.45, device: str = ""):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.device = device if device else None

    def detect(self, frame: np.ndarray, imgsz: int = 640) -> Any:
        """Chạy suy luận detection trên một frame."""
        return self.model.predict(
            source=frame,
            imgsz=imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False
        )[0]
