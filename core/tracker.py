"""
Module Tracker: Theo dõi đa đối tượng phương tiện (Multi-Object Tracking).
Hỗ trợ ByteTrack và BoT-SORT (tích hợp Camera Motion Compensation - CMC).
"""

from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import numpy as np
from ultralytics import YOLO

TRACKER_CONFIG_MAP = {
    "ByteTrack": "bytetrack.yaml",
    "BoT-SORT": "botsort.yaml",
}


def normalize_class_name(raw_name: str) -> str:
    """Chuẩn hóa tên nhãn class phương tiện."""
    name = raw_name.lower().strip()
    return "motorcycle" if name in ("motor", "motorcycle", "motorbike") else name


class VehicleTracker:
    """Quản lý bám vết (MOT) và lưu lịch sử quỹ đạo phương tiện thời gian thực."""

    def __init__(
        self,
        model_path: str = "models/best.pt",
        tracker_name: str = "ByteTrack",
        conf: float = 0.25,
        iou: float = 0.45,
        max_trail_len: int = 30,
        device: str = ""
    ):
        self.model = YOLO(model_path)
        self.tracker_name = tracker_name
        self.tracker_config = TRACKER_CONFIG_MAP.get(tracker_name, tracker_name if tracker_name.endswith(".yaml") else "bytetrack.yaml")
        self.conf = conf
        self.iou = iou
        self.device = device if device else None
        self.max_trail_len = max_trail_len

        self.tracks_history: Dict[int, deque] = {}
        self.unique_track_ids: set = set()

    def track_frame(self, frame: np.ndarray, imgsz: int = 640) -> Tuple[List[Dict[str, Any]], Any]:
        """Chạy bám vết (Detection + MOT) trên một frame."""
        results = self.model.track(
            source=frame,
            imgsz=imgsz,
            persist=True,
            tracker=self.tracker_config,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False
        )[0]

        tracked_objects: List[Dict[str, Any]] = []

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().numpy()
            classes = results.boxes.cls.int().cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            for box, track_id, cls_id, conf_val in zip(boxes, track_ids, classes, confs):
                x1, y1, x2, y2 = map(int, box)
                tid = int(track_id)
                self.unique_track_ids.add(tid)

                raw_name = self.model.names.get(int(cls_id), f"class_{cls_id}")
                cls_name = normalize_class_name(raw_name)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if tid not in self.tracks_history:
                    self.tracks_history[tid] = deque(maxlen=self.max_trail_len)
                self.tracks_history[tid].append((cx, cy))

                tracked_objects.append({
                    "id": tid,
                    "class": cls_name,
                    "bbox": [x1, y1, x2, y2],
                    "center": (cx, cy),
                    "conf": float(conf_val),
                    "trail": list(self.tracks_history[tid])
                })

        return tracked_objects, results

    def reset(self):
        """Reset lịch sử bám vết."""
        self.tracks_history.clear()
        self.unique_track_ids.clear()
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            if hasattr(self.model.predictor, "trackers") and self.model.predictor.trackers:
                for tr in self.model.predictor.trackers:
                    if hasattr(tr, "reset"):
                        tr.reset()
