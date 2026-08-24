"""
Tiện ích trực quan hóa OpenCV cho hệ thống giám sát giao thông UAV.
"""

from typing import List, Dict, Tuple, Any, Optional
import cv2
import numpy as np
from utils.config import CLASS_COLORS_BGR


def _draw_single_box(
    img: np.ndarray,
    bbox: List[int],
    cls_name: str,
    color: Tuple[int, int, int],
    track_id: Optional[int] = None,
    conf_val: Optional[float] = None,
    show_labels: bool = True
):
    """Hàm helper vẽ 1 bounding box kèm nhãn."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    if show_labels:
        parts = []
        if track_id is not None:
            parts.append(f"#{track_id}")
        parts.append(cls_name.upper())
        if conf_val is not None:
            parts.append(f"{conf_val:.2f}")
        label = " ".join(parts)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        box_y1 = max(0, y1 - th - 6)
        cv2.rectangle(img, (x1, box_y1), (x1 + tw + 6, y1), (15, 23, 42), -1)
        cv2.rectangle(img, (x1, box_y1), (x1 + tw + 6, y1), color, 1)
        cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_vehicle_boxes(
    frame: np.ndarray,
    detections: Optional[List[Dict[str, Any]]] = None,
    boxes: Any = None,
    model_names: Optional[Dict[int, str]] = None,
    show_conf: bool = True,
    show_labels: bool = True
) -> np.ndarray:
    """Vẽ bounding boxes của các phương tiện lên frame."""
    img = frame.copy()

    if detections is not None:
        for item in detections:
            cls_name = "motorcycle" if str(item.get("class", "car")).lower() in ("motor", "motorcycle") else str(item.get("class", "car")).lower()
            color = CLASS_COLORS_BGR.get(cls_name, (255, 128, 0))
            _draw_single_box(
                img,
                list(map(int, item.get("bbox", [0, 0, 0, 0]))),
                cls_name,
                color,
                track_id=item.get("id"),
                conf_val=float(item["conf"]) if show_conf and "conf" in item else None,
                show_labels=show_labels
            )
    elif boxes is not None and len(boxes) > 0:
        names = model_names or {}
        xyxy = boxes.xyxy.cpu().numpy()
        cls_arr = boxes.cls.int().cpu().numpy() if boxes.cls is not None else np.zeros(len(boxes))
        conf_arr = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(boxes))
        id_arr = boxes.id.int().cpu().numpy() if boxes.id is not None else [None] * len(boxes)

        for i in range(len(boxes)):
            raw_name = names.get(int(cls_arr[i]), f"class_{int(cls_arr[i])}").lower()
            cls_name = "motorcycle" if raw_name in ("motor", "motorcycle") else raw_name
            color = CLASS_COLORS_BGR.get(cls_name, (255, 128, 0))
            _draw_single_box(
                img,
                list(map(int, xyxy[i])),
                cls_name,
                color,
                track_id=int(id_arr[i]) if id_arr[i] is not None else None,
                conf_val=float(conf_arr[i]) if show_conf else None,
                show_labels=show_labels
            )
    return img


def draw_hud(
    frame: np.ndarray,
    hud_text: str = "UAV TRAFFIC CONTROL",
    fps: float = 0.0,
    vehicle_count: int = 0
) -> np.ndarray:
    """Vẽ thanh thông tin HUD trên góc frame."""
    img = frame.copy()
    overlay_text = f"{hud_text}"
    if fps > 0:
        overlay_text += f" | FPS: {fps:.1f}"
    if vehicle_count > 0:
        overlay_text += f" | VEHICLES: {vehicle_count}"

    cv2.rectangle(img, (10, 10), (min(img.shape[1] - 10, 480), 42), (15, 23, 42), -1)
    cv2.rectangle(img, (10, 10), (min(img.shape[1] - 10, 480), 42), (56, 189, 248), 1)
    cv2.putText(img, overlay_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 189, 248), 1, cv2.LINE_AA)
    return img


def draw_vehicle_tracking_overlay(
    frame: np.ndarray,
    tracked_objects: List[Dict[str, Any]],
    line_y: int = 700,
    show_trail: bool = True,
    show_labels: bool = True,
    roi_polygon: Optional[List[Tuple[int, int]]] = None
) -> np.ndarray:
    """Vẽ bounding boxes, trails, vạch đếm ảo và khung ROI lên frame."""
    img = frame.copy()
    h, w = img.shape[:2]

    # 1. Vẽ Khung ROI
    if roi_polygon is not None and len(roi_polygon) >= 3:
        pts = np.array(roi_polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(56, 189, 248), thickness=2, lineType=cv2.LINE_AA)
    else:
        # 4 góc định vị Cyber HUD
        c_len = min(40, w // 20)
        c_color = (56, 189, 248)
        for (gx, gy, dx, dy) in [(10, 10, 1, 1), (w - 10, 10, -1, 1), (10, h - 10, 1, -1), (w - 10, h - 10, -1, -1)]:
            cv2.line(img, (gx, gy), (gx + dx * c_len, gy), c_color, 2, cv2.LINE_AA)
            cv2.line(img, (gx, gy), (gx, gy + dy * c_len), c_color, 2, cv2.LINE_AA)

    # 2. Vẽ Vạch Đếm Ảo
    if 0 < line_y < h:
        cv2.line(img, (0, line_y), (w, line_y), (10, 10, 10), 4, cv2.LINE_AA)
        cv2.line(img, (0, line_y), (w, line_y), (0, 225, 255), 2, cv2.LINE_AA)
        cv2.rectangle(img, (16, line_y - 24), (260, line_y - 4), (15, 23, 42), -1)
        cv2.rectangle(img, (16, line_y - 24), (260, line_y - 4), (0, 225, 255), 1)
        cv2.putText(img, f"VIRTUAL COUNTING LINE (Y={line_y})", (22, line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 225, 255), 1, cv2.LINE_AA)

    # 3. Vẽ Bounding Box & Trail từng xe
    for obj in tracked_objects:
        tid = obj["id"]
        cls_name = "motorcycle" if obj["class"].lower() in ("motor", "motorcycle") else obj["class"].lower()
        color = CLASS_COLORS_BGR.get(cls_name, (255, 128, 0))

        if show_trail and "trail" in obj and len(obj["trail"]) > 1:
            trail_pts = np.array(obj["trail"], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [trail_pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

        _draw_single_box(img, obj["bbox"], cls_name, color, track_id=tid, show_labels=show_labels)

    return img
