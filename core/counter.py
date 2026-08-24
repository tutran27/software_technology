"""
Module Counter: Đếm số lượng phương tiện vượt qua vạch ảo và thống kê xe hiện diện.
"""

from typing import Dict, Set, Tuple, List, Any


class VehicleCounter:
    """Bộ đếm phương tiện giao thông qua vạch ảo và trong khung hình thời gian thực."""

    def __init__(self, line_y: int = 700):
        self.line_y = line_y
        self.counted_ids: Set[int] = set()
        self.prev_positions: Dict[int, Tuple[int, int]] = {}
        self.counts: Dict[str, int] = {
            "car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "bicycle": 0
        }
        self.all_seen_ids: Set[int] = set()

    def set_line_y(self, line_y: int):
        """Cập nhật tọa độ Y của vạch đếm ảo."""
        self.line_y = line_y

    def update(self, tracked_objects: List[Dict[str, Any]]) -> Dict[str, int]:
        """Cập nhật tọa độ và kiểm tra xe vượt qua vạch đếm."""
        for obj in tracked_objects:
            tid = obj["id"]
            cls_name = obj.get("class", "car")
            cx, cy = obj["center"]
            self.all_seen_ids.add(tid)

            if tid in self.counted_ids:
                self.prev_positions[tid] = (cx, cy)
                continue

            prev_pos = self.prev_positions.get(tid)
            self.prev_positions[tid] = (cx, cy)

            if prev_pos is not None:
                _, prev_y = prev_pos
                crossed = min(prev_y, cy) <= self.line_y <= max(prev_y, cy)
                if not crossed and "bbox" in obj:
                    _, y1, _, y2 = obj["bbox"]
                    if (prev_y < self.line_y and y2 >= self.line_y) or (prev_y > self.line_y and y1 <= self.line_y):
                        crossed = True

                if crossed:
                    self.counted_ids.add(tid)
                    self.counts[cls_name] = self.counts.get(cls_name, 0) + 1

        return dict(self.counts)

    def get_active_counts(self, tracked_objects: List[Dict[str, Any]]) -> Dict[str, int]:
        """Đếm số lượng xe từng loại đang hiện diện trực tiếp trong frame hiện tại."""
        active = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "bicycle": 0}
        for obj in tracked_objects:
            cls_name = obj.get("class", "car")
            active[cls_name] = active.get(cls_name, 0) + 1
        return active

    def get_total_count(self) -> int:
        """Tổng số lượng xe đã qua vạch."""
        return sum(self.counts.values())

    def get_total_unique_seen(self) -> int:
        """Tổng số ID duy nhất từng xuất hiện."""
        return len(self.all_seen_ids)

    def reset(self):
        """Reset bộ đếm."""
        self.counted_ids.clear()
        self.prev_positions.clear()
        self.all_seen_ids.clear()
        for k in self.counts:
            self.counts[k] = 0
