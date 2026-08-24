import unittest
import sys
import os
from pathlib import Path

# Đảm bảo root workspace có trong sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np


class TestSystemIntegrity(unittest.TestCase):
    def test_imports(self):
        """Kiểm tra import các module chính của hệ thống FastAPI & Pipeline."""
        import utils.config as config
        import utils.drawing as drawing
        import llm.traffic_profile as tp
        import pipeline as pl
        from server import app

        self.assertTrue(hasattr(drawing, "draw_vehicle_boxes"))
        self.assertTrue(hasattr(drawing, "draw_hud"))
        self.assertTrue(hasattr(tp, "generate_traffic_profile"))
        self.assertTrue(hasattr(pl, "UAVTrafficPipeline"))
        self.assertIsNotNone(app)

    def test_drawing_functions(self):
        """Kiểm tra logic hàm drawing trên mảng numpy giả lập."""
        from utils.drawing import draw_hud

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated = draw_hud(dummy_frame, "TEST HUD")
        self.assertEqual(annotated.shape, (480, 640, 3))

    def test_llm_profile_generation(self):
        """Kiểm tra hàm sinh khuyến nghị tín hiệu đèn từ metrics."""
        from llm.traffic_profile import generate_traffic_profile

        mock_metrics = {
            "vehicles_roi": 25,
            "counts": {"car": 12, "motorcycle": 10, "bus": 2, "truck": 1},
            "occupancy_rate": 45.0,
            "avg_speed": 18.5,
            "stopped_ratio": 20.0,
            "congestion_index": 35.0
        }
        profile_text = generate_traffic_profile(mock_metrics)
        self.assertIn("TRAFFIC CONTROL PROFILE", profile_text.upper())
        self.assertTrue("BẮC" in profile_text.upper() or "ĐÔNG" in profile_text.upper() or "CHU KỲ" in profile_text.upper())


if __name__ == "__main__":
    unittest.main()
