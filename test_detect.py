"""
Script kiểm tra nhận diện và theo dõi phương tiện bằng YOLO11 trên ảnh/video.
Hỗ trợ:
- Tự động hiển thị cửa sổ xem realtime video/ảnh trong quá trình xử lý (nhấn 'q'/ESC để dừng).
- Chạy trên 1 ảnh, 1 video hoặc toàn bộ thư mục video.
- Bám vết phương tiện (ByteTrack / BoT-SORT).
- Xuất video/ảnh kết quả có bounding box, class, ID và HUD thống kê.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional
import cv2
from ultralytics import YOLO

from utils.drawing import draw_hud, draw_vehicle_boxes


def process_image(
    model: YOLO,
    image_path: str,
    output_path: str,
    conf: float = 0.25,
    imgsz: int = 640,
    device: str = "",
    show: bool = True
):
    """Nhận diện phương tiện trên 1 ảnh và hiển thị + lưu kết quả."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Không thể đọc ảnh: {image_path}")
        return

    results = model.predict(
        source=frame,
        imgsz=imgsz,
        conf=conf,
        device=device if device else None,
        verbose=False
    )[0]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    num_boxes = len(results.boxes) if results.boxes is not None else 0
    annotated = draw_vehicle_boxes(frame=frame, boxes=results.boxes, model_names=model.names)
    annotated = draw_hud(annotated, f"YOLO Detection | Vehicles: {num_boxes}")

    cv2.imwrite(output_path, annotated)
    print(f"💾 Đã lưu ảnh kết quả: {output_path}")

    if show:
        win_name = "YOLO Detection - Image"
        h, w = annotated.shape[:2]
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, min(1280, w), min(720, h))
        cv2.imshow(win_name, annotated)
        print("🖥️ Đang hiển thị ảnh (nhấn phím bất kỳ trên cửa sổ để tiếp tục)...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(
    model: YOLO,
    video_path: str,
    output_path: str,
    conf: float = 0.1,
    iou: float = 0.45,
    imgsz: int = 1280,
    tracker: str = "botsort.yaml",
    device: str = "",
    num_frames: int = 0,
    show: bool = True
):
    """Theo dõi phương tiện trên video, hiển thị realtime và lưu video kết quả."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Không thể mở video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(num_frames, total_frames) if num_frames > 0 else total_frames

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    win_name = "YOLO Tracking - Realtime Video"
    if show:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, min(1280, width), min(720, height))
        print("🖥️ Đang hiển thị video realtime (Nhấn 'q' hoặc ESC trên cửa sổ video để dừng)...")

    print(f"\n🚀 Đang xử lý video: {video_path} ({max_frames} frames)...")
    processed = 0
    t_start = time.time()

    try:
        while cap.isOpened() and processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            results = model.track(
                source=frame,
                persist=True,
                tracker=tracker,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device if device else None,
                verbose=False
            )[0]
            frame_fps = 1.0 / max(1e-5, time.perf_counter() - t0)

            num_boxes = len(results.boxes) if results.boxes is not None else 0
            annotated = draw_vehicle_boxes(frame=frame, boxes=results.boxes, model_names=model.names)
            annotated = draw_hud(annotated, f"Frame: {processed+1}/{max_frames}", fps=frame_fps, vehicle_count=num_boxes)

            writer.write(annotated)
            processed += 1

            if show:
                cv2.imshow(win_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    print("\n⏹️ Người dùng đã dừng bằng phím 'q'/ESC.")
                    break

            if processed % 30 == 0 or processed == max_frames:
                print(f"   ▶ Đã xử lý {processed}/{max_frames} frames ({processed/max_frames*100:.1f}%)")
    finally:
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    avg_fps = processed / elapsed if elapsed > 0 else 0
    print(f"✅ Xong {processed} frames trong {elapsed:.2f}s (~{avg_fps:.1f} FPS) -> Lưu: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Script kiểm tra nhận diện YOLO11 trên ảnh/video")
    parser.add_argument("--source", "-s", type=str, default="videos/output_videos/M0702.mp4", help="Đường dẫn file ảnh/video hoặc thư mục")
    parser.add_argument("--model", "-m", type=str, default="models/best.pt", help="Đường dẫn trọng số YOLO (.pt)")
    parser.add_argument("--output-dir", "-o", type=str, default="outputs", help="Thư mục lưu kết quả")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng confidence")
    parser.add_argument("--iou", type=float, default=0.45, help="Ngưỡng IoU")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh inference")
    parser.add_argument("--device", type=str, default="", help="Thiết bị: '0', 'cpu', ''")
    parser.add_argument("--num-frames", "-n", type=int, default=0, help="Số frame tối đa cần xử lý (0 = tất cả)")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracker config (bytetrack.yaml / botsort.yaml)")
    parser.add_argument("--no-show", action="store_true", help="Tắt cửa sổ hiển thị realtime (chỉ lưu file)")
    args = parser.parse_args()

    show = not args.no_show
    source_p = Path(args.source)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not source_p.exists():
        print(f"❌ Không tìm thấy nguồn dữ liệu: {source_p}")
        return

    print(f"📦 Đang tải mô hình YOLO: {args.model}...")
    model = YOLO(args.model)

    valid_img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    valid_vid_exts = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm"}

    if source_p.is_dir():
        files = sorted([f for f in source_p.iterdir() if f.is_file() and f.suffix.lower() in (valid_img_exts | valid_vid_exts)])
        print(f"📁 Tìm thấy {len(files)} file cần xử lý trong thư mục: {source_p}")
        for idx, f in enumerate(files, 1):
            print(f"\n[{idx}/{len(files)}] Xử lý: {f.name}")
            out_file = str(out_dir / f.name)
            if f.suffix.lower() in valid_img_exts:
                process_image(model, str(f), out_file, args.conf, args.imgsz, args.device, show)
            else:
                process_video(model, str(f), out_file, args.conf, args.iou, args.imgsz, args.tracker, args.device, args.num_frames, show)
    else:
        out_file = str(out_dir / source_p.name)
        if source_p.suffix.lower() in valid_img_exts:
            process_image(model, str(source_p), out_file, args.conf, args.imgsz, args.device, show)
        else:
            process_video(model, str(source_p), out_file, args.conf, args.iou, args.imgsz, args.tracker, args.device, args.num_frames, show)


if __name__ == "__main__":
    main()
