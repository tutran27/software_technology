"""
Modal Cloud Runner for UAV Vehicle Detection & Tracking
Chạy toàn bộ pipeline trên Cloud GPU + RAM lớn của Modal và tải kết quả về máy.

Cách chạy:
    1. Cài modal: python3 -m pip install --user modal --break-system-packages
    2. Đăng nhập: modal setup
    3. Chạy test trên Cloud:
       modal run modal_runner.py --frames 100
"""

import os
from pathlib import Path
import modal

# 1. Định nghĩa môi trường container trên Cloud Modal
app = modal.App("uav-traffic-detection")

# Cài đặt môi trường Python + CUDA + OpenCV + Ultralytics trên Cloud
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "ultralytics>=8.2.0",
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torchvision"
    )
)


# 2. Hàm xử lý chạy trên GPU Cloud
@app.function(
    image=image,
    gpu="T4",               # GPU T4 (rất rẻ & đủ mạnh) hoặc "L4" / "A10G"
    timeout=600,            # Timeout tối đa 10 phút
    memory=8192,            # 8GB RAM trên Cloud (có thể tăng lên 16384 nếu cần)
)
def process_video_cloud(video_bytes: bytes, model_bytes: bytes, conf: float = 0.25, max_frames: int = 100):
    import cv2
    import numpy as np
    import time
    from ultralytics import YOLO

    print("🚀 [Cloud] Đang nạp model và video vào bộ nhớ Cloud...")

    # Ghi bytes tạm ra disk trên container
    with open("/tmp/model.pt", "wb") as f:
        f.write(model_bytes)
    with open("/tmp/input_video.mp4", "wb") as f:
        f.write(video_bytes)

    # Load YOLO
    model = YOLO("/tmp/model.pt")
    print(f"✅ [Cloud] Load model thành công! Classes: {model.names}")

    cap = cv2.VideoCapture("/tmp/input_video.mp4")
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Resize để xuất video mượt mà (1080p)
    out_w, out_h = (1920, 1080) if orig_w > 1920 else (orig_w, orig_h)
    out_path = "/tmp/output_detected.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    frames_to_run = min(max_frames, total_frames) if max_frames > 0 else total_frames
    print(f"🎬 [Cloud] Bắt đầu xử lý {frames_to_run}/{total_frames} frames ({orig_w}x{orig_h} -> {out_w}x{out_h})...")

    processed = 0
    start_time = time.time()

    while cap.isOpened() and processed < frames_to_run:
        ret, frame = cap.read()
        if not ret:
            break

        if (orig_w, orig_h) != (out_w, out_h):
            frame = cv2.resize(frame, (out_w, out_h))

        # Suy luận YOLO
        t0 = time.perf_counter()
        results = model.predict(source=frame, conf=conf, verbose=False)[0]
        t1 = time.perf_counter()

        annotated = results.plot(font_size=1, line_width=2)
        fps_curr = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0

        cv2.putText(
            annotated,
            f"MODAL CLOUD GPU | FPS: {fps_curr:.1f} | Vehicles: {len(results.boxes)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        writer.write(annotated)
        processed += 1

        if processed % 30 == 0 or processed == frames_to_run:
            print(f"   [Cloud] Đã xử lý {processed}/{frames_to_run} frames...")

    cap.release()
    writer.release()
    elapsed = time.time() - start_time
    print(f"✨ [Cloud] Hoàn thành {processed} frames trong {elapsed:.2f}s ({processed/elapsed:.1f} FPS)!")

    # Đọc file video kết quả dạng bytes trả về cho máy local
    with open(out_path, "rb") as f:
        output_bytes = f.read()

    return output_bytes, processed, elapsed


# 3. Entrypoint gọi từ máy local
@app.local_entrypoint()
def main(
    model_path: str = "models/best.pt",
    video_path: str = "videos/DJI_20250516071323_0341_D.MP4",
    output_path: str = "output_modal_video.mp4",
    conf: float = 0.25,
    frames: int = 100
):
    print("=" * 60)
    print("☁️  MODAL CLOUD RUNNER - UAV TRAFFIC SYSTEM")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy model tại {model_path}")
        return
    if not os.path.exists(video_path):
        print(f"❌ Không tìm thấy video tại {video_path}")
        return

    print(f"📦 Đang đọc model local: {model_path} ({os.path.getsize(model_path)/(1024*1024):.2f} MB)")
    with open(model_path, "rb") as f:
        model_bytes = f.read()

    print(f"📹 Đang đọc video local: {video_path} ({os.path.getsize(video_path)/(1024*1024):.2f} MB)")
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    print("\n🚀 Đang gửi dữ liệu lên Cloud Modal để xử lý...")
    output_bytes, processed, elapsed = process_video_cloud.remote(
        video_bytes=video_bytes,
        model_bytes=model_bytes,
        conf=conf,
        max_frames=frames
    )

    print(f"\n💾 Đang tải và lưu video kết quả về máy local: {output_path}")
    with open(output_path, "wb") as f:
        f.write(output_bytes)

    print(f"🎉 THÀNH CÔNG! Đã lưu video tại '{output_path}' ({len(output_bytes)/(1024*1024):.2f} MB)")
    print(f"📊 Thống kê: {processed} frames trong {elapsed:.2f} giây")
    print("=" * 60)
