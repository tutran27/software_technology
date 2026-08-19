"""
Script kiểm tra toàn diện mô hình YOLO11 (Detection & Performance Benchmark)
Hỗ trợ:
- Kiểm tra thông tin mô hình (Classes, Backbone, Parameters, Device)
- Test Detect trên ảnh đơn lẻ (Single Image / Frame)
- Test Detect trên Video với đo đạc FPS, độ trễ suy luận (Preprocess, Inference, Postprocess)
- Tùy chỉnh tham số dòng lệnh (CLI arguments): conf, iou, source, model, save_dir
"""

import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Kiểm tra mô hình YOLO11 Vehicle Detection")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Đường dẫn tới file trọng số model (.pt)")
    parser.add_argument("--source", type=str, default="videos/DJI_20250516071323_0341_D.MP4", help="Đường dẫn video hoặc ảnh đầu vào")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng tin cậy Confidence threshold (0.0 - 1.0)")
    parser.add_argument("--iou", type=float, default=0.45, help="Ngưỡng NMS IoU threshold (0.0 - 1.0)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đưa vào YOLO model (mặc định: 640)")
    parser.add_argument("--classes", nargs="+", default=None, help="Danh sách class cần detect (vd: 1 2 3 4 hoặc car motor bus truck)")
    parser.add_argument("--exclude", nargs="+", default=["bicycle"], help="Danh sách class loại bỏ không detect (mặc định: bicycle)")
    parser.add_argument("--frames", type=int, default=100, help="Số frame cần test nếu đầu vào là video (0 để chạy hết)")
    parser.add_argument("--save-img", type=str, default="output_test_frame.jpg", help="Tên file ảnh xuất kết quả frame đầu")
    parser.add_argument("--save-vid", type=str, default="output_test_video.mp4", help="Tên file video xuất kết quả")
    parser.add_argument("--device", type=str, default="", help="Device chạy: '0', 'cpu', hoặc để trống tự động nhận diện")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Thuật toán Tracking: 'bytetrack.yaml' hoặc 'botsort.yaml'")
    parser.add_argument("--show", action="store_true", default=True, help="Hiển thị cửa sổ xem realtime (Nhấn 'q' để dừng)")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Tắt hiển thị cửa sổ realtime")
    return parser.parse_args()


def resolve_target_classes(model_names, selected_classes=None, excluded_classes=None):
    """Xác định danh sách class IDs cần detect."""
    name_to_id = {name.lower(): cls_id for cls_id, name in model_names.items()}
    all_ids = list(model_names.keys())

    if selected_classes:
        target_ids = []
        for item in selected_classes:
            item_str = str(item).lower()
            if item_str.isdigit() and int(item_str) in model_names:
                target_ids.append(int(item_str))
            elif item_str in name_to_id:
                target_ids.append(name_to_id[item_str])
        return sorted(list(set(target_ids)))

    if excluded_classes:
        exclude_ids = set()
        for item in excluded_classes:
            item_str = str(item).lower()
            if item_str.isdigit() and int(item_str) in model_names:
                exclude_ids.add(int(item_str))
            elif item_str in name_to_id:
                exclude_ids.add(name_to_id[item_str])
        return [cls_id for cls_id in all_ids if cls_id not in exclude_ids]

    return None


def check_hardware():
    print("\n" + "=" * 60)
    print("🖥️  1. KIỂM TRA MÔI TRƯỜNG PHẦN CỨNG & THƯ VIỆN")
    print("=" * 60)
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    print(f"🔹 PyTorch Version: {torch.__version__}")
    print(f"🔹 OpenCV Version : {cv2.__version__}")
    print(f"🔹 CUDA Available : {'✅ Có' if cuda_available else '❌ Không (Dùng CPU)'}")
    if cuda_available:
        print(f"🔹 GPU Device Name: {device_name}")
        print(f"🔹 VRAM Khả dụng  : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")


def load_and_inspect_model(model_path):
    print("\n" + "=" * 60)
    print("📦 2. KIỂM TRA FILE MODEL TRỌNG SỐ")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"❌ LỖI: Không tìm thấy file trọng số tại '{model_path}'")
        sys.exit(1)

    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"📁 Đường dẫn: {model_path}")
    print(f"📊 Dung lượng: {file_size_mb:.2f} MB")

    try:
        model = YOLO(model_path)
        print(f"✅ Tải mô hình YOLO thành công!")
        print(f"🏷️  Số lượng classes: {len(model.names)}")
        print("📋 Danh sách các đối tượng nhận diện:")
        for cls_id, cls_name in model.names.items():
            print(f"   [{cls_id}] {cls_name}")
        return model
    except Exception as e:
        print(f"❌ Không thể khởi tạo YOLO model: {e}")
        sys.exit(1)


def test_single_frame(model, source_path, conf=0.25, iou=0.45, imgsz=640, classes=None, output_img="output_test_frame.jpg", device=""):
    print("\n" + "=" * 60)
    print("🖼️  3. TEST DETECTION TRÊN 1 FRAME ĐẦU TIÊN")
    print("=" * 60)

    if not os.path.exists(source_path):
        print(f"❌ Không tìm thấy file dữ liệu: {source_path}")
        return

    # Đọc ảnh hoặc frame đầu từ video
    if source_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        frame = cv2.imread(source_path)
    else:
        cap = cv2.VideoCapture(source_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("❌ Không thể đọc frame từ nguồn video!")
            return

    h, w = frame.shape[:2]
    print(f"📐 Độ phân giải gốc: {w} x {h} | imgsz detect: {imgsz}")

    # Chạy inference
    start_t = time.perf_counter()
    results = model.predict(
        source=frame,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        classes=classes,
        device=device if device else None,
        verbose=False
    )[0]
    total_time = (time.perf_counter() - start_t) * 1000

    boxes = results.boxes
    num_detected = len(boxes)

    # Thống kê chi tiết từng class
    class_counts = {}
    confidences = []
    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf_val = float(box.conf[0].item())
        cls_name = model.names.get(cls_id, f"class_{cls_id}")
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        confidences.append(conf_val)

    avg_conf = np.mean(confidences) if confidences else 0.0

    print(f"\n⏱️  Thời gian suy luận: {total_time:.2f} ms (~{1000/total_time:.1f} FPS)")
    if hasattr(results, 'speed'):
        speed = results.speed
        print(f"   - Pre-process : {speed.get('preprocess', 0):.2f} ms")
        print(f"   - Inference   : {speed.get('inference', 0):.2f} ms")
        print(f"   - Post-process: {speed.get('postprocess', 0):.2f} ms")

    print(f"\n🎯 Tổng số phương tiện phát hiện: {num_detected} (Conf TB: {avg_conf:.2%})")
    for name, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"   • {name:12s}: {cnt:3d} xe")

    # Vẽ và lưu kết quả
    annotated = results.plot(font_size=1, line_width=2)
    cv2.imwrite(output_img, annotated)
    print(f"\n💾 Đã lưu ảnh kết quả trực quan hóa tại: {output_img}")


def test_video_stream(model, video_path, num_frames=100, conf=0.25, iou=0.45, imgsz=640, classes=None, tracker="bytetrack.yaml", output_vid="output_test_video.mp4", device="", show=True):
    print("\n" + "=" * 60)
    print(f"🎥 4. TEST MULTI-OBJECT TRACKING TRÊN VIDEO (YOLO TRACK)")
    print("=" * 60)

    if not os.path.exists(video_path):
        print(f"❌ Không tìm thấy video: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Không thể mở video {video_path}")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    max_frames = min(num_frames, total_vid_frames) if num_frames > 0 else total_vid_frames
    print(f"📹 Video: {video_path}")
    print(f"📐 Kích thước: {orig_w}x{orig_h} | FPS gốc: {fps_in:.1f} | Tổng số frames: {total_vid_frames}")
    print(f"🎯 Tracker: {tracker} | imgsz: {imgsz}")
    print(f"🚀 Sẽ xử lý: {'Toàn bộ video (' + str(max_frames) + ' frames)' if num_frames == 0 else str(max_frames) + ' frames'}")
    if show:
        print("🖥️  Chế độ xem Realtime: Đang bật (Bấm phím 'q' hoặc 'ESC' trên cửa sổ video để dừng sớm)")

    # Resize để xuất file mượt mà nếu video gốc là 4K
    out_w, out_h = (960, 540) if orig_w > 960 else (orig_w, orig_h)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_vid, fourcc, fps_in, (out_w, out_h))

    processed_count = 0
    inference_times = []
    all_detections_count = []
    unique_track_ids = set()

    start_bench_time = time.time()

    while cap.isOpened() and processed_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize trước khi vẽ/lưu nếu cần
        if (orig_w, orig_h) != (out_w, out_h):
            frame_resized = cv2.resize(frame, (out_w, out_h))
        else:
            frame_resized = frame

        # Tracking Inference
        t0 = time.perf_counter()
        results = model.track(
            source=frame_resized,
            persist=True,
            tracker=tracker,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            classes=classes,
            device=device if device else None,
            verbose=False
        )[0]
        t1 = time.perf_counter()

        inference_times.append((t1 - t0) * 1000)
        num_boxes = len(results.boxes)
        all_detections_count.append(num_boxes)

        # Cập nhật danh sách ID duy nhất
        if results.boxes.id is not None:
            current_ids = results.boxes.id.int().cpu().tolist()
            unique_track_ids.update(current_ids)

        # Vẽ kết quả (tự động vẽ BBox + Track ID + Class + Conf)
        annotated_frame = results.plot(font_size=1, line_width=2)

        # Thêm thông tin Tracking & FPS trực tiếp lên góc video
        current_fps = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0
        cv2.putText(
            annotated_frame,
            f"YOLO Track ({tracker.split('.')[0]}) | Active: {num_boxes} | Total Tracked: {len(unique_track_ids)} | FPS: {current_fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        writer.write(annotated_frame)
        processed_count += 1

        # Hiển thị cửa sổ Realtime nếu bật show
        if show:
            cv2.imshow("YOLO11 Vehicle Tracking Realtime", annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' hoặc ESC
                print(f"\n⏹️ Đã dừng sớm theo yêu cầu người dùng tại frame {processed_count}/{max_frames}!")
                break

        if processed_count % 25 == 0 or processed_count == max_frames:
            avg_inf = np.mean(inference_times[-25:])
            print(f"   ▶ Đã xử lý {processed_count:3d}/{max_frames} frames | Latency gần nhất: {avg_inf:.2f} ms ({1000/avg_inf:.1f} FPS) | Total IDs: {len(unique_track_ids)}")

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    total_elapsed = time.time() - start_bench_time

    # Báo cáo tổng kết
    print("\n" + "-" * 60)
    print("📈 TỔNG KẾT HIỆU NĂNG TRACKING (TRACKING BENCHMARK):")
    print(f"   • Tổng số frames đã xử lý   : {processed_count}")
    print(f"   • Tổng thời gian thực thi   : {total_elapsed:.2f} s")
    print(f"   • Tốc độ xử lý trung bình   : {processed_count / total_elapsed if total_elapsed > 0 else 0:.1f} FPS")
    print(f"   • Độ trễ trung bình/frame   : {np.mean(inference_times) if inference_times else 0:.2f} ms")
    print(f"   • Độ trễ min / max          : {np.min(inference_times) if inference_times else 0:.2f} ms / {np.max(inference_times) if inference_times else 0:.2f} ms")
    print(f"   • Số xe phát hiện TB/frame  : {np.mean(all_detections_count) if all_detections_count else 0:.1f} xe")
    print(f"   • Tổng số xe duy nhất (IDs) : {len(unique_track_ids)} xe")
    print(f"💾 Video kết quả đã được lưu tại: {output_vid}")
    print("=" * 60 + "\n")


def main():
    args = parse_args()
    check_hardware()
    model = load_and_inspect_model(args.model)

    target_classes = resolve_target_classes(
        model.names,
        selected_classes=args.classes,
        excluded_classes=args.exclude
    )
    if target_classes is not None:
        target_names = [f"[{cid}] {model.names[cid]}" for cid in target_classes]
        print(f"🎯 Đang lọc detect {len(target_classes)}/{len(model.names)} classes: {', '.join(target_names)}")
    else:
        print("🎯 Đang detect toàn bộ các classes của model.")

    test_single_frame(
        model=model,
        source_path=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        classes=target_classes,
        output_img=args.save_img,
        device=args.device
    )
    if not args.source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        test_video_stream(
            model=model,
            video_path=args.source,
            num_frames=args.frames,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            classes=target_classes,
            tracker=args.tracker,
            output_vid=args.save_vid,
            device=args.device,
            show=args.show
        )


if __name__ == "__main__":
    main()
