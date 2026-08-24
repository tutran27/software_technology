#!/usr/bin/env python3
"""
Script ghép các frame ảnh trong từng thư mục con của UAV-benchmark-M thành video.
Hỗ trợ:
- Tùy chỉnh FPS (--fps)
- Tùy chỉnh thư mục đầu vào & đầu ra (--input-dir, --output-dir)
- Ghép 1 thư mục cụ thể hoặc toàn bộ tất cả các thư mục (--folder)
- Xử lý đa luồng / đa tiến trình tăng tốc (--workers)
- Sắp xếp thứ tự tự nhiên (natural sort) cho các frame ảnh
"""

import argparse
import glob
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import cv2


def natural_sort_key(s: str):
    """Hàm tách số trong chuỗi để sắp xếp tự nhiên (1, 2, 10 thay vì 1, 10, 2)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def get_image_files(folder_path: Path) -> List[Path]:
    """Tìm và sắp xếp tất cả các file ảnh trong folder."""
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    image_files = [
        p for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in valid_extensions
    ]
    image_files.sort(key=lambda x: natural_sort_key(x.name))
    return image_files


def convert_folder_to_video(
    folder_path: Path,
    output_path: Path,
    fps: float = 30.0,
    codec: str = "mp4v",
    overwrite: bool = False,
) -> Tuple[bool, str]:
    """Ghép tất cả frame trong 1 thư mục thành 1 file video.

    Args:
        folder_path: Đường dẫn thư mục chứa frames
        output_path: Đường dẫn file video đầu ra
        fps: Số khung hình / giây
        codec: FourCC codec (mặc định 'mp4v')
        overwrite: Ghi đè nếu file đã tồn tại

    Returns:
        (thành_công: bool, thông_điệp: str)
    """
    folder_name = folder_path.name

    if output_path.exists() and not overwrite:
        return True, f"[{folder_name}] Bỏ qua vì video đã tồn tại: {output_path.name}"

    image_files = get_image_files(folder_path)
    if not image_files:
        return False, f"[{folder_name}] Không tìm thấy file ảnh hợp lệ nào."

    # Đọc frame đầu tiên để lấy kích thước (width, height)
    first_frame = cv2.imread(str(image_files[0]))
    if first_frame is None:
        return False, f"[{folder_name}] Không thể đọc frame đầu tiên: {image_files[0].name}"

    height, width, _ = first_frame.shape
    frame_size = (width, height)

    # Đảm bảo thư mục đầu ra tồn tại
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

    if not writer.isOpened():
        return False, f"[{folder_name}] Lỗi: Không thể khởi tạo VideoWriter với codec '{codec}'."

    start_time = time.time()
    writer.write(first_frame)

    # Ghi các frame tiếp theo
    for img_path in image_files[1:]:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [Cảnh báo] Không đọc được frame: {img_path.name}, bỏ qua.")
            continue
        # Resize nếu có frame nào lệch kích thước
        if (frame.shape[1], frame.shape[0]) != frame_size:
            frame = cv2.resize(frame, frame_size)
        writer.write(frame)

    writer.release()
    elapsed = time.time() - start_time
    total_frames = len(image_files)
    duration_sec = total_frames / fps

    return (
        True,
        f"[{folder_name}] Hoàn thành: {total_frames} frames -> {output_path.name} "
        f"({width}x{height} @ {fps:.1f} FPS, thời lượng: {duration_sec:.2f}s, xử lý: {elapsed:.2f}s)"
    )


def resolve_paths(input_dir_arg: Optional[str], output_dir_arg: Optional[str]) -> Tuple[Path, Path]:
    """Tự động xác định đường dẫn thư mục input/output dù chạy từ project root hay từ thư mục videos."""
    script_dir = Path(__file__).resolve().parent

    # Xác định input_dir
    if input_dir_arg:
        input_dir = Path(input_dir_arg).resolve()
    else:
        # Kiểm tra các vị trí mặc định phổ biến
        candidates = [
            script_dir / "UAV-benchmark-M",
            script_dir.parent / "videos" / "UAV-benchmark-M",
            Path.cwd() / "videos" / "UAV-benchmark-M",
            Path.cwd() / "UAV-benchmark-M",
        ]
        input_dir = None
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                input_dir = candidate.resolve()
                break
        if input_dir is None:
            input_dir = (script_dir / "UAV-benchmark-M").resolve()

    # Xác định output_dir
    if output_dir_arg:
        output_dir = Path(output_dir_arg).resolve()
    else:
        output_dir = input_dir.parent / "output_videos"

    return input_dir, output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Ghép các frame trong từng folder của UAV-benchmark-M thành video với FPS tùy chỉnh."
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        default=None,
        help="Đường dẫn đến thư mục chứa các folder sequence (mặc định: UAV-benchmark-M trong videos/)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Đường dẫn thư mục lưu các video đầu ra (mặc định: videos/output_videos)"
    )
    parser.add_argument(
        "--fps", "-f",
        type=float,
        default=30.0,
        help="Tốc độ khung hình (Frames Per Second) của video đầu ra (mặc định: 30.0)"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Chỉ xử lý 1 hoặc danh sách folder cụ thể (phân cách bằng dấu phẩy, ví dụ: M0101,M0201)"
    )
    parser.add_argument(
        "--codec", "-c",
        type=str,
        default="mp4v",
        help="FourCC codec video (mặc định: mp4v)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=os.cpu_count() or 4,
        help="Số lượng tiến trình xử lý song song các folder (mặc định: số CPU core)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ghi đè video nếu đã tồn tại"
    )

    args = parser.parse_args()

    input_dir, output_dir = resolve_paths(args.input_dir, args.output_dir)

    print("=" * 70)
    print("🎬 CHƯƠNG TRÌNH GHÉP FRAME THÀNH VIDEO")
    print("=" * 70)
    print(f"📁 Thư mục nguồn (Input) : {input_dir}")
    print(f"📁 Thư mục xuất (Output) : {output_dir}")
    print(f"🎞️  FPS cài đặt           : {args.fps}")
    print(f"⚙️  Codec                : {args.codec}")
    print(f"⚡ Số luồng xử lý        : {args.workers}")
    print(f"🔄 Ghi đè file cũ       : {'Bật' if args.overwrite else 'Tắt'}")
    print("=" * 70)

    if not input_dir.exists():
        print(f"❌ Lỗi: Thư mục nguồn không tồn tại: {input_dir}")
        sys.exit(1)

    # Tìm các subfolder
    if args.folder:
        selected_names = [name.strip() for name in args.folder.split(",")]
        subfolders = [input_dir / name for name in selected_names if (input_dir / name).is_dir()]
        if not subfolders:
            print(f"❌ Không tìm thấy thư mục nào phù hợp trong: {selected_names}")
            sys.exit(1)
    else:
        subfolders = [p for p in input_dir.iterdir() if p.is_dir()]
        subfolders.sort(key=lambda x: natural_sort_key(x.name))

    if not subfolders:
        print(f"❌ Không tìm thấy thư mục con nào trong: {input_dir}")
        sys.exit(1)

    print(f"🔍 Tìm thấy {len(subfolders)} thư mục cần xử lý.\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for folder in subfolders:
        video_filename = f"{folder.name}.mp4"
        out_video_path = output_dir / video_filename
        tasks.append((folder, out_video_path, args.fps, args.codec, args.overwrite))

    total_tasks = len(tasks)
    success_count = 0
    fail_count = 0
    total_start = time.time()

    # Chạy đơn luồng nếu workers <= 1 hoặc chỉ có 1 task
    if args.workers <= 1 or total_tasks == 1:
        for i, task in enumerate(tasks, 1):
            folder, out_path, fps, codec, overwrite = task
            print(f"[{i}/{total_tasks}] Đang xử lý: {folder.name}...")
            ok, msg = convert_folder_to_video(folder, out_path, fps, codec, overwrite)
            print(f"  └─> {msg}")
            if ok:
                success_count += 1
            else:
                fail_count += 1
    else:
        # Chạy đa tiến trình
        print(f"🚀 Bắt đầu xử lý song song với {min(args.workers, total_tasks)} tiến trình...")
        with ProcessPoolExecutor(max_workers=min(args.workers, total_tasks)) as executor:
            future_to_folder = {
                executor.submit(convert_folder_to_video, folder, out_path, fps, codec, overwrite): folder.name
                for folder, out_path, fps, codec, overwrite in tasks
            }

            completed = 0
            for future in as_completed(future_to_folder):
                completed += 1
                folder_name = future_to_folder[future]
                try:
                    ok, msg = future.result()
                    if ok:
                        success_count += 1
                        print(f"[{completed}/{total_tasks}] ✅ {msg}")
                    else:
                        fail_count += 1
                        print(f"[{completed}/{total_tasks}] ❌ {msg}")
                except Exception as exc:
                    fail_count += 1
                    print(f"[{completed}/{total_tasks}] ❌ [{folder_name}] Lỗi ngoại lệ: {exc}")

    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print("✨ TỔNG KẾT QUÁ TRÌNH GHÉP VIDEO")
    print("=" * 70)
    print(f"✅ Thành công : {success_count}/{total_tasks} videos")
    if fail_count > 0:
        print(f"❌ Thất bại   : {fail_count}/{total_tasks} videos")
    print(f"⏱️  Tổng thời gian : {total_time:.2f} giây")
    print(f"📁 Video lưu tại   : {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
