"""
FastAPI Server cho Hệ thống Giám sát & Điều khiển Giao thông UAV.

Chức năng chính:
1. Web Dashboard & MJPEG Video Streaming thời gian thực (~30 FPS).
2. REST API quản lý luồng video, cập nhật tham số AI & lấy số liệu giám sát.
3. Tích hợp LLM tư vấn chu kỳ đèn tín hiệu giao thông thích ứng.
4. Xuất báo cáo dữ liệu lịch sử dạng file CSV.
"""

import io
import os
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline import UAVTrafficPipeline
from llm.traffic_profile import TrafficAdvisor
from utils.config import DEFAULT_MODEL_PATH, DEFAULT_SAMPLE_VIDEO, SAMPLE_VIDEOS

# -----------------------------------------------------------------------------
# 1. Cấu hình Đường dẫn & Thư mục Web
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
UPLOAD_DIR = BASE_DIR / "videos" / "temp_uploads"

# -----------------------------------------------------------------------------
# 2. Khởi tạo Trạng thái Engine (TrafficEngineState)
# -----------------------------------------------------------------------------
class TrafficEngineState:
    """Quản lý toàn bộ trạng thái hệ thống: Video Capture, Pipeline AI và Số liệu."""

    def __init__(self):
        self.lock = threading.RLock()
        
        # Trạng thái phát video
        self.is_running = False
        self.is_paused = False
        self.loop_video = True
        
        # Cấu hình AI & Video
        self.video_path = DEFAULT_SAMPLE_VIDEO
        self.tracker_name = "ByteTrack"
        self.conf = 0.25
        self.iou = 0.45
        self.line_y = 550
        self.pixels_per_meter = 10.0
        self.imgsz = 640
        self.target_w = 1280
        
        # Core Components
        self.pipeline: Optional[UAVTrafficPipeline] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_frame_bytes: Optional[bytes] = None

        # Dữ liệu thống kê thời gian thực
        self.latest_stats = self._default_stats()
        self.latest_counts = self._default_counts()
        self.latest_line_counts = self._default_counts()
        self.latest_alerts: List[Dict[str, Any]] = []
        self.history_records: List[Dict[str, Any]] = []
        self.video_meta: Dict[str, Any] = {}

        # Khởi tạo pipeline & preview
        self.init_pipeline()
        self.update_video_metadata()
        self.generate_preview_frame()

    @staticmethod
    def _default_stats() -> Dict[str, Any]:
        return {
            "vehicle_count": 0, "occupancy_rate": 0.0, "avg_speed": 0.0, "avg_speed_kmh": 0.0,
            "stopped_ratio": 0.0, "avg_dwell_time": 0.0, "congestion_index": 0.0,
            "congestion_level": "Thông thoáng", "total_counted": 0, "fps": 0.0
        }

    @staticmethod
    def _default_counts() -> Dict[str, int]:
        return {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "bicycle": 0}

    def init_pipeline(self):
        """Khởi tạo mô hình nhận diện & theo dõi phương tiện."""
        self.pipeline = UAVTrafficPipeline(
            model_path=DEFAULT_MODEL_PATH,
            tracker_name=self.tracker_name,
            conf=self.conf,
            iou=self.iou,
            line_y=self.line_y,
            pixels_per_meter=self.pixels_per_meter
        )

    def update_video_metadata(self):
        """Cập nhật thông tin độ phân giải và FPS của video nguồn."""
        if os.path.exists(self.video_path):
            cap_m = cv2.VideoCapture(self.video_path)
            if cap_m.isOpened():
                w = int(cap_m.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap_m.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap_m.get(cv2.CAP_PROP_FPS) or 30.0
                self.video_meta = {
                    "name": os.path.basename(self.video_path),
                    "resolution": f"{w}x{h}",
                    "fps": round(fps, 1)
                }
                cap_m.release()

    def prepare_frame(self, frame: np.ndarray):
        """Resize frame theo tỉ lệ chuẩn và quy đổi vị trí vạch đếm ảo."""
        fh, fw = frame.shape[:2]
        if self.target_w and fw != self.target_w:
            scale = float(self.target_w) / fw
            target_h = int(fh * scale)
            resized = cv2.resize(frame, (self.target_w, target_h), interpolation=cv2.INTER_AREA)
            scaled_line_y = int(self.line_y * scale)
        else:
            resized = frame
            scaled_line_y = self.line_y
        return resized, scaled_line_y

    def generate_preview_frame(self):
        """Tạo ảnh xem trước (preview) từ frame đầu tiên của video."""
        try:
            if not os.path.exists(self.video_path):
                return
            temp_cap = cv2.VideoCapture(self.video_path)
            if not temp_cap.isOpened():
                return
            ret, frame = temp_cap.read()
            temp_cap.release()

            if ret and frame is not None:
                frame_resized, scaled_line_y = self.prepare_frame(frame)
                if self.pipeline:
                    self.pipeline.line_y = scaled_line_y
                    annotated, payload = self.pipeline.process_frame(frame_resized, imgsz=self.imgsz)
                    with self.lock:
                        self.latest_stats = payload["density_stats"]
                        self.latest_counts = payload["counts"]
                        self.latest_line_counts = payload.get("line_counts", {})
                        self.latest_alerts = payload["alerts"]
                else:
                    annotated = frame_resized

                success, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if success:
                    self.current_frame_bytes = jpeg.tobytes()
        except Exception:
            pass

    def update_config(self, tracker=None, conf=None, iou=None, line_y=None, ppm=None, video_path=None):
        """Cập nhật các tham số AI & Nguồn video linh hoạt."""
        with self.lock:
            reinit = False
            if tracker and tracker != self.tracker_name:
                self.tracker_name = tracker
                reinit = True
            if conf is not None:
                self.conf = float(conf)
                reinit = True
            if iou is not None:
                self.iou = float(iou)
                reinit = True
            if line_y is not None:
                self.line_y = int(line_y)
                if self.pipeline:
                    self.pipeline.line_y = self.line_y
            if ppm is not None:
                self.pixels_per_meter = float(ppm)
                if self.pipeline:
                    self.pipeline.analyzer.pixels_per_meter = self.pixels_per_meter
            if video_path and video_path != self.video_path:
                self.video_path = video_path
                self.update_video_metadata()
                if self.cap:
                    self.cap.release()
                    self.cap = None
                self.generate_preview_frame()

            if reinit:
                self.init_pipeline()

    def reset_metrics(self):
        """Xóa sạch bộ đếm và lịch sử phân tích."""
        with self.lock:
            self.history_records.clear()
            if self.pipeline:
                self.pipeline.reset()
            self.latest_counts = self._default_counts()
            self.latest_line_counts = self._default_counts()
            self.latest_stats = self._default_stats()


# Instance duy nhất quản lý trạng thái
state = TrafficEngineState()


# -----------------------------------------------------------------------------
# 3. Worker Thread Xử Lý Video (Real-time Pipeline Loop)
# -----------------------------------------------------------------------------
def video_worker_loop():
    """Vòng lặp đọc video, xử lý AI và cập nhật luồng MJPEG chuẩn ~30 FPS."""
    # Frame chờ khi hệ thống ở trạng thái Standby
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(blank, "UAV TRAFFIC CONTROL - STANDBY", (380, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (56, 189, 248), 2, cv2.LINE_AA)
    _, blank_bytes = cv2.imencode('.jpg', blank)
    state.current_frame_bytes = blank_bytes.tobytes()

    frame_counter = 0
    TARGET_FRAME_TIME = 1.0 / 30.0  # ~33ms
    last_time = time.perf_counter()
    fps_smoothed = 30.0

    while True:
        try:
            if not state.is_running or state.is_paused:
                time.sleep(0.03)
                last_time = time.perf_counter()
                continue

            if state.cap is None or not state.cap.isOpened():
                state.cap = cv2.VideoCapture(state.video_path)

            ret, frame = state.cap.read()
            if not ret:
                if state.loop_video:
                    state.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    state.is_running = False
                    continue

            frame_counter += 1
            frame_resized, scaled_line_y = state.prepare_frame(frame)
            state.pipeline.line_y = scaled_line_y

            # Suy luận AI qua Pipeline
            annotated_frame, payload = state.pipeline.process_frame(frame_resized, imgsz=state.imgsz)

            # Kiểm soát tốc độ phát ~30 FPS
            now = time.perf_counter()
            elapsed = now - last_time
            if elapsed < TARGET_FRAME_TIME:
                time.sleep(TARGET_FRAME_TIME - elapsed)
                now = time.perf_counter()

            actual_dt = now - last_time
            last_time = now
            inst_fps = 1.0 / actual_dt if actual_dt > 0 else 30.0
            fps_smoothed = (fps_smoothed * 0.8) + (min(30.0, max(28.0, inst_fps)) * 0.2)
            payload["density_stats"]["fps"] = round(fps_smoothed, 1)

            # Cập nhật thông số vào state
            with state.lock:
                state.latest_stats = payload["density_stats"]
                state.latest_counts = payload["counts"]
                state.latest_line_counts = payload.get("line_counts", {})
                state.latest_alerts = payload["alerts"]

                # Lưu lịch sử mỗi 10 frames
                if frame_counter % 10 == 0:
                    state.history_records.append({
                        "time": time.strftime("%H:%M:%S"),
                        "congestion_index": payload["density_stats"]["congestion_index"],
                        "occupancy_rate": payload["density_stats"]["occupancy_rate"],
                        "avg_speed": payload["density_stats"]["avg_speed_kmh"],
                        "vehicle_count": payload["density_stats"]["vehicle_count"]
                    })
                    if len(state.history_records) > 40:
                        state.history_records.pop(0)

            # Encode JPEG phát luồng MJPEG
            success, jpeg = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if success:
                state.current_frame_bytes = jpeg.tobytes()

        except Exception:
            time.sleep(0.03)


# -----------------------------------------------------------------------------
# 4. Tạo App FastAPI & Lifespan Context Manager
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi động worker thread khi server chạy."""
    worker = threading.Thread(target=video_worker_loop, daemon=True)
    worker.start()
    yield


app = FastAPI(
    title="UAV Traffic Intelligent Control Center",
    description="Hệ thống giám sát, phân tích mật độ phương tiện UAV & điều khiển đèn giao thông thích ứng",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))


# -----------------------------------------------------------------------------
# 5. Pydantic Request Models
# -----------------------------------------------------------------------------
class ControlAction(BaseModel):
    action: str  # "run", "pause", "resume", "stop", "reset"


class ConfigUpdate(BaseModel):
    video_source: Optional[str] = None
    tracker: Optional[str] = None
    conf: Optional[float] = None
    iou: Optional[float] = None
    line_y: Optional[int] = None
    ppm: Optional[float] = None
    resolution: Optional[str] = None


class ProfileRequest(BaseModel):
    vehicles: int
    ocr: float
    avg_speed: float
    stopped_ratio: float
    congestion_index: float
    state_text: str


# -----------------------------------------------------------------------------
# 6. REST API Endpoints
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Trang chủ Web Dashboard."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "sample_videos": SAMPLE_VIDEOS,
            "current_video": state.video_path,
            "video_meta": state.video_meta,
            "tracker_name": state.tracker_name
        }
    )


def generate_mjpeg_stream():
    """Generator stream MJPEG cho video_feed."""
    last_sent = None
    while True:
        frame_bytes = state.current_frame_bytes
        if frame_bytes is not None and frame_bytes is not last_sent:
            last_sent = frame_bytes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.015)


@app.get("/video_feed")
def video_feed():
    """Luồng video MJPEG thời gian thực."""
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/metrics")
def get_metrics():
    """Lấy toàn bộ số liệu thống kê realtime."""
    with state.lock:
        return JSONResponse({
            "is_running": state.is_running,
            "is_paused": state.is_paused,
            "stats": state.latest_stats,
            "counts": state.latest_counts,
            "line_counts": state.latest_line_counts,
            "alerts": state.latest_alerts,
            "history": state.history_records[-20:],
            "video_meta": state.video_meta
        })


@app.post("/api/control")
def control_playback(cmd: ControlAction):
    """Điều khiển trạng thái phát video (Run, Pause, Resume, Stop, Reset)."""
    with state.lock:
        if cmd.action in ("run", "resume"):
            state.is_running = True
            state.is_paused = False
        elif cmd.action == "pause":
            state.is_running = False
            state.is_paused = True
        elif cmd.action in ("stop", "reset"):
            state.is_running = False
            state.is_paused = False
            if state.cap:
                state.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if cmd.action == "reset":
                state.reset_metrics()

    return {
        "status": "ok",
        "action": cmd.action,
        "is_running": state.is_running,
        "is_paused": state.is_paused
    }


@app.post("/api/config")
def update_configuration(cfg: ConfigUpdate):
    """Cập nhật các tham số AI và nguồn video."""
    target_video = None
    if cfg.video_source:
        if cfg.video_source in SAMPLE_VIDEOS:
            target_video = SAMPLE_VIDEOS[cfg.video_source]
        elif cfg.video_source in SAMPLE_VIDEOS.values():
            target_video = cfg.video_source
        elif os.path.exists(cfg.video_source):
            target_video = cfg.video_source
        elif os.path.exists(str(BASE_DIR / cfg.video_source)):
            target_video = str(BASE_DIR / cfg.video_source)

    if cfg.resolution:
        state.target_w = 1920 if "1080p" in cfg.resolution else (960 if "540p" in cfg.resolution else 1280)

    state.update_config(
        tracker=cfg.tracker,
        conf=cfg.conf,
        iou=cfg.iou,
        line_y=cfg.line_y,
        ppm=cfg.ppm,
        video_path=target_video
    )
    return {
        "status": "ok",
        "video_meta": state.video_meta,
        "current_video": state.video_path
    }


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Upload file video mới và tự động chạy stream."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    state.update_config(video_path=str(file_path))
    with state.lock:
        state.is_running = True
        state.is_paused = False

    return {
        "status": "ok",
        "filename": file.filename,
        "video_meta": state.video_meta,
        "is_running": True
    }


@app.post("/api/llm_profile")
def generate_llm_profile(req: ProfileRequest):
    """Gọi LLM tư vấn phương án chu kỳ đèn tín hiệu giao thông thích ứng."""
    advisor = TrafficAdvisor()
    stats_input = {
        "vehicle_count": req.vehicles,
        "occupancy_rate": req.ocr,
        "avg_speed": req.avg_speed,
        "avg_speed_kmh": req.avg_speed,
        "stopped_ratio": req.stopped_ratio,
        "congestion_index": req.congestion_index,
        "congestion_level": req.state_text
    }
    profile_text = advisor.generate_control_profile(stats_input, state.latest_counts)
    return {"status": "ok", "profile": profile_text}


@app.get("/api/download_csv")
def download_history_csv():
    """Xuất file CSV chứa dữ liệu lịch sử đo đạc."""
    records = state.history_records if state.history_records else [{
        "time": time.strftime("%H:%M:%S"),
        "congestion_index": 0,
        "occupancy_rate": 0,
        "avg_speed": 0,
        "vehicle_count": 0
    }]
    df = pd.DataFrame(records)
    stream = io.StringIO()
    df.to_csv(stream, index=False)

    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=uav_traffic_history.csv"}
    )


# -----------------------------------------------------------------------------
# 7. Khởi Chạy Web Server
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8501, reload=False)
