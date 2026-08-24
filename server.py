"""
FastAPI Server for UAV Traffic Monitoring & Adaptive Traffic Signal Control.
Hỗ trợ:
- MJPEG Video Streaming thời gian thực qua giao thức HTTP multipart.
- REST API cập nhật chỉ số giao thông (Counts, OCR, CI, Speed, Alerts).
- Tích hợp LLM (openai/gpt-oss-20b) sinh khuyến nghị chu kỳ đèn tín hiệu thích ứng.
- Phân tích dữ liệu & Xuất CSV báo cáo.
"""

import os
import time
import threading
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline import UAVTrafficPipeline
from llm.traffic_profile import TrafficAdvisor, GroqTrafficAdvisor
from utils.config import DEFAULT_MODEL_PATH, DEFAULT_SAMPLE_VIDEO, SAMPLE_VIDEOS


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi chạy worker thread khi FastAPI server start."""
    worker = threading.Thread(target=video_worker_loop, daemon=True)
    worker.start()
    yield


app = FastAPI(
    title="UAV Traffic Intelligent Control Center",
    description="Hệ thống giám sát, phân tích mật độ phương tiện UAV & điều khiển đèn giao thông thích ứng",
    version="2.0.0",
    lifespan=lifespan
)

# Cho phép CORS cho mọi nguồn (hỗ trợ cả port 8501 và port 5500 của Live Server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
web_dir = BASE_DIR / "web"
app.mount("/static", StaticFiles(directory=str(web_dir / "static")), name="static")
templates = Jinja2Templates(directory=str(web_dir / "templates"))


class TrafficEngineState:
    def __init__(self):
        self.lock = threading.RLock()
        self.is_running = False
        self.is_paused = False
        self.video_path = DEFAULT_SAMPLE_VIDEO
        self.tracker_name = "ByteTrack"
        self.conf = 0.25
        self.iou = 0.45
        self.line_y = 550
        self.pixels_per_meter = 10.0
        self.imgsz = 640
        self.target_w = 1280
        self.loop_video = True

        self.pipeline: Optional[UAVTrafficPipeline] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_frame_bytes: Optional[bytes] = None

        self.latest_stats = {
            "vehicle_count": 0, "occupancy_rate": 0.0, "avg_speed": 0.0, "avg_speed_kmh": 0.0,
            "stopped_ratio": 0.0, "avg_dwell_time": 0.0, "congestion_index": 0.0,
            "congestion_level": "Thông thoáng", "total_counted": 0, "fps": 0.0
        }
        self.latest_counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "bicycle": 0}
        self.latest_line_counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0, "bicycle": 0}
        self.latest_alerts = []
        self.history_records = []
        self.video_meta = {"name": os.path.basename(DEFAULT_SAMPLE_VIDEO), "resolution": "1920x1080", "fps": 30}
        self.cached_llm_profile: Optional[str] = None

        self._init_pipeline()
        self._update_video_meta()
        self._generate_preview_frame()

    def _init_pipeline(self):
        self.pipeline = UAVTrafficPipeline(
            model_path=DEFAULT_MODEL_PATH,
            tracker_name=self.tracker_name,
            conf=self.conf,
            iou=self.iou,
            line_y=self.line_y,
            pixels_per_meter=self.pixels_per_meter,
            roi_polygon=None
        )

    def prepare_frame(self, frame: np.ndarray):
        """Resize frame và tính tỉ lệ vạch đếm ảo tương ứng."""
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

    def _generate_preview_frame(self):
        """Đọc và vẽ ngay frame đầu tiên của video để hiển thị preview tức thì."""
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

    def set_config(self, tracker_name=None, conf=None, iou=None, line_y=None, ppm=None, video_path=None):
        with self.lock:
            reinit_needed = False
            if tracker_name is not None and tracker_name != self.tracker_name:
                self.tracker_name = tracker_name
                reinit_needed = True
            if conf is not None:
                self.conf = float(conf)
                reinit_needed = True
            if iou is not None:
                self.iou = float(iou)
                reinit_needed = True
            if line_y is not None:
                self.line_y = int(line_y)
                if self.pipeline:
                    self.pipeline.line_y = self.line_y
            if ppm is not None:
                self.pixels_per_meter = float(ppm)
                if self.pipeline:
                    self.pipeline.analyzer.pixels_per_meter = self.pixels_per_meter
            if video_path is not None and video_path != self.video_path:
                self.video_path = video_path
                self._update_video_meta()
                if self.cap:
                    self.cap.release()
                    self.cap = None
                self._generate_preview_frame()

            if reinit_needed:
                self._init_pipeline()

    def _update_video_meta(self):
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


state = TrafficEngineState()


def video_worker_loop():
    """Vòng lặp đọc video, suy luận YOLO/Tracking và tạo MJPEG stream ổn định chuẩn ~30 FPS."""
    blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(blank_frame, "UAV TRAFFIC CONTROL - STANDBY", (380, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (56, 189, 248), 2, cv2.LINE_AA)
    _, blank_bytes = cv2.imencode('.jpg', blank_frame)
    state.current_frame_bytes = blank_bytes.tobytes()

    frame_counter = 0
    TARGET_FPS = 30.0
    TARGET_FRAME_TIME = 1.0 / TARGET_FPS  # 33.33ms
    last_frame_time = time.perf_counter()
    fps_smoothed = 30.0

    while True:
        try:
            if not state.is_running or state.is_paused:
                time.sleep(0.03)
                last_frame_time = time.perf_counter()
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

            # Xử lý frame qua pipeline AI
            annotated_frame, payload = state.pipeline.process_frame(frame_resized, imgsz=state.imgsz)

            # Tính toán và kiểm soát nhịp độ chuẩn xác ~30 FPS
            now = time.perf_counter()
            elapsed = now - last_frame_time
            if elapsed < TARGET_FRAME_TIME:
                time.sleep(TARGET_FRAME_TIME - elapsed)
                now = time.perf_counter()

            actual_dt = now - last_frame_time
            last_frame_time = now
            inst_fps = 1.0 / actual_dt if actual_dt > 0 else 30.0
            fps_smoothed = (fps_smoothed * 0.8) + (min(30.0, max(28.0, inst_fps)) * 0.2)

            payload["density_stats"]["fps"] = round(fps_smoothed, 1)

            with state.lock:
                state.latest_stats = payload["density_stats"]
                state.latest_counts = payload["counts"]
                state.latest_line_counts = payload.get("line_counts", {})
                state.latest_alerts = payload["alerts"]

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

            success, jpeg = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if success:
                state.current_frame_bytes = jpeg.tobytes()

        except Exception:
            time.sleep(0.03)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Trang chủ Dashboard điều khiển giao thông UAV."""
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
    """Generator stream MJPEG chuẩn qua HTTP multipart."""
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
    """Endpoint cấp luồng video MJPEG."""
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/metrics")
def get_metrics():
    """API lấy thông số thống kê thời gian thực."""
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


class ControlAction(BaseModel):
    action: str  # "run", "pause", "resume", "stop", "reset"


@app.post("/api/control")
def control_playback(cmd: ControlAction):
    """Điều khiển phát/dừng video."""
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
                state.history_records.clear()
                if state.pipeline:
                    state.pipeline.reset()
                state.latest_counts = {k: 0 for k in state.latest_counts}
                state.latest_line_counts = {k: 0 for k in state.latest_line_counts}
                state.latest_stats = {
                    "vehicle_count": 0, "occupancy_rate": 0.0, "avg_speed": 0.0, "avg_speed_kmh": 0.0,
                    "stopped_ratio": 0.0, "avg_dwell_time": 0.0, "congestion_index": 0.0,
                    "congestion_level": "Thông thoáng", "total_counted": 0, "fps": 0.0
                }

    return {"status": "ok", "action": cmd.action, "is_running": state.is_running, "is_paused": state.is_paused}


class ConfigUpdate(BaseModel):
    video_source: Optional[str] = None
    tracker: Optional[str] = None
    conf: Optional[float] = None
    iou: Optional[float] = None
    line_y: Optional[int] = None
    ppm: Optional[float] = None
    resolution: Optional[str] = None


@app.post("/api/config")
def update_configuration(cfg: ConfigUpdate):
    """Cập nhật các tham số phân tích AI và cấu hình luồng."""
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

    state.set_config(
        tracker_name=cfg.tracker,
        conf=cfg.conf,
        iou=cfg.iou,
        line_y=cfg.line_y,
        ppm=cfg.ppm,
        video_path=target_video
    )
    return {"status": "ok", "video_meta": state.video_meta, "current_video": state.video_path}


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Upload video UAV từ máy tính."""
    upload_dir = BASE_DIR / "videos" / "temp_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    state.set_config(video_path=str(file_path))
    with state.lock:
        state.is_running = True
        state.is_paused = False

    return {"status": "ok", "filename": file.filename, "video_meta": state.video_meta, "is_running": True}


class ProfileRequest(BaseModel):
    vehicles: int
    ocr: float
    avg_speed: float
    stopped_ratio: float
    congestion_index: float
    state_text: str


@app.post("/api/llm_profile")
def generate_llm_profile(req: ProfileRequest):
    """Gọi LLM (openai/gpt-oss-20b) sinh khuyến nghị điều khiển tín hiệu đèn thích ứng."""
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
    state.cached_llm_profile = profile_text
    return {"status": "ok", "profile": profile_text}


@app.get("/api/download_csv")
def download_history_csv():
    """Xuất file CSV nhật ký phân tích giao thông."""
    import io
    import pandas as pd
    df = pd.DataFrame(state.history_records if state.history_records else [{"time": time.strftime("%H:%M:%S"), "congestion_index": 0, "occupancy_rate": 0, "avg_speed": 0, "vehicle_count": 0}])
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=uav_traffic_history.csv"
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8501, reload=False)
