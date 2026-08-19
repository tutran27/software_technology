# 📋 KẾ HOẠCH TRIỂN KHAI DỰ ÁN

## Phát hiện, theo dõi và phân tích mật độ phương tiện từ video UAV — Bản Demo

---

## 1. Tổng quan dự án

**Mục tiêu:** Xây dựng bản demo hoàn chỉnh với giao diện dashboard, cho phép:

- Upload video UAV → Phát hiện phương tiện (Car, Motorcycle, Bus, Truck) bằng YOLO11
- Theo dõi phương tiện bằng ByteTrack (duy trì ID, tránh đếm trùng)
- Đếm phương tiện qua virtual line / ROI
- Phân tích mật độ giao thông theo thời gian thực
- Cảnh báo khi mật độ vượt ngưỡng
- Tích hợp LLM (Groq API) sinh Traffic Control Profile — khuyến nghị điều khiển đèn giao thông

**Nguyên tắc:** Code đơn giản, dễ hiểu, chạy được demo trên máy cá nhân.

---

## 2. Công nghệ sử dụng

| Thành phần | Công nghệ | Lý do chọn |
|---|---|---|
| Ngôn ngữ | **Python 3.10+** | Hệ sinh thái AI/ML mạnh |
| Detection | **Ultralytics YOLO11** (`best.pt`) | One-stage detector nhanh, chính xác |
| Tracking | **ByteTrack + BoT-SORT** (qua Ultralytics built-in) | Cả 2 đều tích hợp sẵn, chuyển đổi qua config |
| Xử lý video | **OpenCV** | Đọc/ghi frame, vẽ bounding box |
| LLM | **Groq API** (model: `llama-3.1-70b-versatile`) | Miễn phí, tốc độ cực nhanh, chất lượng tốt |
| Dashboard | **Streamlit** | Nhanh nhất để tạo demo, không cần frontend riêng |
| Biểu đồ | **Plotly** | Tương tác, đẹp, tích hợp tốt với Streamlit |
| Quản lý config | **Python dataclass / dict** | Đơn giản, không cần DB |

---

## 3. Kiến trúc hệ thống

```
📁 uav-traffic-analyzer/
│
├── 📁 models/
│   └── best.pt                    # Model YOLO11 đã train
│
├── 📁 videos/
│   └── sample.mp4                 # Video UAV mẫu để demo
│
├── 📁 core/
│   ├── __init__.py
│   ├── detector.py                # Module phát hiện (YOLO11)
│   ├── tracker.py                 # Module theo dõi (ByteTrack / BoT-SORT chuyển đổi)
│   ├── counter.py                 # Module đếm (Virtual Line / ROI)
│   ├── analyzer.py                # Module phân tích mật độ
│   └── alert.py                   # Module cảnh báo
│
├── 📁 llm/
│   ├── __init__.py
│   └── traffic_profile.py         # Tích hợp Groq LLM
│
├── 📁 utils/
│   ├── __init__.py
│   ├── drawing.py                 # Vẽ bounding box, trail, thông tin lên frame
│   └── config.py                  # Cấu hình hệ thống (ngưỡng, ROI, ...)
│
├── app.py                         # 🚀 Streamlit Dashboard (entry point)
├── pipeline.py                    # Pipeline xử lý video chính
├── benchmark_tracking.py          # Script so sánh ByteTrack vs BoT-SORT
├── requirements.txt               # Dependencies
├── .env                           # GROQ_API_KEY
└── README.md                      # Hướng dẫn cài đặt & chạy
```

### Luồng xử lý dữ liệu (Pipeline)

```
Video UAV (.mp4)
    │
    ▼
┌─────────────────────┐
│  1. Đọc từng frame   │  ← OpenCV VideoCapture
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  2. YOLO11 Detect    │  ← best.pt → bounding boxes + class + confidence
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  3. Tracker          │  ← ByteTrack HOẶC BoT-SORT (chọn qua sidebar)
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  4. Vehicle Counter  │  ← Đếm khi tâm bbox vượt qua virtual line
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  5. Density Analyzer │  ← Tính mật độ, lưu lượng theo thời gian
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  6. Alert Engine     │  ← Kiểm tra ngưỡng → cảnh báo
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  7. Groq LLM API    │  ← Sinh Traffic Control Profile
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  8. Streamlit UI     │  ← Dashboard hiển thị tất cả
└─────────────────────┘
```

---

## 4. Chi tiết từng Module

### 4.1. `core/detector.py` — Phát hiện phương tiện

**Nhiệm vụ:** Load model YOLO11, chạy inference trên từng frame.

```python
# Ý tưởng code chính
from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path="models/best.pt", conf=0.25):
        self.model = YOLO(model_path)
        self.conf = conf
        self.class_names = {0: "Car", 1: "Motorcycle", 2: "Bus", 3: "Truck"}

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)
        return results[0]  # trả về Result object
```

**Đầu ra:** Danh sách bounding boxes với class, confidence cho mỗi frame.

---

### 4.2. `core/tracker.py` — Theo dõi phương tiện (ByteTrack / BoT-SORT)

**Nhiệm vụ:** Gán Track ID cho phương tiện, hỗ trợ **chuyển đổi** giữa ByteTrack và BoT-SORT.

#### Cơ chế chuyển đổi tracker

Ultralytics tích hợp sẵn cả ByteTrack và BoT-SORT — chỉ cần đổi tham số `tracker=` là chuyển thuật toán. **Cách hợp lý nhất để chuyển đổi là qua Streamlit sidebar** (selectbox), vì:

- ✅ Người dùng chọn trực quan trên giao diện, không cần sửa code hay restart
- ✅ Phù hợp demo — bấm chuyển tracker, chạy lại video, so sánh ngay
- ✅ Nội bộ chỉ thay đổi 1 string: `"bytetrack.yaml"` ↔ `"botsort.yaml"`

> **Tại sao không dùng `.env` hay file config?** Vì mỗi lần đổi phải restart app, không tiện cho demo và so sánh trực quan. `.env` phù hợp cho API key (ít thay đổi), còn tracker cần chuyển qua lại liên tục khi đánh giá.

```python
# Ý tưởng code chính
from ultralytics import YOLO

# Tracker name mapping — chỉ cần đổi string
TRACKER_CONFIGS = {
    "ByteTrack": "bytetrack.yaml",
    "BoT-SORT":  "botsort.yaml",
}

class VehicleTracker:
    def __init__(self, model_path="models/best.pt", tracker_name="ByteTrack", conf=0.25):
        self.model = YOLO(model_path)
        self.tracker_config = TRACKER_CONFIGS[tracker_name]
        self.tracker_name = tracker_name
        self.conf = conf

    def track(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,   # ← đổi tracker tại đây
            conf=self.conf,
            verbose=False
        )
        return results[0]

    def reset(self):
        """Reset tracker state khi chuyển đổi hoặc chạy video mới"""
        self.model = YOLO(self.model.model_name)
```

#### Trên Streamlit sidebar

```python
# Trong app.py — sidebar
tracker_name = st.sidebar.selectbox(
    "🔄 Chọn Tracker",
    options=["ByteTrack", "BoT-SORT"],
    index=0,
    help="Chuyển đổi thuật toán tracking để so sánh hiệu quả"
)
# Khi user đổi tracker → reset pipeline với tracker mới
pipeline = TrafficPipeline(model_path="models/best.pt", tracker_name=tracker_name)
```

> **Lưu ý quan trọng:** Khi chuyển tracker, phải **reset model** (tạo lại instance YOLO) để xóa trạng thái tracking cũ. Nếu không, ID từ tracker cũ sẽ xung đột với tracker mới.

**Đầu ra:** Bounding boxes + Track ID + Class cho mỗi phương tiện.

---

### 4.3. `core/counter.py` — Đếm phương tiện

**Nhiệm vụ:** Đếm phương tiện khi tâm bbox vượt qua virtual line.

```python
# Ý tưởng code chính
class VehicleCounter:
    def __init__(self, line_y):
        """line_y: tọa độ y của virtual line ngang"""
        self.line_y = line_y
        self.counted_ids = set()        # Đã đếm (tránh trùng)
        self.prev_positions = {}        # Track ID → y trước đó
        self.counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}

    def update(self, track_id, class_name, cx, cy):
        if track_id in self.counted_ids:
            return False

        prev_y = self.prev_positions.get(track_id)
        self.prev_positions[track_id] = cy

        # Kiểm tra: tâm bbox đã vượt qua line chưa?
        if prev_y is not None:
            if (prev_y < self.line_y <= cy) or (prev_y > self.line_y >= cy):
                self.counted_ids.add(track_id)
                self.counts[class_name] = self.counts.get(class_name, 0) + 1
                return True  # Đã đếm thành công
        return False
```

**Đầu ra:** Bộ đếm theo từng loại phương tiện.

---

### 4.4. `core/analyzer.py` — Phân tích mật độ & Nhận diện tắc nghẽn

**Nhiệm vụ:** Tính toán mật độ, tỷ lệ chiếm dụng mặt đường, vận tốc dòng xe, thời gian chờ và xác định chỉ số tắc nghẽn (Congestion Index).

#### 💡 Nguyên tắc nhận diện tắc nghẽn thông minh (Đường lớn vs Đường nhỏ)
> Không thể chỉ đếm số lượng xe thô (vì 20 xe ở ngõ nhỏ là tắc cứng, nhưng trên đại lộ 6 làn là thông thoáng; hoặc 50 xe đang chạy nhanh 50 km/h thì không phải tắc).

Hệ thống kết hợp **3 cơ chế** từ dữ liệu ByteTrack/BoT-SORT:

1. **Tỷ lệ chiếm dụng diện tích mặt đường (Occupancy Rate - $OCR$):**
   $$OCR = \frac{\sum \text{Diện tích BBox các xe trong ROI}}{\text{Diện tích ROI}} \times 100\%$$
   - *Tự thích ứng đường lớn / nhỏ*: Đường nhỏ có ROI bé → ít xe nhưng OCR đã cao → phát hiện tắc ngay.
   - *Tự cân bằng kích thước xe*: 1 xe Bus/Truck to chiếm diện tích nhiều hơn xe máy.

2. **Vận tốc trung bình & Tỷ lệ xe dừng (Speed & Stationary Ratio):**
   - Vận tốc từng xe $v_i = \frac{\Delta d}{\Delta t}$ (pixel/s hoặc m/s nếu calibrated).
   - Tỷ lệ xe đứng yên: $R_{stop} = \frac{\text{Số xe có } v_i < v_{ngưỡng}}{\text{Tổng số xe trong ROI}} \times 100\%$.
   - Xe đông nhưng chạy nhanh ($R_{stop} \approx 0\%$) → **Không tắc**. Xe đông mà đứng im ($R_{stop} > 60\%$) → **TẮC ĐƯỜNG**.

3. **Thời gian lưu lại / chờ trong vùng (Dwell Time):**
   - Theo dõi thời gian Track ID $i$ tồn tại trong ROI: $T_{wait} = t_{now} - t_{entry}$.
   - Nếu đa số xe lưu lại quá lâu (ví dụ > 45s–60s) mà không thoát khỏi ROI → **Dòng xe đang bị nghẽn (Gridlock)**.

4. **Chỉ số tắc nghẽn tổng hợp (Congestion Index - $CI \in [0, 100]$):**
   $$CI = 0.4 \times OCR + 0.3 \times \left(1 - \frac{\bar{v}}{v_{max}}\right) \times 100 + 0.3 \times R_{stop}$$

| Mức độ $CI$ | Trạng thái | Hành động khuyến nghị cho đèn tín hiệu |
|:---:|:---:|---|
| **0 – 30** | 🟢 **Thông thoáng** | Giữ chu kỳ cơ bản / giảm thời gian đèn xanh |
| **31 – 60** | 🟡 **Bình thường** | Duy trì chu kỳ hiện tại |
| **61 – 80** | 🟠 **Đông / Di chuyển chậm** | Tăng +10s đèn xanh hướng này |
| **81 – 100** | 🔴 **Ùn tắc nghiêm trọng** | Ưu tiên tối đa đèn xanh (+20s–30s), cảnh báo điều phối |

```python
# Ý tưởng code chính
import time
import numpy as np
from collections import deque

class TrafficAnalyzer:
    def __init__(self, time_window=30, roi_polygon=None, stop_speed_threshold=2.0):
        """
        time_window: cửa sổ thời gian trượt (giây)
        roi_polygon: tọa độ vùng ROI [(x1,y1), (x2,y2), ...] (None = toàn frame)
        stop_speed_threshold: ngưỡng vận tốc coi là xe dừng (pixels/frame)
        """
        self.time_window = time_window
        self.stop_speed_threshold = stop_speed_threshold
        self.track_history = {}     # track_id -> deque of (timestamp, cx, cy)
        self.track_entry_time = {}   # track_id -> entry timestamp
        self.history = deque()      # lưu lịch sử stats theo thời gian

    def update_tracks(self, tracked_objects, frame_shape):
        """
        tracked_objects: list các dict {id, class, bbox: [x1,y1,x2,y2], center: [cx,cy]}
        """
        now = time.time()
        frame_h, frame_w = frame_shape[:2]
        roi_area = frame_h * frame_w  # hoặc diện tích đa giác ROI

        total_bbox_area = 0
        speeds = []
        stopped_count = 0
        dwell_times = []

        current_ids = set()

        for obj in tracked_objects:
            tid = obj["id"]
            x1, y1, x2, y2 = obj["bbox"]
            cx, cy = obj["center"]
            current_ids.add(tid)

            # 1. Diện tích chiếm dụng
            bbox_area = (x2 - x1) * (y2 - y1)
            total_bbox_area += bbox_area

            # 2. Ghi nhận thời gian vào ROI
            if tid not in self.track_entry_time:
                self.track_entry_time[tid] = now
            dwell_times.append(now - self.track_entry_time[tid])

            # 3. Tính vận tốc dựa vào lịch sử di chuyển
            if tid not in self.track_history:
                self.track_history[tid] = deque(maxlen=10)
            self.track_history[tid].append((now, cx, cy))

            if len(self.track_history[tid]) >= 2:
                t_old, x_old, y_old = self.track_history[tid][0]
                dt = now - t_old
                if dt > 0:
                    dist = np.sqrt((cx - x_old)**2 + (cy - y_old)**2)
                    speed = dist / dt  # pixels/s
                    speeds.append(speed)
                    if speed < self.stop_speed_threshold * 30:  # ~scale theo fps
                        stopped_count += 1

        # Xóa các ID đã rời khỏi khung hình
        dead_ids = set(self.track_entry_time.keys()) - current_ids
        for did in dead_ids:
            self.track_entry_time.pop(did, None)
            self.track_history.pop(did, None)

        # 4. Tính toán các chỉ số
        num_vehicles = len(tracked_objects)
        ocr = min(100.0, (total_bbox_area / roi_area) * 100.0)
        avg_speed = np.mean(speeds) if speeds else 0.0
        r_stop = (stopped_count / num_vehicles * 100.0) if num_vehicles > 0 else 0.0
        avg_dwell = np.mean(dwell_times) if dwell_times else 0.0

        # 5. Congestion Index (0 - 100)
        # Giả định max_speed chuẩn là 150 px/s
        speed_factor = max(0.0, 1.0 - (avg_speed / 150.0)) * 100.0
        ci = 0.4 * ocr + 0.3 * speed_factor + 0.3 * r_stop
        ci = min(100.0, max(0.0, ci))

        # Phân loại mức độ
        if ci < 30:
            congestion_level = "Thông thoáng"
        elif ci < 60:
            congestion_level = "Bình thường"
        elif ci < 80:
            congestion_level = "Đông đúc"
        else:
            congestion_level = "Ùn tắc nghiêm trọng"

        stats = {
            "vehicle_count": num_vehicles,
            "occupancy_rate": round(ocr, 1),
            "avg_speed": round(avg_speed, 1),
            "stopped_ratio": round(r_stop, 1),
            "avg_dwell_time": round(avg_dwell, 1),
            "congestion_index": round(ci, 1),
            "congestion_level": congestion_level,
        }

        self.history.append((now, stats))
        while self.history and (now - self.history[0][0]) > self.time_window:
            self.history.popleft()

        return stats
```

**Đầu ra:** 
- `occupancy_rate` (% chiếm dụng mặt đường)
- `avg_speed` (vận tốc trung bình)
- `stopped_ratio` (% xe đứng yên)
- `avg_dwell_time` (thời gian chờ trung bình)
- `congestion_index` (Điểm tắc nghẽn 0–100)
- `congestion_level` (Thông thoáng / Bình thường / Đông đúc / Ùn tắc nghiêm trọng)

---

### 4.5. `core/alert.py` — Cảnh báo

**Nhiệm vụ:** Kiểm tra ngưỡng tắc nghẽn, tỷ lệ xe dừng và phát cảnh báo.

```python
# Ý tưởng code chính
class AlertEngine:
    def __init__(self, congestion_threshold=60.0, stop_ratio_threshold=50.0):
        self.congestion_threshold = congestion_threshold
        self.stop_ratio_threshold = stop_ratio_threshold
        self.alerts = []

    def check(self, stats):
        alerts = []
        ci = stats.get("congestion_index", 0)
        r_stop = stats.get("stopped_ratio", 0)
        ocr = stats.get("occupancy_rate", 0)

        # 1. Cảnh báo mức độ tắc nghẽn dựa trên Congestion Index
        if ci >= 80:
            alerts.append({
                "type": "SEVERE_CONGESTION",
                "message": f"🚨 ÙN TẮC NGHIÊM TRỌNG! Chỉ số tắc nghẽn {ci:.0f}/100",
                "severity": "critical"
            })
        elif ci >= self.congestion_threshold:
            alerts.append({
                "type": "MODERATE_CONGESTION",
                "message": f"⚠️ Giao thông đông đúc, có nguy cơ ùn tắc (CI: {ci:.0f}/100)",
                "severity": "warning"
            })

        # 2. Cảnh báo xe dừng hàng loạt (Gridlock)
        if r_stop >= self.stop_ratio_threshold and stats.get("vehicle_count", 0) >= 5:
            alerts.append({
                "type": "GRIDLOCK",
                "message": f"⛔ Phát hiện {r_stop:.0f}% phương tiện đang đứng yên / di chuyển rất chậm!",
                "severity": "critical"
            })

        # 3. Cảnh báo mật độ chiếm dụng mặt đường cao
        if ocr >= 70:
            alerts.append({
                "type": "HIGH_OCCUPANCY",
                "message": f"📦 Mặt đường bị chiếm dụng {ocr:.0f}% diện tích!",
                "severity": "warning"
            })

        self.alerts = alerts
        return alerts
```

---

### 4.6. `llm/traffic_profile.py` — Tích hợp Groq LLM

**Nhiệm vụ:** Gửi dữ liệu thống kê → Groq API → nhận Traffic Control Profile.

```python
# Ý tưởng code chính
import os
from groq import Groq

class TrafficProfileGenerator:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-70b-versatile"

    def generate(self, stats: dict) -> str:
        prompt = f"""
Bạn là chuyên gia phân tích và điều khiển giao thông.
Dựa vào dữ liệu giám sát bên dưới, hãy tạo Traffic Control Profile
gồm: đánh giá tình trạng, khuyến nghị điều chỉnh đèn tín hiệu,
và chu kỳ đèn đề xuất.

=== DỮ LIỆU GIÁM SÁT THỜI GIAN THỰC ===
- Số phương tiện hiện diện trong khung hình: {stats['vehicle_count']}
- Tổng lượt xe đã qua vạch: {stats['total_count']} (Car: {stats.get('Car', 0)}, Moto: {stats.get('Motorcycle', 0)}, Bus: {stats.get('Bus', 0)}, Truck: {stats.get('Truck', 0)})
- Tỷ lệ chiếm dụng mặt đường (Occupancy Rate): {stats.get('occupancy_rate', 0)}%
- Vận tốc trung bình dòng xe: {stats.get('avg_speed', 0)} px/s
- Tỷ lệ phương tiện đứng yên: {stats.get('stopped_ratio', 0)}%
- Thời gian chờ trung bình trong khu vực: {stats.get('avg_dwell_time', 0)} giây
- Điểm tắc nghẽn (Congestion Index): {stats.get('congestion_index', 0)}/100 ({stats.get('congestion_level', 'N/A')})
- Cảnh báo hiện tại: {stats.get('alerts_text', 'Không có')}

Hãy trả lời bằng tiếng Việt với format rõ ràng.
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content
```

**Đầu ra mẫu từ LLM:**

```
📊 TRAFFIC CONTROL PROFILE
═══════════════════════════

🔍 Đánh giá tình trạng:
- Mật độ giao thông: CAO
- Lưu lượng: 45.2 xe/phút
- Xu hướng: Đang tăng
- Phương tiện chủ yếu: Xe máy (65%), Ô tô (25%)

⚙️ Khuyến nghị điều chỉnh:
- Tăng thời gian đèn xanh hướng chính: +15 giây
- Giảm thời gian đèn xanh hướng phụ: -10 giây
- Kích hoạt chế độ ưu tiên xe buýt

🕐 Chu kỳ đèn đề xuất:
- Hướng Bắc–Nam: Xanh 45s | Vàng 3s | Đỏ 42s
- Hướng Đông–Tây: Xanh 30s | Vàng 3s | Đỏ 57s
- Tổng chu kỳ: 90 giây
```

---

### 4.7. `pipeline.py` — Pipeline xử lý chính

**Nhiệm vụ:** Kết nối tất cả module, xử lý video frame-by-frame.

```python
# Ý tưởng code chính
import cv2
from core.tracker import VehicleTracker
from core.counter import VehicleCounter
from core.analyzer import TrafficAnalyzer
from core.alert import AlertEngine

class TrafficPipeline:
    def __init__(self, model_path, tracker_name="ByteTrack", line_y=400):
        self.tracker = VehicleTracker(model_path, tracker_name=tracker_name)
        self.counter = VehicleCounter(line_y)
        self.analyzer = TrafficAnalyzer()
        self.alert_engine = AlertEngine()

    def process_frame(self, frame):
        # 1. Detection + Tracking
        results = self.tracker.track(frame)

        # 2. Lấy thông tin tracked objects
        tracked_objects = []
        if results.boxes.id is not None:
            for box, track_id, cls in zip(
                results.boxes.xyxy,
                results.boxes.id,
                results.boxes.cls
            ):
                x1, y1, x2, y2 = box.tolist()
                tid = int(track_id)
                class_name = self.tracker.model.names[int(cls)]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                # 3. Đếm
                self.counter.update(tid, class_name, cx, cy)
                tracked_objects.append({
                    "id": tid, "class": class_name,
                    "bbox": [x1, y1, x2, y2], "center": [cx, cy]
                })

        # 4. Phân tích mật độ
        self.analyzer.update(len(tracked_objects))
        stats = self.analyzer.get_stats()

        # 5. Kiểm tra cảnh báo
        alerts = self.alert_engine.check(stats)

        return tracked_objects, stats, alerts
```

---

### 4.8. `app.py` — Streamlit Dashboard

**Nhiệm vụ:** Giao diện demo chính.

**Layout Dashboard:**

```
┌──────────────────────────────────────────────────────┐
│  🚁 UAV Traffic Analyzer                             │
├──────────────────────────────────────────────────────┤
│                                                      │
│  [Sidebar]              [Main Area — Tabs]                │
│  ┌──────────┐                                            │
│  │ Upload   │  Tab 1: 🎥 Live Analysis                   │
│  │ Video    │  ┌──────────────────────────────────┐      │
│  │          │  │   Video Player                    │      │
│  │ Cấu hình │  │   (bbox, ID, trail, virtual line) │      │
│  │ - line_y │  └──────────────────────────────────┘      │
│  │ - conf   │  ┌─────┬─────┬─────┬─────┐                │
│  │ - ROI    │  │ Car │Moto │ Bus │Truck│                │
│  │          │  │ 42  │ 128 │  5  │  12 │                │
│  │ 🔄Tracker│  └─────┴─────┴─────┴─────┘                │
│  │[ByteTrack│  ┌────────────────┐ ┌──────────────┐      │
│  │ BoT-SORT]│  │ Biểu đồ mật độ │ │ Cảnh báo     │      │
│  │          │  └────────────────┘ └──────────────┘      │
│  │ Nút chạy │                                            │
│  └──────────┘  Tab 2: 📊 Tracker Comparison              │
│                ┌──────────────────────────────────┐      │
│                │ Bảng: ByteTrack vs BoT-SORT      │      │
│                │ Bar chart: FPS, IDs, Accuracy    │      │
│                │ Line chart: Counting tích lũy    │      │
│                │ Radar chart: đa chiều (nếu có GT)│      │
│                │ Side-by-side video (tùy chọn)    │      │
│                └──────────────────────────────────┘      │
│                                                          │
│                Tab 3: 📋 Traffic Profile                  │
│                ┌──────────────────────────────────┐      │
│                │ 📊 Traffic Control Profile (Groq) │      │
│                │ Khuyến nghị chu kỳ đèn tín hiệu  │      │
│                └──────────────────────────────────┘      │
└──────────────────────────────────────────────────────┘
```

**Tính năng chính của Dashboard:**

| # | Tính năng | Mô tả |
|---|---|---|
| 1 | Upload video | Kéo thả file .mp4 |
| 2 | Video player | Hiển thị video đã xử lý (bbox, ID, trail) |
| 3 | Bộ đếm realtime | 4 metric cards cho 4 loại xe |
| 4 | Biểu đồ mật độ | Line chart theo thời gian (Plotly) |
| 5 | Bảng cảnh báo | Hiển thị cảnh báo khi vượt ngưỡng |
| 6 | Tracker Comparison | So sánh ByteTrack vs BoT-SORT (bảng, bar, line, radar chart) |
| 7 | Traffic Profile | Kết quả từ Groq LLM |
| 8 | Config sidebar | Điều chỉnh tham số (tracker, confidence, line_y, ...) |

---

## 5. Hướng dẫn cài đặt & chạy

### 5.1. Yêu cầu hệ thống

- Python 3.10+
- GPU (khuyến nghị, không bắt buộc — có thể chạy CPU cho demo)
- Groq API Key (miễn phí tại [console.groq.com](https://console.groq.com))

### 5.2. Cài đặt

```bash
# 1. Clone project
git clone <repo-url>
cd uav-traffic-analyzer

# 2. Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Cài dependencies
pip install -r requirements.txt

# 4. Cấu hình API key
echo "GROQ_API_KEY=gsk_your_key_here" > .env

# 5. Đặt model vào thư mục
cp /path/to/best.pt models/best.pt

# 6. Đặt video mẫu
cp /path/to/sample.mp4 videos/sample.mp4
```

### 5.3. File `requirements.txt`

```
ultralytics>=8.2.0
opencv-python>=4.8.0
streamlit>=1.35.0
plotly>=5.18.0
groq>=0.5.0
python-dotenv>=1.0.0
numpy>=1.24.0
lapx>=0.5.0
```

### 5.4. Chạy demo

```bash
streamlit run app.py
```

Mở trình duyệt → `http://localhost:8501`

---

## 6. Kế hoạch triển khai theo giai đoạn

### Phase 1 — Nền tảng (Ngày 1–2)

| Task | Chi tiết | File |
|---|---|---|
| ✅ Khởi tạo project | Tạo cấu trúc thư mục, `requirements.txt` | Toàn bộ |
| ✅ Module Detection | Load `best.pt`, chạy inference | `core/detector.py` |
| ✅ Module Tracking | Tích hợp ByteTrack qua Ultralytics | `core/tracker.py` |
| ✅ Test nhanh | Chạy detection + tracking trên 1 video | `test_basic.py` |

### Phase 2 — Logic nghiệp vụ (Ngày 3–4)

| Task | Chi tiết | File |
|---|---|---|
| ✅ Module Counter | Đếm xe qua virtual line | `core/counter.py` |
| ✅ Module Analyzer | Tính mật độ, lưu lượng, xu hướng | `core/analyzer.py` |
| ✅ Module Alert | Cảnh báo ngưỡng | `core/alert.py` |
| ✅ Pipeline | Kết nối tất cả module | `pipeline.py` |

### Phase 3 — LLM Integration (Ngày 5)

| Task | Chi tiết | File |
|---|---|---|
| ✅ Groq API | Kết nối, tạo prompt, parse kết quả | `llm/traffic_profile.py` |
| ✅ Test LLM | Gửi dữ liệu mẫu, kiểm tra output | Manual test |

### Phase 4 — Dashboard (Ngày 6–7)

| Task | Chi tiết | File |
|---|---|---|
| ✅ Layout cơ bản | Sidebar + Main area | `app.py` |
| ✅ Video player | Hiển thị video đã xử lý | `app.py` |
| ✅ Metric cards | 4 bộ đếm + mật độ | `app.py` |
| ✅ Biểu đồ | Line chart mật độ theo thời gian | `app.py` |
| ✅ Cảnh báo | Hiển thị alert | `app.py` |
| ✅ Traffic Profile | Hiển thị kết quả LLM | `app.py` |

### Phase 5 — Hoàn thiện (Ngày 8)

| Task | Chi tiết | File |
|---|---|---|
| ✅ Vẽ đẹp | Bounding box, trail, virtual line | `utils/drawing.py` |
| ✅ Config | Cho phép chỉnh tham số qua sidebar | `utils/config.py` |
| ✅ README | Hướng dẫn cài đặt, chạy, demo | `README.md` |
| ✅ Test toàn bộ | Chạy end-to-end, quay video demo | Manual |

---

## 7. Ghi chú kỹ thuật quan trọng

### 7.1. Tại sao dùng Groq?

- **Miễn phí** — có free tier đủ cho demo
- **Cực nhanh** — inference speed nhanh nhất hiện tại (vài trăm token/giây)
- **Model mạnh** — Llama 3.1 70B cho output chất lượng
- **API đơn giản** — tương thích OpenAI SDK format

### 7.2. So sánh đối chứng ByteTrack vs BoT-SORT

Đề cương yêu cầu **đối chứng** giữa ByteTrack và BoT-SORT. Dưới đây là kế hoạch chi tiết.

#### 7.2.1. Bảng so sánh lý thuyết

| Tiêu chí | ByteTrack | BoT-SORT |
|---|---|---|
| **Cốt lõi** | Kalman Filter + Hungarian (2 bước association) | Kalman Filter + Hungarian + Camera Motion Compensation (CMC) + ReID |
| **Đặc trưng ngoại hình** | ❌ Không dùng (chỉ dựa vào vị trí + IoU) | ✅ Có (dùng ReID feature để khớp lại đối tượng bị mất) |
| **Bù chuyển động camera** | ❌ Không | ✅ Có (CMC — quan trọng cho video UAV vì camera bay liên tục di chuyển) |
| **Tốc độ** | ⚡ Nhanh hơn (ít tính toán hơn) | 🐢 Chậm hơn (thêm ReID + CMC) |
| **Xử lý che khuất** | Tốt (tận dụng bbox confidence thấp) | Tốt hơn (có ReID khớp lại sau khi mất) |
| **Phù hợp UAV** | Tốt cho UAV ổn định, ít rung | Tốt hơn cho UAV di chuyển nhiều, rung lắc |
| **Tích hợp Ultralytics** | ✅ `bytetrack.yaml` | ✅ `botsort.yaml` |

#### 7.2.2. Cách chạy thí nghiệm so sánh

**Nguyên tắc:** Cùng video, cùng model detection (`best.pt`), cùng tham số confidence → chỉ đổi tracker.

```python
# benchmark_tracking.py — Script so sánh 2 tracker
import cv2
import time
import json
from core.tracker import VehicleTracker, TRACKER_CONFIGS
from core.counter import VehicleCounter

def benchmark_tracker(video_path, model_path, tracker_name, line_y=400):
    """Chạy 1 tracker trên 1 video, thu thập metrics"""
    tracker = VehicleTracker(model_path, tracker_name=tracker_name)
    counter = VehicleCounter(line_y)

    cap = cv2.VideoCapture(video_path)
    fps_list = []
    total_ids = set()
    id_switches = 0     # Đếm ID switch (ID đổi bất thường)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t_start = time.time()
        results = tracker.track(frame)
        t_end = time.time()

        fps_list.append(1.0 / (t_end - t_start + 1e-6))

        if results.boxes.id is not None:
            for box, tid, cls in zip(results.boxes.xyxy, results.boxes.id, results.boxes.cls):
                tid = int(tid)
                total_ids.add(tid)
                cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                counter.update(tid, tracker.model.names[int(cls)], cx, cy)

        frame_count += 1

    cap.release()

    return {
        "tracker": tracker_name,
        "total_frames": frame_count,
        "unique_ids": len(total_ids),         # Số ID duy nhất (nhiều → nhiều ID switch)
        "vehicle_counts": dict(counter.counts),
        "total_counted": sum(counter.counts.values()),
        "avg_fps": sum(fps_list) / len(fps_list),
        "min_fps": min(fps_list),
        "max_fps": max(fps_list),
    }

# Chạy benchmark
video = "videos/sample.mp4"
model = "models/best.pt"

result_bytetrack = benchmark_tracker(video, model, "ByteTrack")
result_botsort   = benchmark_tracker(video, model, "BoT-SORT")

# Lưu kết quả
with open("benchmark_results.json", "w") as f:
    json.dump([result_bytetrack, result_botsort], f, indent=2, ensure_ascii=False)
```

#### 7.2.3. Các metrics đánh giá

| Metric | Ý nghĩa | Cách tính | Tracker tốt hơn khi... |
|---|---|---|---|
| **Unique IDs** | Tổng số Track ID phát sinh | `len(set(all_track_ids))` | Gần với số phương tiện thực tế (ít ID thì ít bị phân mảnh) |
| **ID Fragmentation** | Tỷ lệ ID bị phân mảnh | `unique_ids / ground_truth_vehicles` | Gần 1.0 (= mỗi xe 1 ID) |
| **Counting Accuracy** | Sai số đếm | `abs(counted - ground_truth) / ground_truth` | Gần 0 |
| **Avg FPS** | Tốc độ xử lý trung bình | `frames / total_time` | Cao hơn |
| **MOTA** *(nếu có GT)* | Multi-Object Tracking Accuracy | `1 - (FN + FP + ID_sw) / GT` | Gần 1.0 |
| **IDF1** *(nếu có GT)* | ID F1-Score | Harmonic mean of ID Precision & ID Recall | Gần 1.0 |

> **Nếu có ground-truth tracking data** (VisDrone-MOT / UAVDT): dùng thư viện `motmetrics` hoặc `TrackEval` để tính MOTA, IDF1, HOTA tự động.  
> **Nếu không có ground-truth:** so sánh bằng unique_ids, counting accuracy (đếm tay), FPS.

#### 7.2.4. Visualize kết quả so sánh

**A. Bảng so sánh tổng hợp trên Dashboard**

```python
# Trong app.py — Tab "So sánh Tracker"
import streamlit as st
import pandas as pd

results = [result_bytetrack, result_botsort]
df = pd.DataFrame(results)

# Bảng so sánh
st.subheader("📊 So sánh ByteTrack vs BoT-SORT")
st.dataframe(
    df[["tracker", "unique_ids", "total_counted", "avg_fps"]],
    use_container_width=True,
    hide_index=True
)
```

**B. Bar chart so sánh từng metric**

```
       So sánh Counting Accuracy              So sánh FPS
  ┌─────────────────────────────┐   ┌─────────────────────────────┐
  │  ByteTrack ████████████ 187 │   │  ByteTrack ██████████████ 32│
  │  BoT-SORT  ███████████  183 │   │  BoT-SORT  ██████████    24│
  │  GT        ███████████  185 │   │                             │
  └─────────────────────────────┘   └─────────────────────────────┘

       So sánh Unique IDs                So sánh theo loại xe
  ┌─────────────────────────────┐   ┌─────────────────────────────┐
  │  ByteTrack ██████████   210 │   │     Car  Moto  Bus  Truck   │
  │  BoT-SORT  ████████    195 │   │  BT  42   128    5    12    │
  │  (ít hơn = ít ID switch)   │   │  BS  40   125    5    13    │
  └─────────────────────────────┘   └─────────────────────────────┘
```

```python
# Plotly grouped bar chart
import plotly.graph_objects as go

fig = go.Figure(data=[
    go.Bar(name='ByteTrack', x=['Unique IDs', 'Total Counted', 'Avg FPS'],
           y=[210, 187, 32], marker_color='#4C78A8'),
    go.Bar(name='BoT-SORT',  x=['Unique IDs', 'Total Counted', 'Avg FPS'],
           y=[195, 183, 24], marker_color='#F58518'),
])
fig.update_layout(barmode='group', title='ByteTrack vs BoT-SORT')
st.plotly_chart(fig, use_container_width=True)
```

**C. Side-by-side Video Replay** (nâng cao, tùy chọn)

Xử lý cùng 1 video với 2 tracker, lưu 2 output video → hiển thị cạnh nhau:

```python
# Trong app.py
col1, col2 = st.columns(2)
with col1:
    st.markdown("**ByteTrack**")
    st.image(frame_bytetrack, channels="BGR")
with col2:
    st.markdown("**BoT-SORT**")
    st.image(frame_botsort, channels="BGR")
```

**D. Biểu đồ Counting theo thời gian (Line chart)**

So sánh tích lũy đếm xe theo frame — tracker nào đếm ổn định hơn:

```
  Tích lũy đếm xe theo frame
  200 ┤
      │                          ╱── BoT-SORT
  150 ┤                    ╱───╱
      │              ╱───╱╱
  100 ┤         ╱──╱╱──── ByteTrack
      │    ╱──╱╱
   50 ┤ ╱╱╱
      │╱
    0 ┼─────────────────────────────
      0    200   400   600   800  1000
                   Frame
```

```python
# Plotly line chart so sánh counting tích lũy
import plotly.express as px

fig = px.line(
    df_cumulative,     # columns: frame, bytetrack_count, botsort_count
    x="frame",
    y=["bytetrack_count", "botsort_count"],
    title="Tích lũy đếm phương tiện: ByteTrack vs BoT-SORT",
    labels={"value": "Số xe đếm được", "frame": "Frame"},
)
st.plotly_chart(fig, use_container_width=True)
```

**E. Radar Chart — So sánh đa chiều** (nếu có GT metrics)

```python
# So sánh MOTA, IDF1, FPS, Counting Accuracy trên radar chart
import plotly.graph_objects as go

categories = ['MOTA', 'IDF1', 'FPS (norm)', 'Count Acc', 'ID Stability']
fig = go.Figure()
fig.add_trace(go.Scatterpolar(
    r=[0.72, 0.68, 1.0, 0.95, 0.85],  # ByteTrack (normalized)
    theta=categories, fill='toself', name='ByteTrack'
))
fig.add_trace(go.Scatterpolar(
    r=[0.78, 0.75, 0.75, 0.97, 0.92],  # BoT-SORT (normalized)
    theta=categories, fill='toself', name='BoT-SORT'
))
fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])))
st.plotly_chart(fig, use_container_width=True)
```

#### 7.2.5. Tổ chức trên Dashboard

Thêm **tab "Tracker Comparison"** trong Streamlit:

```python
# app.py — thêm tab
tab1, tab2, tab3 = st.tabs(["🎥 Live Analysis", "📊 Tracker Comparison", "📋 Traffic Profile"])

with tab2:
    st.header("So sánh ByteTrack vs BoT-SORT")

    if st.button("▶️ Chạy Benchmark"):
        with st.spinner("Đang chạy ByteTrack..."):
            r1 = benchmark_tracker(video, model, "ByteTrack")
        with st.spinner("Đang chạy BoT-SORT..."):
            r2 = benchmark_tracker(video, model, "BoT-SORT")

        # Hiển thị bảng + chart
        ...
```

#### 7.2.6. Kết luận mẫu cho báo cáo

Sau khi chạy benchmark, tổng hợp kết luận dạng:

> **ByteTrack** cho tốc độ xử lý nhanh hơn (~30 FPS vs ~24 FPS), phù hợp ứng dụng realtime. Tuy nhiên, **BoT-SORT** cho kết quả tracking ổn định hơn nhờ cơ chế Camera Motion Compensation (CMC) — đặc biệt quan trọng với video UAV vì camera bay liên tục di chuyển. BoT-SORT có ít ID switch hơn (195 unique IDs vs 210) và counting accuracy cao hơn (98.9% vs 97.3%).
>
> **Khuyến nghị:** Dùng BoT-SORT cho bài toán phân tích mật độ cần độ chính xác cao; dùng ByteTrack khi ưu tiên tốc độ realtime.

### 7.3. Virtual Line vs ROI

- **Virtual Line**: Đơn giản nhất — vẽ 1 đường ngang, đếm xe đi qua
- **ROI (Region of Interest)**: Vẽ vùng đa giác, đếm xe trong vùng
- Cho bản demo: **dùng Virtual Line trước** (dễ code), có thể mở rộng ROI sau

### 7.4. Xử lý video trong Streamlit

```python
# Cách đơn giản: xử lý video → lưu output → hiển thị
# Không stream realtime (Streamlit không hỗ trợ tốt)
# Thay vào đó: xử lý từng frame → hiển thị bằng st.image() trong vòng lặp
stframe = st.empty()
for frame in frames:
    processed = pipeline.process_frame(frame)
    stframe.image(processed, channels="BGR")
```

### 7.5. Khi nào gọi LLM?

- **Không gọi mỗi frame** (tốn quota, chậm)
- Gọi LLM khi:
  - Người dùng nhấn nút "Phân tích"
  - Sau khi xử lý xong toàn bộ video
  - Hoặc mỗi N giây (configurable, mặc định 30s)

---

## 8. Class mapping (YOLO11 → 4 lớp)

Khi train YOLO11 trên VisDrone, cần map lại class:

| VisDrone Class ID | VisDrone Label | Project Label | Project ID |
|---|---|---|---|
| 4 | car | Car | 0 |
| 1 | pedestrian | *(bỏ qua)* | — |
| 5 | van | Car | 0 |
| 9 | motorcycle | Motorcycle | 1 |
| 6 | truck | Truck | 2 |
| 3 | bus | Bus | 3 |

> Mapping này đã được xử lý khi train model. File `best.pt` đã sẵn sàng với 4 class: Car, Motorcycle, Bus, Truck.

---

## 9. Tổng kết

| Tiêu chí | Giải pháp |
|---|---|
| **Đơn giản** | Dùng Ultralytics built-in tracking, Streamlit no-frontend |
| **Chạy được** | Pipeline hoàn chỉnh từ video → dashboard |
| **LLM tích hợp** | Groq API miễn phí, tốc độ cao |
| **Dễ mở rộng** | Module tách biệt, thêm ROI / model mới dễ dàng |
| **Demo-ready** | Upload video → xem kết quả + Traffic Control Profile |

---

*Kế hoạch này tập trung vào việc xây dựng bản demo chạy được, code rõ ràng, đơn giản, đủ để trình bày đồ án tốt nghiệp.*
