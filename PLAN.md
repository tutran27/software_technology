# 📋 KẾ HOẠCH TRIỂN KHAI DỰ ÁN (CHI TIẾT & HOÀN CHỈNH)

## Đề tài: Phát triển phần mềm có tính năng phát hiện, theo dõi và phân tích mật độ phương tiện từ dữ liệu video UAV bằng Deep Learning phục vụ giám sát và hỗ trợ điều khiển tín hiệu giao thông thích ứng

---

## 1. Tổng quan dự án & Mục tiêu

Dự án xây dựng một hệ thống hoàn chỉnh từ xử lý AI (Deep Learning, MOT, Phân tích mật độ, LLM) đến giao diện người dùng chuyên nghiệp (Streamlit UI/UX đa trang, trực quan hóa khoa học), đáp ứng đầy đủ yêu cầu của đề cương luận văn tốt nghiệp:

1. **Phát hiện phương tiện (Object Detection):** YOLO11 (`best.pt`) nhận diện các lớp phương tiện giao thông (Car, Motorcycle, Bus, Truck, Bicycle).
2. **Theo dõi đa đối tượng (Multi-Object Tracking - MOT):** Hỗ trợ chuyển đổi mượt mà và đối chứng giữa **ByteTrack** và **BoT-SORT** (tích hợp Camera Motion Compensation - CMC bù trừ chuyển động của drone).
3. **Giữ vùng ROI khi UAV chuyển động:** Bù chuyển động camera bằng thuật toán CMC / Optical Flow và cơ chế buffer tolerance.
4. **Đếm phương tiện thông minh:** Virtual Counting Line đa hướng và Polygon ROI.
5. **Mô hình hóa mật độ & Tắc nghẽn:** Đánh giá $OCR$ (Tỷ lệ chiếm dụng diện tích mặt đường), Vận tốc trung bình $\bar{v}$, Tỷ lệ xe dừng $R_{stop}$, Thời gian lưu trú $T_{wait}$, và Chỉ số tắc nghẽn tổng hợp $CI \in [0, 100]$.
6. **Động cơ cảnh báo đa cấp (Alert Engine):** Tự động phát hiện ùn tắc cục bộ, dừng xe hàng loạt (gridlock), lưu lượng bất thường.
7. **Tích hợp Groq LLM (Llama 3.1 70B):** Tự động tạo bản khuyến nghị điều khiển đèn tín hiệu giao thông thích ứng (Traffic Control Profile) chuẩn xác.
8. **Module Đối chứng & Đánh giá Tracking 2 phương diện:**
   - **Cách 1: Đánh giá Thực nghiệm (Không cần file label):** Dùng video UAV bất kỳ, tự đo FPS, Latency, Unique IDs, Phân mảnh ID, Sai số đếm xe so với Ground Truth đếm tay.
   - **Cách 2: Đánh giá Học thuật chuẩn MOT (Có file label):** Dùng dataset chuẩn (VisDrone-MOT/UAVDT), đo HOTA, MOTA, IDF1, ID Switches (IDSW).
9. **Giao diện Dashboard Streamlit Chuyên nghiệp & Tối ưu:** Thiết kế theo chuẩn Dashboard điều hành giao thông hiện đại (Operations Center UI), giao diện Dark Mode cao cấp, chia module rõ ràng, biểu đồ tương tác Plotly.

---

## 2. Công nghệ sử dụng

| Thành phần | Công nghệ | Phiên bản / Chi tiết | Lý do lựa chọn |
|---|---|---|---|
| **Ngôn ngữ** | Python | 3.10 – 3.12 | Chuẩn công nghiệp cho AI & Data |
| **Deep Learning** | Ultralytics YOLO11 | `>= 8.2.0` | SOTA Object Detection cho ảnh UAV nhỏ/dày đặc |
| **Tracking** | ByteTrack & BoT-SORT | YAML configs tích hợp | Dễ đối chứng, BoT-SORT có sẵn CMC bù rung UAV |
| **Thị giác máy tính** | OpenCV | `>= 4.8.0` | Xử lý frame, vẽ overlay, Optical Flow |
| **LLM Reasoning** | Groq Cloud API | `llama-3.1-70b-versatile` | Tốc độ suy luận >300 tokens/s, tư duy logic giao thông xuất sắc |
| **Backend & Web Server** | FastAPI + Uvicorn | `>= 0.110.0` | Xử lý bất đồng bộ, stream MJPEG 60 FPS, REST API siêu nhẹ |
| **Frontend / Web UI** | HTML5, CSS3, JavaScript | Modern Dark HUD | Native MJPEG video tag, Chart.js tương tác, không lag DOM |
| **Trực quan hóa** | Chart.js | `4.x` | Biểu đồ Canvas thời gian thực siêu nhẹ (Line, Donut) |

---

## 3. Cấu trúc Source Code & Thư mục Web Dashboard (FastAPI Architecture)

Cấu trúc mã nguồn được phân chia module hóa triệt để, tách biệt phần Core AI xử lý dữ liệu và phần Web Frontend hiển thị:

```
📁 software_technology/
│
├── 📁 models/                          # Chứa file trọng số mô hình
│   └── best.pt                         # Trọng số YOLO11 đã train
│
├── 📁 videos/                          # Video dữ liệu UAV đầu vào & mẫu
│   ├── DJI_20250516071323_0341_D.MP4
│   └── temp_uploads/                   # Thư mục lưu tạm video người dùng tải lên
│
├── 📁 core/                            # THƯ VIỆN LÕI XỬ LÝ AI
│   ├── __init__.py
│   ├── detector.py                     # Wrapper YOLO11 Object Detector
│   ├── tracker.py                      # Multi-Object Tracker (BoT-SORT + CMC / ByteTrack)
│   ├── counter.py                      # Bộ đếm xe (Virtual Line & Polygon ROI Crossing)
│   ├── analyzer.py                     # Tính mật độ, vận tốc, dwell time, Congestion Index (CI)
│   └── alert.py                        # Động cơ cảnh báo ngưỡng giao thông
│
├── 📁 llm/                             # MODULE TÍCH HỢP AI TƯ DUY
│   ├── __init__.py
│   └── traffic_profile.py              # Groq API client sinh khuyến nghị đèn giao thông
│
├── 📁 web/                             # 🎨 THƯ MỤC FRONTEND & WEB DASHBOARD (FASTAPI)
│   ├── 📁 static/                      # Static assets
│   │   ├── 📁 css/
│   │   │   └── styles.css              # Dark Glassmorphism CCTV Dashboard Theme
│   │   └── 📁 js/
│   │       └── app.js                  # Polling metrics, Chart.js, REST API client
│   └── 📁 templates/
│       └── index.html                  # Giao diện chính 3 Tabs (Live 70/30, LLM Profile, Analytics)
│
├── 📁 utils/                           # TIỆN ÍCH HỆ THỐNG
│   ├── __init__.py
│   ├── config.py                       # Cấu hình màu sắc, nhãn class, video mẫu
│   └── drawing.py                      # Vẽ BBox, trail, HUD, vạch ảo, ROI polygon
│
├── server.py                           # 🚀 FastAPI Backend Server & MJPEG Stream Engine (Port 8501)
├── pipeline.py                         # Pipeline liên kết tuần tự Core AI
├── test_detect.py                      # Script test nhanh YOLO11 & đo FPS
├── requirements.txt                    # Danh sách thư viện cần thiết
└── PLAN.md                             # Tài liệu kế hoạch chi tiết này
```

---

## 4. Quản lý Nguồn Video & Khả Năng Tải Lên (Video Upload & Management)

Giao diện hỗ trợ quản lý video đa nguồn linh hoạt và ổn định:
1. **Video Mẫu Tích Hợp Sẵn:** Danh mục video UAV quay tại các nút giao thông đô thị phức tạp (DJI 4K UAV) với thông tin độ phân giải, FPS và thời lượng rõ ràng.
2. **Tải Lên Video Tùy Chỉnh (Custom Video Upload):**
   - Hỗ trợ đầy đủ các định dạng: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`.
   - Cơ chế lưu trữ an toàn trong `videos/temp_uploads/`, tự động phân tích metadata (Resolution, FPS, Total Frames).
   - Tối ưu hóa chu trình phát luồng (Stream Engine) với cơ chế Play / Pause / Resume / Stop / Reset và tự động lặp lại (Loop).

---

## 5. Thiết kế Giao diện Người dùng (Streamlit Frontend UI/UX)

Giao diện được thiết kế theo phong cách **High-Tech CCTV Operations Center (Dark Glassmorphism)**:

### 5.1. Bố cục tổng thể (App Layout)

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│  🚁 UAV TRAFFIC INTELLIGENT OPERATIONS CENTER                       [● LIVE SYSTEM] [YOLO11]   │
│  [Source: DJI_4K_UAV.mp4]  [Active Tracker: BoT-SORT (CMC Enabled)]  [Resolution: 1920x1080]    │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 🎮 TOPBAR: [▶ CHẠY / TIẾP TỤC]  [⏸ TẠM DỪNG]  [⏹ DỪNG]  [🔄 ĐẶT LẠI]  [Trạng thái: Đang chạy] │
├─────────────────────────┬──────────────────────────────────────────────────────────────────────┤
│ ⚙️ SIDEBAR ĐIỀU KHIỂN    │ 📑 KHU VỰC NỘI DUNG CHÍNH (3 TABS CHUYÊN NGHIỆP)                    │
│                         ├──────────────────────────────────────────────────────────────────────┤
│ 1. Nguồn Dữ Liệu        │ 🎥 TAB 1: GIÁM SÁT REALTIME & PHÂN TÍCH MẬT ĐỘ (70% - 30%)           │
│    • Video mẫu          │ ┌──────────────────────────────────┬───────────────────────────────┐ │
│    • Upload video file  │ │ Khung Video UAV HUD              │ 🔢 Lưới 2x2 Đếm Phương Tiện   │ │
│                         │ │ (BBox, Track ID, Trail, Line Y)  │ • Car, Motor, Bus, Truck      │ │
│ 2. Cấu hình AI          │ │                                  ├───────────────────────────────┤ │
│    • BoT-SORT / ByteTrack│ │                                  │ 📊 Mật độ OCR & Chỉ số CI     │ │
│    • Confidence Slider  │ │ ──────────────────────────────── │ • Gauge: Thông thoáng -> Ùn tắc│
│    • NMS IoU Slider     │ │ 📈 Biểu đồ Mini Diễn Biến Realtime│ 🚨 Cảnh Báo & Pha Đèn Thích Ứng│
│    • Vạch đếm Y Slider  │ └──────────────────────────────────┴───────────────────────────────┘ │
│                         ├──────────────────────────────────────────────────────────────────────┤
│ 3. Hiệu năng & Tốc độ   │ 🚦 TAB 2: ĐIỀU KHIỂN ĐÈN TÍN HIỆU THÍCH ỨNG (GROQ LLM)               │
│    • Frame Skip (1/2/3) │ ┌──────────────────────────────────┬───────────────────────────────┐ │
│    • Delay mô phỏng (ms)│ │ 📥 Nhập/Đồng bộ thông số dòng xe │ ⏱️ Sơ đồ chu kỳ đèn trực quan │ │
│    • Lặp lại video      │ │ 🚀 [Sinh khuyến nghị Llama 3.1]  │ 📄 Báo cáo Traffic Profile    │ │
│                         │ └──────────────────────────────────┴───────────────────────────────┘ │
│                         ├──────────────────────────────────────────────────────────────────────┤
│                         │ 📊 TAB 3: BÁO CÁO & THỐNG KÊ TỔNG HỢP                                │
│                         │ ┌──────────────────────────────────┬───────────────────────────────┐ │
│                         │ │ 🚗 Biểu đồ phân bổ loại xe Donut │ 📈 Chuỗi thời gian mật độ     │ │
│                         │ ├──────────────────────────────────┴───────────────────────────────┤ │
│                         │ │ 📋 Bảng nhật ký dữ liệu chi tiết & [📥 Xuất File CSV]            │ │
│                         │ └──────────────────────────────────────────────────────────────────┘ │
└─────────────────────────┴──────────────────────────────────────────────────────────────────────┘
```

### 5.2. Các tính năng UI nổi bật
1. **Live Stream Player HUD:** Hiển thị sắc nét khung hình BBox kèm ID, vệt quỹ đạo chuyển động mượt mà, vạch đếm ảo tùy chỉnh vị trí thời gian thực.
2. **Live KPI Metric Cards:** Lưới 2x2 trực quan phân biệt 4 loại phương tiện, bảng phân tích OCR %, vận tốc và điểm tắc nghẽn CI.
3. **Cảnh Báo & Điều Khiển Tức Thì:** Động cơ cảnh báo đa cấp (Ùn tắc, Gridlock, Chiếm dụng cao) kèm trạng thái đèn tín hiệu thích ứng.
4. **Adaptive Signal Control Center:** Tích hợp Groq Llama 3.1 70B tự động sinh bản Traffic Control Profile kèm sơ đồ thời gian pha đèn cho hướng chính và phụ.
5. **Analytics & Data Export:** Biểu đồ Donut tỷ lệ xe, biểu đồ xu hướng mật độ và nút tải xuống toàn bộ dữ liệu lịch sử đo đạc dạng file CSV.

---

## 6. Lộ trình Triển khai Hoàn Tất

```
Giai đoạn 1: Khởi tạo Cấu trúc UI & Tiện ích Frontend (utils/config, ui/styles, ui/components) [HOÀN THÀNH]
     │
     ▼
Giai đoạn 2: Hoàn thiện Thư viện Lõi AI (core/detector, tracker, counter, analyzer, alert) [HOÀN THÀNH]
     │
     ▼
Giai đoạn 3: Tích hợp Groq LLM Client & Module Phân tích (llm/traffic_profile) [HOÀN THÀNH]
     │
     ▼
Giai đoạn 4: Xây dựng Giao diện Đa Tab Streamlit (ui/pages/tab_*, app.py) [HOÀN THÀNH]
     │
     ▼
Giai đoạn 5: Tối ưu Trải nghiệm Video Upload, Điều khiển Playback & Xuất Dữ liệu [HOÀN THÀNH]
```

---
*Tài liệu kế hoạch đã được đồng bộ hóa hoàn chỉnh với giao diện Streamlit hiện đại và các tính năng vận hành thời gian thực.*

