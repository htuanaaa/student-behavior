# Hệ Thống Giám Sát Hành Vi Lớp Học (Classroom Behavior Monitoring)

Hệ thống ứng dụng thị giác máy tính và học sâu (Computer Vision & Deep Learning) để tự động phát hiện và cảnh báo các hành vi của học sinh trong lớp học trong thời gian thực, bao gồm:
- **Giơ tay phát biểu** (Hand Raising Detection).
- **Ngủ gật / Gục đầu trong lớp** (Sleeping & Drowsiness Detection qua cơ chế xác thực 2 bước: Pose Keypoints + Object Detection).

---

## 📌 Tính Năng Nổi Bật

1. **Nhận diện thời gian thực qua nhiều nguồn:**
   - Hỗ trợ truyền hình ảnh trực tiếp từ **Webcam** thông qua giao thức **WebSocket** (độ trễ thấp).
   - Hỗ trợ luồng **Live Camera** (RTSP / HTTP stream).
   - Hỗ trợ tải lên video có sẵn (**Upload Video**) để phân tích và trích xuất video kết quả.

2. **Cơ chế Two-Stage Verification (Chống báo động giả):**
   - **Giai đoạn 1 (Pose Estimation):** Sử dụng YOLOv8-Pose bắt các khớp xương (tay, khuỷu tay, mắt) để phát hiện trạng thái giơ tay hoặc nghi vấn ngủ gật (nhắm mắt / nghiêng đầu tích lũy quá thời gian cho phép).
   - **Giai đoạn 2 (Object Detection):** Khi nghi vấn có người ngủ, hệ thống tự động cắt (crop) vùng đối tượng và đưa qua mô hình YOLO Detection để xác nhận lại, đảm bảo độ chính xác cao.

---

## 📂 Cấu Trúc Dự Án

```text
├── dataset/             # Cấu hình dataset (data.yaml)
├── yolomodel/           # Chứa weights YOLO Pose (yolov8n-pose.pt)
├── trainmodel/          # Chứa mô hình phát hiện hành vi tùy biến (model_v1.pt)
├── train.py             # Script huấn luyện mô hình YOLO
├── requirements.txt     # Danh sách thư viện phụ thuộc
├── .gitignore           # Cấu hình bỏ qua file rác và file dung lượng lớn
│
└── src/                 # Thư mục ứng dụng chính
    ├── main.py          # FastAPI Server (WebSocket, Video Stream, API)
    ├── templates/       # Giao diện Web Client (HTML/CSS/JS)
    └── services/        # Các module xử lý logic AI
        ├── predict.py         # Điều phối luồng xử lý khung hình
        ├── pose_predict.py    # Xử lý dáng điệu & toạ độ khớp xương
        └── object_predict.py  # Xác thực đối tượng trên vùng crop
```

---

## 🛠️ Yêu Cầu Cài Đặt

### 1. Khởi tạo môi trường ảo (Khuyên dùng)
```bash
python3 -m venv .venv
source .venv/bin/activate   # Trên Linux / macOS
# .venv\Scripts\activate    # Trên Windows
```

### 2. Cài đặt PyTorch
Nếu máy có card đồ hoạ rời (NVIDIA GPU), hãy cài đặt bản PyTorch hỗ trợ CUDA để đạt hiệu năng tốt nhất:
```bash
# Cài đặt PyTorch với CUDA 12.1 (hoặc thay bằng bản tương thích với GPU của bạn)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
*(Nếu chỉ dùng CPU, bạn có thể bỏ qua bước trên, pip sẽ tự động cài bản CPU khi chạy bước 3).*

### 3. Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

---

## 🚀 Hướng Dẫn Sử Dụng

### Khởi động Web Server
Di chuyển vào thư mục `src/` và chạy FastAPI bằng `uvicorn`:
```bash
cd src
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Sau khi server khởi động thành công, mở trình duyệt web và truy cập:
```
http://localhost:8000
```

### Triển khai bằng Docker

Chỉ cần cài đặt Docker trên máy đích, bạn có thể đóng gói và chạy ứng dụng mà không cần cài đặt môi trường Python:

```bash
# 1. Build image từ Dockerfile (chỉ cần chạy lần đầu hoặc khi sửa code)
docker build -t classroom-monitor .

# 2. Khởi chạy container:
# Đối với máy chạy CPU:
docker run -d --name classroom_app -p 8000:8000 -v $(pwd)/src/output:/app/src/output classroom-monitor

# (Tùy chọn) Đối với máy có GPU NVIDIA (yêu cầu cài NVIDIA Container Toolkit):
# docker run -d --name classroom_app --gpus all -p 8000:8000 -v $(pwd)/src/output:/app/src/output classroom-monitor

# 3. Xem log hoạt động:
docker logs -f classroom_app

# 4. Dừng và xóa container khi không sử dụng:
docker stop classroom_app && docker rm classroom_app
```

Sau đó truy cập trình duyệt tại: `http://localhost:8000`

---

## 📄 Bản Quyền & Giấy Phép
Dự án được xây dựng phục vụ mục đích nghiên cứu và học tập.
