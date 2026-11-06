# Hệ Thống Giám Sát Hành Vi Lớp Học (Classroom Behavior Monitoring)

Hệ thống ứng dụng thị giác máy tính và học sâu (Computer Vision & Deep Learning) để tự động phát hiện và cảnh báo các hành vi của học sinh trong lớp học trong thời gian thực, bao gồm:
- **Giơ tay phát biểu** (Hand Raising Detection).
- **Ngủ gật / Gục đầu trong lớp** (Phát hiện qua cơ chế xác thực 2 bước: Pose Keypoints + CNN Classification).

---

## Tính Năng Nổi Bật

1. **Nhận diện thời gian thực:**
   - Hỗ trợ truyền hình ảnh trực tiếp từ **Webcam** thông qua giao thức **WebSocket**.
   - Hỗ trợ luồng **Live Camera** (RTSP / HTTP stream).
   - Hỗ trợ tải lên video có sẵn (**Upload Video**) để phân tích và trích xuất video kết quả.

2. **Cơ chế chống báo động giả:**
   - **Giai đoạn 1 (Pose Estimation):** Dùng YOLOv8-Pose ở độ phân giải suy luận 1280 để bắt khớp xương đầu, vai, tay. Tầng này phát hiện giơ tay, đồng thời đánh dấu nghi vấn ngủ gật khi mắt nhắm hoặc đầu nghiêng tích luỹ quá 50 khung liên tiếp.
   - **Giai đoạn 2 (CNN Classification):** Khi có nghi vấn, hệ thống cắt vùng đầu theo khung xương rồi đưa qua mạng CNN phân loại hai lớp `sleep` và `awake`. Chỉ khung nào CNN xác nhận mới phát cảnh báo.

---

## Cấu Trúc Dự Án

```text
├── dataset/                 # Bộ dữ liệu đã chia tập, mỗi tập có hai thư mục awake/ và sleep/
│   └── manifest.tsv         # Nhãn, góc quay, nguồn, track và số khung của từng ảnh
├── dataset_raw/             # Ảnh cắt thô từ video, chưa chia tập (kèm manifest.tsv)
├── trainmodel/
│   ├── cnn_sleep_model.h5   # Model CNN phân loại ngủ/thức, dùng trong luồng chính
│   └── model_yolo.pt        # Model YOLO detection cũ, không còn dùng trong luồng chính
├── yolomodel/               # Weights YOLO cho pose và detection
├── crop_dataset.py          # Cắt ảnh từ video, sinh dataset_raw
├── split_dataset.py         # Lọc ảnh trùng, chia tập, sinh dataset
├── train.py                 # Huấn luyện CNN, sinh model và hai biểu đồ
├── card_check.py            # Chọn thiết bị chạy (CPU/CUDA)
├── tests/                   # Kiểm thử logic tầng pose
├── requirements.txt         # Danh sách thư viện phụ thuộc
├── Dockerfile               # Đóng gói ứng dụng web
├── .gitignore               # Cấu hình bỏ qua file rác và file dung lượng lớn
│
└── src/                     # Thư mục ứng dụng chính
    ├── main.py              # FastAPI Server (WebSocket, Video Stream, API)
    ├── templates/           # Giao diện Web Client (HTML/CSS/JS)
    └── services/            # Các module xử lý logic AI
        ├── predict.py         # Điều phối luồng xử lý khung hình
        ├── pose_predict.py    # Tầng 1: nhận diện tư thế, toạ độ khớp xương, cắt vùng đầu
        ├── cnn_predict.py     # Tầng 2: CNN xác thực trạng thái ngủ
        └── object_predict.py  # Tầng 2 phiên bản YOLO cũ, giữ lại làm option phụ
```

---

## Yêu Cầu Cài Đặt

Môi trường: Python 3.10 đến 3.11, TensorFlow 2.20 trở lên, ultralytics 8.2 trở lên.

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
*(Nếu chỉ dùng CPU, có thể bỏ qua bước trên, pip sẽ tự động cài bản CPU khi chạy bước 3).*

### 3. Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

---

## Hướng Dẫn Sử Dụng

### Huấn luyện lại bộ dữ liệu và mô hình

Ba script chạy nối tiếp tại thư mục gốc dự án. Không cần truyền tham số, mọi đường dẫn đã đặt mặc định và có thể ghi đè bằng biến môi trường `PROJECT_DIR`, `DATA_DIR`, `OUT_MODEL`.

```bash
python crop_dataset.py     # video nguồn -> dataset_raw
python split_dataset.py    # dataset_raw  -> dataset
python train.py            # dataset      -> trainmodel/cnn_sleep_model.h5
```

Bước 1 cần các video nguồn đặt sẵn ở thư mục gốc dự án, danh sách khai báo trong biến `SOURCES`. Bước 2 bắt buộc phải chạy sau bước 1, vì bước 1 chỉ cắt ảnh còn bước 2 mới lọc trùng và chia tập. Bước 3 sinh thêm hai file metric `training_history.png` và `confusion_matrix.png`.

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

## Lưu Ý Khi Triển Khai

- **Camera đặt xa thì tầng pose gần như mất người:** trên clip thử 1280x720 quay từ cuối lớp, tầng pose chỉ phát hiện trung bình 0,6 người mỗi khung, đúng 5 người trên 8 khung lấy mẫu. Vùng đầu của những người phát hiện được chỉ còn khoảng 38 pixel cạnh, phóng lên 128 thì gần như không còn chi tiết. Khi triển khai nên đặt camera gần hơn hoặc dùng cảm biến độ phân giải cao hơn.
- **Cảnh báo có độ trễ:** tầng 1 chỉ báo khi mắt nhắm hoặc đầu nghiêng liên tục 50 khung đã xử lý, tương đương khoảng 4 đến 5 giây tuỳ bước lấy khung.
- **Tốc độ xử lý:** khoảng 12 khung/giây trên CPU, tương đương 0,083 giây mỗi khung. Với nguồn 30 khung/giây nên xử lý cách 3 khung một lần.
