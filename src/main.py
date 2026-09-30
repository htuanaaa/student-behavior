from fastapi import FastAPI, Request, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from services.predict import Predict
import base64
import imageio
from datetime import datetime
import numpy as np
import cv2
import os
import shutil

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = Predict()

os.makedirs("temp_videos", exist_ok=True)
os.makedirs("output", exist_ok=True)

latest_stats = {"total_students": 0, "alerts": []}

def update_global_stats(ui_data):
    global latest_stats
    latest_stats["total_students"] = ui_data.get("total_students", 0)
    new_alerts = ui_data.get("alerts", [])
    if new_alerts:
        for alert in new_alerts:
            latest_stats["alerts"].insert(0, alert)
        latest_stats["alerts"] = latest_stats["alerts"][:10]

def frame_generator(source):
    cap = cv2.VideoCapture(source)
    now = datetime.now()
    source_normalize = now.strftime("%Y_%m_%d_%H_%M_%S")
    with imageio.get_writer(f"output/{source_normalize}.mp4", fps=30) as writer:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            annotated_frame, ui_data = model.process_frame(frame)
            update_global_stats(ui_data)
            writer.append_data(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html"
    )

@app.websocket("/ws/client_webcam")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            if "," not in data:
                continue
            encoded_data = data.split(",")[1]
            if not encoded_data:
                continue
            img_data = base64.b64decode(encoded_data)
            nparr = np.frombuffer(img_data, np.uint8)
            if nparr.size == 0:
                continue
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            annotated_frame, ui_data = model.process_frame(frame)
            update_global_stats(ui_data)
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            processed_base64 = base64.b64encode(buffer).decode("utf-8")
            await websocket.send_text(f"data:image/jpeg;base64,{processed_base64}")
    except WebSocketDisconnect:
        print("Client đã ngắt kết nối Camera")
    except Exception as e:
        print(f"Lỗi luồng WebSocket: {e}")

@app.get("/live_stream_feed")
def live_stream_feed(source: str = "0"):
    if source.isdigit():
        camera_source = int(source)
    else:
        if not source.startswith("http://") and not source.startswith("https://") and not source.startswith("rtsp://"):
            camera_source = f"http://{source}"
        else:
            camera_source = source
    return StreamingResponse(frame_generator(camera_source), media_type="multipart/x-mixed-replace; boundary=frame")
@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    file_path = f"temp_videos/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"stream_url": f"/video_feed/{file.filename}"}

@app.get("/video_feed/{filename}")
def video_feed(filename: str):
    file_path = f"temp_videos/{filename}"
    return StreamingResponse(frame_generator(file_path), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/stats")
def get_stats():
    return latest_stats
