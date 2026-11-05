import cv2
import numpy as np
import logging
import math
import os
import sys
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from card_check import get_device, get_device_name

logger = logging.getLogger("uvicorn.error")

class PoseEstimation:
    def __init__(self, model_path='../yolomodel/yolov8n-pose.pt'):
        self.device = get_device()
        logger.info(f"Đang nạp mô hình pose estimation vào {get_device_name()}...")
        self.model = YOLO(model_path)
        logger.info("Tải mô hình thành công...")
        self.ALARM = 50     # 1,7 giây ở 30 fps
        self.trackers = {}

    def check_hand(self, person, min_conf=0.5) -> bool:
        if len(person[0]) >= 3:
            left_clear = min(person[7][2], person[9][2]) >= min_conf
            right_clear = min(person[8][2], person[10][2]) >= min_conf
            if not (left_clear or right_clear):
                return False

        # left hand check
        dx_l = person[9][0] - person[7][0]
        dy_l = person[9][1] - person[7][1]
        angle_l = math.degrees(math.atan2(dy_l, dx_l))
        if -110 <= angle_l <= -70:
            if person[9][1] < person[5][1]:
                return True

        # right hand check
        dx_r = person[10][0] - person[8][0]
        dy_r = person[10][1] - person[8][1]
        angle_r = math.degrees(math.atan2(dy_r, dx_r))
        if -110 <= angle_r <= -70:
            if person[10][1] < person[6][1]:
                return True
        return False

    def check_sleep(self, track_id, person_data) -> bool:
        if track_id not in self.trackers:
            self.trackers[track_id] = {'hid_frame': 0, 'tilt_frame': 0}

        l_eye = person_data[1]
        r_eye = person_data[2]
        nose = person_data[0]
        l_ear = person_data[3]
        r_ear = person_data[4]
        eye_conf_threshold = 0.3
        tracker = self.trackers[track_id]
        face_visible = (nose[2] >= 0.5) and (l_ear[2] >= 0.4 or r_ear[2] >= 0.4)
        
        if l_eye[2] < eye_conf_threshold and r_eye[2] < eye_conf_threshold and face_visible:
            tracker['hid_frame'] += 1
            tracker['tilt_frame'] = 0
        elif l_eye[2] >= eye_conf_threshold and r_eye[2] >= eye_conf_threshold:
            tracker['hid_frame'] = 0
            dx = r_eye[0] - l_eye[0]
            dy = r_eye[1] - l_eye[1]

            shoulder_w = math.hypot(person_data[6][0] - person_data[5][0],
                                    person_data[6][1] - person_data[5][1])
            turned_away = shoulder_w > 1e-3 and abs(dx) < 0.35 * shoulder_w
            if turned_away:
                tracker['tilt_frame'] = max(0, tracker['tilt_frame'] - 1)

            elif dx != 0:
                angle = math.degrees(math.atan(abs(dy) / abs(dx)))
                if abs(angle) > 20:
                    tracker['tilt_frame'] += 1
                else:
                    tracker['tilt_frame'] = 0
        else:
            tracker['hid_frame'] = max(0, tracker['hid_frame'] - 1)
            tracker['tilt_frame'] = max(0,  tracker['tilt_frame'] - 1)

        if tracker['hid_frame'] >= self.ALARM or tracker['tilt_frame'] >= self.ALARM:
            tracker['hid_frame'] = 0
            tracker['tilt_frame'] = 0
            return True
        return False

    def get_face_crop(self, frame, person_data, margin=1.8, min_conf=0.35, out_size=(128, 128)):
        if frame is None or frame.size == 0:
            return None, False

        kp = np.asarray(person_data, dtype=np.float32)[0:5]
        good = kp[:, 2] >= min_conf
        if int(good.sum()) < 2:
            return None, False

        xs, ys = kp[good, 0], kp[good, 1]
        cx = float((xs.min() + xs.max()) / 2.0)
        cy = float((ys.min() + ys.max()) / 2.0)
        span = float(max(xs.max() - xs.min(), ys.max() - ys.min()))

        side = max(span * margin, 24.0)
        h, w = frame.shape[:2]
        x1 = int(round(cx - side / 2.0))
        y1 = int(round(cy - side / 2.0))
        x2 = int(round(cx + side / 2.0))
        y2 = int(round(cy + side / 2.0))

        x1 = max(0, min(x1, w - 2))
        y1 = max(0, min(y1, h - 2))
        x2 = max(x1 + 2, min(x2, w))
        y2 = max(y1 + 2, min(y2, h))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, False

        return cv2.resize(crop, out_size, interpolation=cv2.INTER_AREA), True

    def detect(self, frame):
        results = self.model.track(
            source=frame, 
            conf=0.5,
            imgsz=416,
            persist=True,
            device=self.device,
            verbose=False
        )
        annotated_frame = results[0].plot(boxes=False)
        detected_persons = []

        if len(results[0].boxes) > 0 and results[0].keypoints is not None and results[0].boxes.id is not None:
            box = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            person = results[0].keypoints.xy.cpu().numpy()
            person_data = results[0].keypoints.data.cpu().numpy()

            # lx, ly = int(l_eye[0]), int(l_eye[1])
            # rx, ry = int(r_eye[0]), int(r_eye[1])

            # cv2.circle(annotated_frame, (lx, ly), 3, (0, 0, 255), -1)
            # cv2.circle(annotated_frame, (rx, ry), 3, (0, 0, 255), -1)

            # cv2.putText(annotated_frame, f"L:{l_eye[2]}%", (lx + 5, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            # cv2.putText(annotated_frame, f"R:{r_eye[2]}%", (rx + 5, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            for i in range(len(results[0].boxes)):
                detected_persons.append({
                    'id': track_ids[i],
                    'box': box[i],
                    'person': person[i],
                    'person_data': person_data[i]
                })
            return annotated_frame, detected_persons
        return annotated_frame, []