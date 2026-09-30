import cv2
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
        self.ALARM = 90    # 3 giây / 30 fps
        self.trackers = {}

    def check_hand(self, person) -> bool:
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
        eye_conf_threshold = 0.3
        tracker = self.trackers[track_id]
        
        if l_eye[2] < eye_conf_threshold and r_eye[2] < eye_conf_threshold:
            tracker['hid_frame'] += 1
            tracker['tilt_frame'] = 0
        elif l_eye[2] >= eye_conf_threshold and r_eye[2] >= eye_conf_threshold:
            tracker['hid_frame'] = 0
            dx = r_eye[0] - l_eye[0]
            dy = r_eye[1] - l_eye[1]

            if dx != 0:
                angle = math.degrees(math.atan(abs(dy) / abs(dx)))
                if abs(angle) > 10:
                    tracker['tilt_frame'] += 1
                else:
                    tracker['tilt_frame'] = 0
        else:
            tracker['hid_frame'] = 0
            tracker['tilt_frame'] = 0

        if tracker['hid_frame'] == self.ALARM or tracker['tilt_frame'] == self.ALARM:
            tracker['hid_frame'] = 0
            tracker['tilt_frame'] = 0
            return True
        return False

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