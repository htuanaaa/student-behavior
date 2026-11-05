import cv2
import time
import base64
import os
from datetime import datetime
import logging
from services.pose_predict import PoseEstimation
from services.cnn_predict import ObjectDetection

logger = logging.getLogger("uvicorn.error")
os.makedirs("debug_crops", exist_ok=True)

class Predict:
    def __init__(self):
        self.pose_estimation = PoseEstimation()
        self.object_detection = ObjectDetection()

    def process_frame(self, frame):
        annotated_frame, detected_persons = self.pose_estimation.detect(frame)

        current_alerts = []

        if not detected_persons:
            return annotated_frame, {"alerts": current_alerts}

        h, w, _ = frame.shape

        for p in detected_persons:
            track_id = p['id']
            box = p['box']
            person = p['person']
            person_data = p['person_data']

            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if self.pose_estimation.check_hand(person_data):
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"ID {track_id} DANG GIO TAY", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # is_sleep_suspect = self.pose_estimation.check_sleep(track_id, person_data)
            # tracker = self.pose_estimation.trackers.get(track_id, {'hid_frame': 0, 'tilt_frame': 0})
            # hid_val = tracker.get('hid_frame', 0)
            # tilt_val = tracker.get('tilt_frame', 0)
            # cv2.putText(annotated_frame, f"hid:{hid_val} | tilt:{tilt_val}", (x1, max(0, y1 + 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if self.pose_estimation.check_sleep(track_id, person_data):
                logger.warning("Nghi vấn có người ngủ!")
                logger.warning("Đang đẩy vào model object detection...")
                crop_data, face_oke = self.pose_estimation.get_face_crop(frame, person_data)
                if not face_oke:
                    continue

                crop_filename = f"debug_crops/crop_{int(time.time() * 1000)}.jpg"
                cv2.imwrite(crop_filename, crop_data)

                if self.object_detection.check_sleep(crop_data):
                    logger.warning("Model xác nhận: CÓ NGƯỜI NGỦ!")
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(annotated_frame, f"CANH BAO ID {track_id}: CO NGUOI NGU!", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

                    _, buffer = cv2.imencode('.jpg', crop_data)
                    b64_img = base64.b64encode(buffer).decode('utf-8')
                    current_alerts.append({
                        "id": int(track_id),
                        "behavior": "Ngủ gục",
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "image": b64_img,
                        "type": "bad"
                    })
                else:
                    logger.warning("Model xác nhận: KHÔNG CÓ NGƯỜI NGỦ!")
        ui_data = {
            "total_students": len(detected_persons),
            "alerts": current_alerts
        }
        return annotated_frame, ui_data
