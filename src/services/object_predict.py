import cv2
import logging
import os
import sys
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from card_check import get_device, get_device_name

logger = logging.getLogger("uvicorn.error")

class ObjectDetection:
    def __init__(self, model_path='../trainmodel/model_yolo.pt'):
        self.device = get_device()
        logger.info(f"Đang nạp mô hình object detection vào {get_device_name()}...")
        self.model = YOLO(model_path)
        logger.info("Tải mô hình thành công...")

    def check_class(self, crop_frame, class_id) -> bool:
        results = self.model.predict(
            source=crop_frame, 
            conf=0.65,
            imgsz=416,
            max_det=1,
            classes=[0, 2],
            device=self.device,
            verbose=False
        )
        detected_classes = results[0].boxes.cls.cpu().numpy()
        
        return class_id in detected_classes