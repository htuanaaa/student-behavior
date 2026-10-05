import cv2
import logging
import os
import sys
import numpy as np
from tensorflow.keras.models import load_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from card_check import get_device, get_device_name

logger = logging.getLogger("uvicorn.error")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ObjectDetection:
    def __init__(self, model_path='../trainmodel/cnn_sleep_model.h5'):
        self.device = get_device()
        logger.info(f"Đang nạp mô hình object detection vào {get_device_name()}...")
        self.model = load_model(model_path)
        logger.info("Tải mô hình thành công...")
        self.target_size = (128, 128)

    def check_sleep(self, crop_frame) -> bool:
        try:
            img_rgb = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, self.target_size)
            input_tensor = np.expand_dims(img_resized, axis=0)
            prediction = self.model.predict(input_tensor, verbose=0)[0][0]

            if prediction > 0.5:
                return True
            else:
                return False
        except Exception as e:
            logging.error(f"Lỗi khi dự đoán: {e}")
            return False
