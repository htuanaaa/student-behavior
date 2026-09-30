from ultralytics import YOLO
from card_check import get_device

model = YOLO("trainmodel/model_v1.pt")

results = model.train(
    data='dataset/data.yaml',
    epochs=50,
    imgsz=480,
    batch=4,
    device=get_device(),
    workers=2,
    name='trainnn',
    exist_ok=True,
    save=True,
    plots=True,
    patience=15
)
metrics = results.box
print(f"Mean Precision (P): {metrics.mp:.4f}")
print(f"Mean Recall (R): {metrics.mr:.4f}")
print(f"mAP@50: {metrics.map50:.4f}")
print(f"mAP@50-95: {metrics.map:.4f}")
print("doneeeee!!!!")
