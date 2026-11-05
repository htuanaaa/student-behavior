import os
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

DATA_DIR = "dataset"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20

os.makedirs("trainmodel", exist_ok=True)

print("Đang nạp tập dữ liệu huấn luyện (Train)...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print("Đang nạp tập dữ liệu kiểm định (Validation)...")
val_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "valid"),
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print("\nĐang nạp tập dữ liệu kiểm tra độc lập (Test)...")
test_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names
print(f"Các nhãn đã nhận diện: {class_names}") 

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),

    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall')])

model.summary()

print("\nBắt đầu quá trình huấn luyện mô hình...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

model_path = "trainmodel/cnn_sleep_model.h5"
model.save(model_path)
print(f"\nĐã lưu mô hình thành công tại: {model_path}")

print("\nĐang chấm điểm mô hình trên tập Test...")
test_loss, test_acc, test_precision, test_recall = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Precision: {test_precision * 100:.2f}%")
print(f"Test Recall:    {test_recall * 100:.2f}%")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Accuracy', linewidth=2)
plt.plot(epochs_range, val_acc, label='Validation Accuracy', linewidth=2)
plt.legend(loc='lower right')
plt.title('Accuracy')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss', linewidth=2)
plt.plot(epochs_range, val_loss, label='Validation Loss', linewidth=2)
plt.legend(loc='upper right')
plt.title('Loss')
plt.grid(True)

chart_path = "training_history.png"
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
print(f"\ndone: {chart_path}")

print("\nĐang xuất Ma trận nhầm lẫn (Confusion Matrix)...")
y_true = np.concatenate([y for x, y in test_dataset], axis=0)
y_pred_probs = model.predict(test_dataset)
y_pred = (y_pred_probs > 0.5).astype("int32")

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Ma Trận Nhầm Lẫn (Confusion Matrix)')
plt.ylabel('Thực tế (True Label)')
plt.xlabel('Dự đoán (Predicted Label)')

cm_path = "confusion_matrix.png"
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"\ndone: {cm_path}")
