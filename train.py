import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, models

DATA_DIR = os.environ.get("DATA_DIR", "dataset")
OUT_MODEL = os.environ.get("OUT_MODEL", "trainmodel/cnn_sleep_model.h5")
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30

os.makedirs("trainmodel", exist_ok=True)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"), seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
val_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "valid"), seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
test_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"), image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_dataset.class_names
print("các nhãn:", class_names, flush=True)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

counts = {i: len(os.listdir(os.path.join(DATA_DIR, "train", c))) for i, c in enumerate(class_names)}
total = sum(counts.values())
class_weight = {i: total / (len(counts) * n) for i, n in counts.items()}
print("số ảnh mỗi lớp:", counts, flush=True)
print("trọng số:", {class_names[i]: round(w, 3) for i, w in class_weight.items()}, flush=True)

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

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(OUT_MODEL, monitor='val_loss', save_best_only=True, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
]

print("bắt đầu huấn luyện...", flush=True)
history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS,
                    class_weight=class_weight, callbacks=callbacks)

test_loss, test_acc, test_precision, test_recall = model.evaluate(test_dataset)
print("Test Accuracy: %.2f%%" % (test_acc * 100))
print("Test Loss: %.4f" % test_loss)
print("Test Precision: %.2f%%" % (test_precision * 100))
print("Test Recall: %.2f%%" % (test_recall * 100))

y_true = np.concatenate([y for x, y in test_dataset], axis=0)
y_pred = (model.predict(test_dataset, verbose=0).ravel() > 0.5).astype("int32")
cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:")
print(cm)

rows = [r for r in csv.DictReader(open(os.path.join(DATA_DIR, "manifest.tsv")), delimiter="\t")
        if r["split"] == "test"]
rows.sort(key=lambda r: r["file"])
acc_by_angle = {}
for r, truth, pred in zip(rows, y_true, y_pred):
    ok = int(truth) == int(pred)
    stat = acc_by_angle.setdefault(r["angle"], [0, 0])
    stat[0] += int(ok)
    stat[1] += 1
print("độ chính xác theo góc quay ở tập test:")
for angle, (ok, n) in sorted(acc_by_angle.items()):
    print("  %-16s %3d/%3d = %5.1f%%" % (angle, ok, n, 100.0 * ok / n))

n_epochs = len(history.history['accuracy'])
epochs_range = range(n_epochs)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.legend(loc='lower right')
plt.title('Accuracy')
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.legend(loc='upper right')
plt.title('Loss')
plt.grid(True)
plt.savefig("training_history.png", dpi=300, bbox_inches='tight')

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Ma trận nhầm lẫn - ' + DATA_DIR)
plt.ylabel('Thực tế')
plt.xlabel('Dự đoán')
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
print("đã lưu:", OUT_MODEL, "training_history.png", "confusion_matrix.png")
