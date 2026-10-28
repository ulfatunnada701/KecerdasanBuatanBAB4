
# Exercise 4.11 - VGG16 vs VGG19 Webcam Classification

import cv2
import time
import numpy as np
from keras.utils import img_to_array
from keras.applications import VGG16, VGG19
from keras.applications.vgg16 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions

# Pilih model: "VGG16" atau "VGG19"
MODEL_NAME = "VGG19"

# Load model
if MODEL_NAME == "VGG16":
    model = VGG16(weights='imagenet')
else:
    model = VGG19(weights='imagenet')

print(f"\n===== MODEL {MODEL_NAME} =====")
model.summary()

print(f"Total Parameters: {model.count_params():,}")

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Webcam tidak terdeteksi!")
    raise SystemExit

image_size = 224
total_time = 0
total_frames = 0
total_confidence = 0

print("\nTekan ESC untuk keluar.")

while camera.isOpened():
    ok, cam_frame = camera.read()

    if not ok:
        print("Gagal membaca frame webcam.")
        break

    # Preprocessing
    frame = cv2.resize(cam_frame, (image_size, image_size))
    numpy_image = img_to_array(frame)
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_image = preprocess_input(image_batch.copy())

    # Menghitung waktu prediksi
    start_time = time.perf_counter()

    predictions = model.predict(processed_image, verbose=0)

    prediction_time = time.perf_counter() - start_time

    # Decode prediction
    label = decode_predictions(predictions, top=1)

    class_name = label[0][0][1]
    confidence = label[0][0][2] * 100

    # Hitung FPS prediksi
    fps = 1 / prediction_time if prediction_time > 0 else 0

    total_time += prediction_time
    total_frames += 1
    total_confidence += confidence

    # Tampilkan hasil pada webcam
    cv2.putText(cam_frame, f"Model: {MODEL_NAME}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    cv2.putText(cam_frame, f"Object: {class_name}",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    cv2.putText(cam_frame, f"Confidence: {confidence:.2f}%",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    cv2.putText(cam_frame, f"FPS: {fps:.2f}",
                (10, 135), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    cv2.imshow('Webcam Classification', cam_frame)

    # ESC untuk keluar
    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()

# Hasil performa
if total_frames > 0:
    avg_time = total_time / total_frames
    avg_fps = total_frames / total_time
    avg_confidence = total_confidence / total_frames

    print("\n===== HASIL PERFORMA =====")
    print(f"Model              : {MODEL_NAME}")
    print(f"Total Parameters   : {model.count_params():,}")
    print(f"Total Frames       : {total_frames}")
    print(f"Average Time       : {avg_time:.4f} detik")
    print(f"Average FPS        : {avg_fps:.2f}")
    print(f"Average Confidence : {avg_confidence:.2f}%")
else:
    print("Tidak ada frame yang berhasil diproses.")
