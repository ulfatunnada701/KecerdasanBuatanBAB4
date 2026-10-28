
# Exercise 4.12 - Webcam Classification with DenseNet

import cv2
import time
import numpy as np

from keras.applications import (
    vgg16,
    resnet50,
    mobilenet,
    inception_v3,
    densenet
)
from keras.applications.imagenet_utils import decode_predictions

# Pilih salah satu:
# VGG16, ResNet50, MobileNet, InceptionV3, DenseNet121
MODEL_NAME = "DenseNet121"

# Konfigurasi model
models = {
    "VGG16": (vgg16.VGG16, vgg16.preprocess_input, 224),
    "ResNet50": (resnet50.ResNet50, resnet50.preprocess_input, 224),
    "MobileNet": (mobilenet.MobileNet, mobilenet.preprocess_input, 224),
    "InceptionV3": (
        inception_v3.InceptionV3,
        inception_v3.preprocess_input,
        299
    ),
    "DenseNet121": (
        densenet.DenseNet121,
        densenet.preprocess_input,
        224
    )
}

# Load model
model_class, preprocess, image_size = models[MODEL_NAME]
model = model_class(weights='imagenet')

print(f"\n===== MODEL {MODEL_NAME} =====")
model.summary()
print(f"Total Parameters: {model.count_params():,}")

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Webcam tidak terdeteksi!")
    raise SystemExit

total_time = 0
total_frames = 0
total_confidence = 0

print("\nTekan ESC untuk keluar.")

try:
    while camera.isOpened():
        ok, cam_frame = camera.read()

        if not ok:
            print("Gagal membaca frame kamera.")
            break

        # Resize gambar
        frame = cv2.resize(
            cam_frame, (image_size, image_size)
        )

        # OpenCV menggunakan BGR.
        # VGG16 dan ResNet50 memerlukan BGR,
        # sedangkan model lain memerlukan RGB.
        if MODEL_NAME in ["MobileNet", "InceptionV3", "DenseNet121"]:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Ubah gambar menjadi batch
        image_batch = np.expand_dims(
            frame.astype("float32"), axis=0
        )

        processed_image = preprocess(image_batch.copy())

        # Hitung waktu prediksi
        start_time = time.perf_counter()

        predictions = model.predict(
            processed_image, verbose=0
        )

        prediction_time = time.perf_counter() - start_time

        # Hasil klasifikasi
        label = decode_predictions(predictions, top=1)
        class_name = label[0][0][1]
        confidence = float(label[0][0][2]) * 100

        # Hitung FPS prediksi
        fps = 1 / prediction_time if prediction_time > 0 else 0

        total_time += prediction_time
        total_frames += 1
        total_confidence += confidence

        # Tampilkan hasil pada webcam
        cv2.putText(
            cam_frame, f"Model: {MODEL_NAME}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

        cv2.putText(
            cam_frame, f"Object: {class_name}",
            (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

        cv2.putText(
            cam_frame, f"Confidence: {confidence:.2f}%",
            (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

        cv2.putText(
            cam_frame, f"Prediction FPS: {fps:.2f}",
            (10, 135), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

        cv2.imshow("Exercise 4.12 - Webcam", cam_frame)

        # Tekan ESC untuk keluar
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    camera.release()
    cv2.destroyAllWindows()

# Tampilkan hasil evaluasi performa
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
