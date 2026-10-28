
# Exercise 4.13 - CNN Performance Comparison

import cv2
import time
import gc
import numpy as np
import keras

from keras.applications import (
    vgg16, resnet50, mobilenet_v2,
    densenet, inception_v3
)
from keras.applications.imagenet_utils import decode_predictions

# Daftar model
models = [
    ("VGG16", vgg16.VGG16, vgg16.preprocess_input, 224),
    ("ResNet50", resnet50.ResNet50, resnet50.preprocess_input, 224),
    ("MobileNetV2", mobilenet_v2.MobileNetV2,
     mobilenet_v2.preprocess_input, 224),
    ("DenseNet201", densenet.DenseNet201,
     densenet.preprocess_input, 224),
    ("InceptionV3", inception_v3.InceptionV3,
     inception_v3.preprocess_input, 299)
]

results = []

for name, model_class, preprocess, size in models:
    print(f"\n===== MODEL {name} =====")

    model = model_class(weights='imagenet')
    parameters = model.count_params()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Webcam tidak terdeteksi!")
        break

    total_time = 0
    total_frames = 0
    total_confidence = 0

    print("Tekan ESC untuk model berikutnya.")

    try:
        while True:
            ret, cam_frame = camera.read()

            if not ret:
                break

            frame = cv2.resize(cam_frame, (size, size))

            # Konversi warna untuk model tertentu
            if name in ["MobileNetV2", "DenseNet201", "InceptionV3"]:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = np.expand_dims(
                frame.astype("float32"), axis=0
            )
            image = preprocess(image)

            # Mengukur waktu prediksi
            start = time.perf_counter()
            predictions = model.predict(image, verbose=0)
            elapsed = time.perf_counter() - start

            label = decode_predictions(predictions, top=1)
            class_name = label[0][0][1]
            confidence = float(label[0][0][2]) * 100

            total_time += elapsed
            total_frames += 1
            total_confidence += confidence

            fps = 1 / elapsed if elapsed > 0 else 0

            # Tampilan webcam
            cv2.putText(cam_frame, f"Model: {name}",
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

            cv2.imshow("Exercise 4.13", cam_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    if total_frames > 0:
        avg_time = total_time / total_frames
        avg_fps = total_frames / total_time
        avg_confidence = total_confidence / total_frames

        results.append(
            (name, parameters, avg_time, avg_fps,
             avg_confidence)
        )

    # Bersihkan model sebelum model berikutnya
    del model
    keras.backend.clear_session()
    gc.collect()

# Hasil perbandingan
print("\n========== HASIL PERBANDINGAN ==========")

for name, params, avg_time, fps, confidence in results:
    print(f"\nModel        : {name}")
    print(f"Parameters   : {params:,}")
    print(f"Avg Time     : {avg_time:.4f} detik")
    print(f"FPS          : {fps:.2f}")
    print(f"Confidence   : {confidence:.2f}%")
