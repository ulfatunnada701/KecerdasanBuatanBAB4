
# Exercise 4.14 - CNN Model Selection Using If-Else

import cv2
import numpy as np
from keras.applications import (
    vgg16, resnet50, mobilenet_v2,
    densenet, inception_v3
)
from keras.applications.imagenet_utils import decode_predictions

# Menampilkan pilihan model
print("===== PILIH MODEL CNN =====")
print("1. VGG16")
print("2. ResNet50")
print("3. MobileNetV2")
print("4. DenseNet201")
print("5. InceptionV3")

choice = input("Masukkan pilihan model (1-5): ")

# Pemilihan model menggunakan if-else
if choice == "1":
    name = "VGG16"
    clf = vgg16.VGG16
    preprocess = vgg16.preprocess_input
    image_size = 224

elif choice == "2":
    name = "ResNet50"
    clf = resnet50.ResNet50
    preprocess = resnet50.preprocess_input
    image_size = 224

elif choice == "3":
    name = "MobileNetV2"
    clf = mobilenet_v2.MobileNetV2
    preprocess = mobilenet_v2.preprocess_input
    image_size = 224

elif choice == "4":
    name = "DenseNet201"
    clf = densenet.DenseNet201
    preprocess = densenet.preprocess_input
    image_size = 224

elif choice == "5":
    name = "InceptionV3"
    clf = inception_v3.InceptionV3
    preprocess = inception_v3.preprocess_input
    image_size = 299

else:
    print("Pilihan tidak valid!")
    raise SystemExit

# Load model terpilih
print(f"\nModel yang dipilih: {name}")

model = clf(weights='imagenet')
model.summary()

# Mengaktifkan webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Webcam tidak terdeteksi!")
    raise SystemExit

print("Tekan ESC untuk keluar.")

try:
    while True:
        ret, cam_frame = camera.read()

        if not ret:
            break

        # Resize gambar
        frame = cv2.resize(
            cam_frame, (image_size, image_size)
        )

        # Konversi warna jika diperlukan
        if name in ["MobileNetV2", "DenseNet201", "InceptionV3"]:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Preprocessing
        image = np.expand_dims(
            frame.astype("float32"), axis=0
        )
        image = preprocess(image)

        # Prediksi
        predictions = model.predict(image, verbose=0)
        label = decode_predictions(predictions, top=1)

        class_name = label[0][0][1]
        confidence = float(label[0][0][2]) * 100

        # Menampilkan hasil
        cv2.putText(
            cam_frame, f"Model: {name}",
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

        cv2.imshow("Exercise 4.14", cam_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
