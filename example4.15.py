
# Example 4.15 - Webcam Classification
# Modified for modern Keras

import cv2
import numpy as np

from keras.applications import (
    VGG16,
    ResNet50,
    MobileNetV2,
    DenseNet201,
    InceptionV3
)

from keras.applications.vgg16 import preprocess_input as vgg_preprocess
from keras.applications.resnet50 import preprocess_input as resnet_preprocess
from keras.applications.mobilenet_v2 import preprocess_input as mobile_preprocess
from keras.applications.densenet import preprocess_input as dense_preprocess
from keras.applications.inception_v3 import preprocess_input as inception_preprocess
from keras.applications.imagenet_utils import decode_predictions

# Set up a model
# Pilih: VGG16, ResNet50, MobileNetV2, DenseNet201, InceptionV3
MODEL_NAME = "VGG16"

models = {
    "VGG16": (VGG16, vgg_preprocess, 224),
    "ResNet50": (ResNet50, resnet_preprocess, 224),
    "MobileNetV2": (MobileNetV2, mobile_preprocess, 224),
    "DenseNet201": (DenseNet201, dense_preprocess, 224),
    "InceptionV3": (InceptionV3, inception_preprocess, 299)
}

clf, preprocess_input, sz = models[MODEL_NAME]

model = clf(
    input_shape=(sz, sz, 3),
    weights='imagenet',
    classes=1000
)

model.summary()

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Webcam tidak terdeteksi!")
    raise SystemExit

image_size = sz

while camera.isOpened():
    ret, cam_frame = camera.read()

    if not ret:
        print("Gagal membaca frame kamera.")
        break

    # Resize frame
    frame = cv2.resize(
        cam_frame, (image_size, image_size)
    )

    # Convert BGR to RGB for models that require RGB
    if MODEL_NAME in ["MobileNetV2", "DenseNet201", "InceptionV3"]:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocessing
    image = np.asarray(frame, dtype=np.float32)
    image = np.expand_dims(image, 0)
    image = preprocess_input(image)

    # Prediction
    preds = model.predict(image, verbose=0)
    label = decode_predictions(preds)

    # Display classification
    cv2.putText(
        cam_frame,
        "{}, {:.1f}".format(
            label[0][0][1],
            label[0][0][2]
        ),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.imshow("Classification", cam_frame)

    key = cv2.waitKey(30)

    if key == 27:  # Press ESC to quit
        break

camera.release()
cv2.destroyAllWindows()
