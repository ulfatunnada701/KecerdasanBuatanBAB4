
# Example 4.13 - WebcamClassify

import cv2
from keras.utils import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.applications import vgg16
import numpy as np

image_size = 224
model = vgg16.VGG16(weights='imagenet')
model.summary()

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Webcam tidak terdeteksi!")
    exit()

while camera.isOpened():
    ok, cam_frame = camera.read()

    if not ok:
        print("Gagal membaca frame kamera.")
        break

    frame = cv2.resize(cam_frame, (image_size, image_size))
    numpy_image = img_to_array(frame)
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_image = vgg16.preprocess_input(image_batch.copy())

    # Get predicted probabilities for each class
    predictions = model.predict(processed_image, verbose=0)
    label = decode_predictions(predictions)

    # Format final image visualization
    cv2.putText(
        cam_frame,
        "VGG16: {}, {:.1f}".format(
            label[0][0][1],
            label[0][0][2]
        ),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow('video image', cam_frame)

    key = cv2.waitKey(30)
    if key == 27:  # Press ESC to quit
        break

camera.release()
cv2.destroyAllWindows()
