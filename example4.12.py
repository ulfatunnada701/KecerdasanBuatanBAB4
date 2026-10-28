
# Example 4.12 - MultiClassify

from pathlib import Path
from keras.utils import load_img, img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.applications import (
    vgg16,
    resnet50,
    mobilenet,
    inception_v3
)
import numpy as np

# Init the models
# model = vgg16.VGG16(weights='imagenet')
# model = resnet50.ResNet50(weights='imagenet')
model = mobilenet.MobileNet(weights='imagenet')
# model = inception_v3.InceptionV3(weights='imagenet')

model.summary()

# Load image
img_path = Path(__file__).resolve().parent / 'Elephant.jpg'
img = load_img(img_path, target_size=(224, 224))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)

# Preprocess image
# processed_image = vgg16.preprocess_input(x)
# processed_image = resnet50.preprocess_input(x)
processed_image = mobilenet.preprocess_input(x)
# processed_image = inception_v3.preprocess_input(x)

# Prediction
predictions = model.predict(processed_image)
results = decode_predictions(predictions)
print(results)
