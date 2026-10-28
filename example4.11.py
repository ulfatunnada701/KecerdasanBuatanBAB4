
# Example 4.11 - VGG16A

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
import numpy as np

# Load VGG16 model
model = VGG16(weights='imagenet')

# Load image
img_path = 'Elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Prediction
predictions = model.predict(x)
results = decode_predictions(predictions)

print(results)
