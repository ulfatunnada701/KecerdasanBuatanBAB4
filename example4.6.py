# Example 4.6
from tensorflow.keras.applications import InceptionV3

# Initialize model
model = InceptionV3(weights='imagenet')

# Show model summary
model.summary()