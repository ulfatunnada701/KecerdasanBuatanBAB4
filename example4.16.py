
# Example 4.16 AutoencoderKeras.py
# Modified from:
# https://stackabuse.com/autoencoders-for-image-reconstruction-in-python-and-keras/

# Load the MNIST handwritten digital images
from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print(x_train.shape)
print(x_test.shape)

# Build Autoencoder
from keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from keras.models import Sequential, Model

def build_autoencoder(img_shape, code_size):

    # The encoder
    encoder = Sequential()
    encoder.add(InputLayer(shape=img_shape))
    encoder.add(Flatten())
    encoder.add(Dense(code_size))

    # The decoder
    decoder = Sequential()
    decoder.add(InputLayer(shape=(code_size,)))
    decoder.add(Dense(np.prod(img_shape)))
    decoder.add(Reshape(img_shape))

    return encoder, decoder


IMG_SHAPE = x_train[0].shape
encoder, decoder = build_autoencoder(IMG_SHAPE, 32)

inp = Input(shape=IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = Model(inp, reconstruction)
autoencoder.compile(optimizer='adamax', loss='mse')

autoencoder.summary()

# Train Autoencoder
history = autoencoder.fit(
    x=x_train,
    y=x_train,
    epochs=20,
    validation_data=(x_test, x_test)
)

# Plot the training results
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Plot original, encoded and decoded images
def show_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1), cmap='gray')
    plt.axis('off')


def visualize(img, encoder, decoder):
    """Draws original, encoded and decoded images"""

    code = encoder.predict(img[None], verbose=0)[0]
    reco = decoder.predict(code[None], verbose=0)[0]

    plt.figure(figsize=(10, 3))

    # Original Image
    plt.subplot(1, 3, 1)
    show_image(img)
    plt.title('Original')

    # Encoded Image
    plt.subplot(1, 3, 2)
    plt.imshow(code.reshape(1, -1), cmap='gray', aspect='auto')
    plt.title('Encoded')
    plt.axis('off')

    # Decoded Image
    plt.subplot(1, 3, 3)
    show_image(reco)
    plt.title('Decoded')

    plt.tight_layout()
    plt.show()


# Visualize reconstruction
visualize(x_test[0], encoder, decoder)
