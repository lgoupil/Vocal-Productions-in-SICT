from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import numpy as np


class Encoder(Model):
    def __init__(self, latent_dim=64):
        super(Encoder, self).__init__()
        self.conv1 = layers.Conv2D(filters=32, kernel_size=3, strides=2, activation="relu")
        self.conv2 = layers.Conv2D(filters=64, kernel_size=3, strides=2, activation="relu")
        self.conv3 = layers.Conv2D(filters=64, kernel_size=3, strides=2, activation="relu")
        self.flat = layers.Flatten()
        self.dense = layers.Dense(latent_dim, activation="relu")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(x)
        z = self.dense(x)
        return z


class Decoder(Model):    #Decoder network
    def __init__(self, input_shape):
        super(Decoder, self).__init__()
        self.dense1 = layers.Dense(units=int( 64 * np.round(input_shape[0]/8) * np.round(input_shape[1]/8) ), activation="relu") #64 channels, input_shape[1]/8 * input_shape[1]/8 pixels
        self.reshape = layers.Reshape(target_shape=( int(np.round(input_shape[0]/8)), int(np.round(input_shape[1]/8)), 64 ))
        self.deconv1 = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation="relu")
        self.deconv2 = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation="relu")
        self.deconv3 = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation="relu")
        self.deconv_last = layers.Convolution2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')           #1 filters to restore the single channel

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return self.deconv_last(x)


class Autoencoder(Model):
    def __init__(self, input_shape, latent_dim=64):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(input_shape=input_shape)

    def call(self, input_image):
        z = self.encoder(input_image)
        return self.decoder(z)