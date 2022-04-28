import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class CNN(layers.Layer):
    def __init__(self, units, kernel_size, strides, padding):
        super(CNN, self).__init__()
        self.cnn = layers.Conv2D(filters=units, kernel_size=kernel_size, strides=strides, padding=padding)
        self.bn = layers.BatchNormalization()
    
    def call(self, x, training=False):
        x = self.cnn(x)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x
    
class model64(keras.Model):
    def __init__(self):
        super(model64, self).__init__()
        self.layer1 = CNN(units=128, kernel_size=2, strides=1, padding='same')
        self.layer2 = layers.MaxPool2D(pool_size=(2, 2))
        self.layer3 = layers.Dropout(rate=0.3)
        self.layer4 = CNN(units=80, kernel_size=3, strides=1, padding='same')
        self.layer5 = layers.MaxPool2D(pool_size=(2, 2))
        self.layer6 = CNN(units=60, kernel_size=3, strides=1, padding='same')
        self.layer7 = layers.MaxPool2D(pool_size=(2, 2))
        self.layer8 = CNN(units=50, kernel_size=3, strides=1, padding='same')
        self.layer9 = layers.MaxPool2D(pool_size=(2, 2))
        self.flatten = layers.Flatten()
        self.layer10 = layers.Dense(units=512, activation='sigmoid')
        self.layer11 = layers.Dropout(rate=0.3)
        self.layer12 = layers.Dense(units=1, activation='sigmoid')
    
    def call(self, x, training=False):
        x = self.layer1(x, training=training)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x, training=training)
        x = self.layer5(x)
        x = self.layer6(x, training=training)
        x = self.layer7(x)
        x = self.layer8(x, training=training)
        x = self.layer9(x)
        x = self.flatten(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        return x

    def model(self):
        x = keras.Input(shape=(64, 64, 3))
        return keras.Model(inputs=[x], outputs=[self.call(x)])
    
