import tensorflow as tf
import numpy as np


class ValueNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32, 
            kernel_size=(2, 2),
            padding="same",
            activation=tf.keras.layers.PReLU())
        self.conv2 = tf.keras.layers.Conv2D(
            filters=16, 
            kernel_size=(2, 2),
            padding="same",
            activation=tf.keras.layers.PReLU())
        self.conv3 = tf.keras.layers.Conv2D(
            filters=8, 
            kernel_size=(2, 2),
            padding="same",
            activation=tf.keras.layers.PReLU())

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(
            units=32,
            activation=tf.keras.layers.PReLU())
        self.dense2 = tf.keras.layers.Dense(
            units=16,
            activation=tf.keras.layers.PReLU())
        self.output_layer = tf.keras.layers.Dense(
            units=1,
            activation=tf.keras.activations.tanh,
            kernel_initializer="zeros",
            bias_initializer="zeros")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.output_layer(x)
        return x

if __name__ == "__main__":
    value_est = ValueNetwork()

    state = np.ones((10, 3, 3, 1), dtype=np.float32)
    values = value_est(state)
    print(values)