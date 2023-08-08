import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa

from skimage import io, exposure, transform

# read, then preprocess
def read(img_path):
    # read
    x = io.imread(img_path)
    # enhance
    x = exposure.adjust_log(x)
    # padding
    d = np.abs(x.shape[0] - x.shape[1]) // 2
    if x.shape[0] > x.shape[1]: x = np.pad(x, ((0, 0), (d, d), (0, 0)), mode="constant")
    if x.shape[0] < x.shape[1]: x = np.pad(x, ((d, d), (0, 0), (0, 0)), mode="constant")
    # resize
    x = transform.resize(x, (224, 224), anti_aliasing=True)
    # augment
    x = [transform.rotate(x, np.random.uniform(-360, 360)) for _ in range(15)]
    x = np.stack(x)
    return x

# probabilitic sampling
# https://keras.io/examples/generative/vae/
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        # mean and log variance
        z_mean, z_log_var = inputs
        # epsilon
        # from standard normal
        batch =tf.shape(z_mean)[0]
        dim   =tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        # actual sampling
        return z_mean +tf.exp(0.5 *z_log_var) *epsilon

# create model
def build(base_model):
    # inputs
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = inputs
    # base model
    x = base_model(x, training=True)
    # fully connected layer
    x = tf.keras.layers.Dense(128, activation=tfa.activations.mish, activity_regularizer=tf.keras.regularizers.L1L2(1e-3, 1e-3))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation=tfa.activations.mish, activity_regularizer=tf.keras.regularizers.L1L2(1e-3, 1e-3))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    # probabilitic sampling
    x = [tf.keras.layers.Dense(128, activity_regularizer=tf.keras.regularizers.L1L2(1e-3, 1e-3))(x), tf.keras.layers.Dense(128, activity_regularizer=tf.keras.regularizers.L1L2(1e-3, 1e-3))(x)]
    x = Sampling()(x)
    # feature output
    x = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="output1")(x)
    output1 = x
    # classification output
    x = tf.keras.layers.Dense(1, activation="sigmoid", name="output2")(x)
    output2 = x
    # build model
    x = tf.keras.Model(inputs, [output1, output2])
    return x
