import numpy as np
from skimage import io, exposure, transform
import tensorflow as tf
import tensorflow_addons as tfa

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def load_and_preprocess_image(img_path):
    x = io.imread(img_path)
    x = exposure.adjust_log(x)
    d = np.abs(x.shape[0] - x.shape[1]) // 2
    if x.shape[0] > x.shape[1]:
        x = np.pad(x, ((0, 0), (d, d), (0, 0)), mode="constant")
    if x.shape[0] < x.shape[1]:
        x = np.pad(x, ((d, d), (0, 0), (0, 0)), mode="constant")
    x = transform.resize(x, (224, 224), anti_aliasing=True)
    x = [transform.rotate(x, np.random.uniform(-360, 360)) for _ in range(15)]
    x = np.stack(x)
    return x

def build_model(base_model):
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = inputs
    x = base_model(x, training=True)
    x = tf.keras.layers.Dense(128, activation=tfa.activations.mish, activity_regularizer=tf.keras.regularizers.L1L2(1e-3, 1e-3))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation=tfa.activations.mish, activity_regularizer=tf.keras.regularizers.L1L2(1e-3, 1e-3))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = [tf.keras.layers.Dense(128, activity_regularizer=tf.keras.regularizers.L1L2(1e-3, 1e-3))(x), tf.keras.layers.Dense(128, activity_regularizer=tf.keras.regularizers.L1L2(1e-3, 1e-3))(x)]
    x = Sampling()(x)
    x = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="output1")(x)
    output1 = x
    x = tf.keras.layers.Dense(1, activation="sigmoid", name="output2")(x)
    output2 = x
    x = tf.keras.Model(inputs, [output1, output2])
    return x
