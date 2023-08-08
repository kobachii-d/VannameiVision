import numpy as np

import argparse
import tensorflow as tf

from util import read, build

# make prediction
def predict(image_path):
    # base model
    base_model = tf.keras.applications.DenseNet121(weights="imagenet", input_shape=(224, 224, 3), pooling="avg", include_top=False)
    base_model.trainable = True
    # create model
    model = build(base_model)
    model.load_weights("model/DenseNet121-Triplet-ImageNet.h5")
    model.trainable = False
    # read, then preprocess
    x = read(image_path)
    # make prediction
    x = model.predict(x, verbose=0)
    x = x[1]
    x = x.reshape(np.prod(x.shape))
    # convert to binary
    x = (x >= 0.3).astype(int)
    # print to display results
    if len(set(x)) == 1:
        print(f"Prediction: {['Robust', 'Susceptible'][x[0]]}\nConfidence: 100.0%")
    else:
        d = np.bincount(x)
        print(f"Prediction: {['Robust', 'Susceptible'][np.argmax(d)]}\nConfidence: {str(np.round(d[np.argmax(d)] / np.sum(d) * 100, 1)).rjust(5, ' ')}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict if an image is Robust or Susceptible.")
    parser.add_argument("image_path", type=str, help="Path to the image to be analyzed.")
    args = parser.parse_args()
    predict(args.image_path)
