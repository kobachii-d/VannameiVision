import numpy as np
import tensorflow as tf
import argparse
from util import load_and_preprocess_image, build_model

def main(image_path):
    base_model = tf.keras.applications.DenseNet121(weights="imagenet", input_shape=(224, 224, 3), pooling="avg", include_top=False)
    base_model.trainable = True
    model = build_model(base_model)
    model.load_weights("model/DenseNet121-Triplet-ImageNet.h5")
    model.trainable = False

    x = load_and_preprocess_image(image_path)
    x = model.predict(x, verbose=0)
    x = x[1]
    x = x.reshape(np.prod(x.shape))

    thr = 0.3
    x = (x >= thr).astype(int)

    if len(set(x)) == 1:
        print(f"Prediction: {['Robust', 'Susceptible'][x[0]]}\nConfidence: 100.0%")
    else:
        d = np.bincount(x)
        print(f"Prediction: {['Robust', 'Susceptible'][np.argmax(d)]}\nConfidence: {str(np.round(d[np.argmax(d)] / np.sum(d) * 100, 1)).rjust(5, ' ')}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict if an image is Robust or Susceptible.")
    parser.add_argument("image_path", type=str, help="Path to the image to be analyzed.")
    args = parser.parse_args()
    main(args.image_path)
