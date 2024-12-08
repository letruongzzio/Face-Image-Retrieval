import os
from typing import Callable
import joblib

import numpy as np
from tensorflow import keras
from sklearn.neighbors import KDTree

from prepare import download_data

DIR_PATH = os.path.dirname(__file__)


def build_embedding(
    image_path: str, predict_func: Callable, preprocess_func: Callable = None
) -> np.ndarray:
    """
    Build embedding vector from the image using the passed functions.

    Parameters:
    -----------
    image_path: str
        The path to the image
    predict_func: Callable
        The function to predict the image, from the CNN
    preprocess_func: Callable, default None
        The function to preprocess the image. If not passed, the image will be passed directly to the predict function
    """
    image = keras.utils.load_img(image_path)
    image = keras.utils.img_to_array(np.array(image))
    if preprocess_func is not None:
        image: np.ndarray = preprocess_func(image)
    image = np.expand_dims(image, axis=0)
    return predict_func(image).flatten()


def build_query(features_vector: np.ndarray, model_name: str):
    query_dirpath = os.path.join((os.path.dirname(DIR_PATH)), "retrieval")
    os.makedirs(query_dirpath, exist_ok=True)

    tree = KDTree(features_vector)

    joblib.dump(tree, os.path.join(query_dirpath, f"{model_name}.joblib"))


if __name__ == "__main__":
    mobile_net: keras.Model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), weights="imagenet", include_top=False, pooling="max"
    )
    image_dirpath = os.path.join(
        download_data(), "img_align_celeba", "img_align_celeba"
    )
    celeb_features = np.array(
        [
            build_embedding(
                os.path.join(image_dirpath, image_path),
                predict_func=mobile_net.predict,
                preprocess_func=keras.applications.mobilenet_v2.preprocess_input,
            )
            for image_path in os.listdir(image_dirpath)
        ]
    )
    build_query(celeb_features, "mobilenet")
