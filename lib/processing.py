import os
from typing import Any, Literal
import cv2
from cv2.typing import Rect
import numpy as np
from .constants import IMAGE_SHAPE

DIR_PATH = os.path.dirname(__file__)


def crop_face(
    image, scaleFactor, minNeighbors
) -> (
    tuple[Literal["none"], Literal[0]]
    | tuple[Literal["many"], Literal[0]]
    | tuple[Literal["one"], np.ndarray]
):
    """
    Returns state of images, and the crop image, if it's unique.

    Parameters:
    - image: The image need to be cropped
    - scaleFactor:
    - minNeighbors:
    -----
    Returns:
        A string represents the state of images, which can be "none", "one", "many"
        0 if not "one", a np.ndarray represents the image if "one"
    """
    face_cascade = cv2.CascadeClassifier(
        os.path.join(DIR_PATH, "haarcascade_frontalface_default.xml")
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbors)
    if len(faces) == 0:
        # We may try to decrease the scaleFactor parameter
        faces = face_cascade.detectMultiScale(gray, scaleFactor - 0.1, minNeighbors)
        if len(faces) == 0:
            return ("none", 0)
        elif len(faces) > 1:
            return ("many", 0)
        else:
            return ("one", face_coordination(image, faces[0]))
    elif len(faces) > 1:
        # We may try to increase the scaleFactor parameter
        faces = face_cascade.detectMultiScale(gray, scaleFactor + 0.1, minNeighbors)
        if len(faces) == 0:
            return ("none", 0)
        elif len(faces) > 1:
            return ("many", 0)
        else:
            return ("one", face_coordination(image, faces[0]))

    else:
        return ("one", face_coordination(image, faces[0]))


def face_coordination(image, face_coord: Rect) -> np.ndarray:
    x, y, w, h = face_coord
    a, b, c = 0, 0, 0
    # Crop the face
    if y - 30 >= 0:
        a = 30
    if y + h + 10 < IMAGE_SHAPE[0]:
        b = 10
    if x - 5 >= 0 and x + w + 5 < IMAGE_SHAPE[1]:
        c = 5
    face = image[y - a : y + h + b, x - c : x + w + c]
    return face
