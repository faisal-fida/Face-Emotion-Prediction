import cv2
import base64
import numpy as np


SIZE = 48
OBJECTS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")

face_cascade = cv2.CascadeClassifier("utils/face.xml")


def b64_to_image(b64_string):
    """
    Convert base64 string to image
    """
    sbuf = base64.decodebytes(b64_string.encode())
    pimg = np.frombuffer(sbuf, dtype=np.uint8)
    image = cv2.imdecode(pimg, cv2.IMREAD_UNCHANGED)
    return image


def detect_single_face(b64_string):
    """
    Detects a single face in an image and returns the face image
    """

    image = b64_to_image(b64_string)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(SIZE, SIZE))
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_image = gray[y : y + h, x : x + w]
        return face_image
    else:
        return None


def preprocessing(face_image):
    """
    Preprocesses the face image
    """

    x = cv2.resize(face_image, (SIZE, SIZE))
    x = x.reshape(x.shape + (1,))
    x = x.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    return x / 255
