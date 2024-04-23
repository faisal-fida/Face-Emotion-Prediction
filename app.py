import os
import cv2
import time
import base64
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.models import load_model  # noqa: E402
from keras.preprocessing import image  # noqa: E402
import tensorflow as tf  # noqa: F401, E402

avg_time = []

SIZE = 48
TEST_IMAGES = "utils/test_images"
OBJECTS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")

face_cascade = cv2.CascadeClassifier("utils/face.xml")
model = load_model("utils/model.h5")

with open("cb-base64-string.txt", "r") as f:
    base64_string = f.read()
    sbuf = base64.decodebytes(base64_string.encode())
    pimg = np.frombuffer(sbuf, dtype=np.uint8)
    IMAGE = cv2.imdecode(pimg, cv2.IMREAD_UNCHANGED)


def detect_single_face(image):
    """
    Detects a single face in an image and returns the face image
    """

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
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    return x / 255


def predict(model, face_image):
    """
    Load the model and predict emotions from the image
    """

    if face_image is None:
        return "No face detected"
    preprocessed = preprocessing(face_image)
    predictions = model.predict(preprocessed)
    predictions = OBJECTS[np.argmax(predictions)]
    return predictions, avg_time


def main():
    time_start = time.time()
    face_image = detect_single_face(IMAGE)
    prediction, avg_time = predict(model, face_image)
    time_end = time.time()
    avg_time.append(time_end - time_start)
    print(f"Prediction: {prediction}, Time: {round(sum(avg_time) / len(avg_time), 4)}")

    # Save the face image
    # file_name = f"pred/{prediction}_{time.time()}.png"
    # cv2.imwrite(file_name, face_image)


if __name__ == "__main__":
    for _ in range(200):
        main()
