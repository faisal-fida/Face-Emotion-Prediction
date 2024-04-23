import os
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

SIZE = 48
MODEL_PATH = "model.h5"
HAARCASCADE_PATH = "utils/face.xml"
OBJECTS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")


def detect_single_face(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(SIZE, SIZE))
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_image = gray[y : y + h, x : x + w]
        return face_image
    else:
        raise Exception("No face detected, skipping prediction")


def preprocessing(face_image):
    x = cv2.resize(face_image, (SIZE, SIZE))
    x = x.reshape(x.shape + (1,))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    return x / 255


def load_and_predict_model(image_path, model):
    face_image = detect_single_face(image_path)
    preprocessed = preprocessing(face_image)
    predictions = model.predict(preprocessed)
    predictions = np.argmax(predictions[0])
    # Show image and prediction
    cv2.imshow("Image", cv2.imread(image_path))
    return OBJECTS[predictions]


def main():
    model = load_model(MODEL_PATH)
    for i in os.listdir("images"):
        if i.endswith(".jpg"):
            image_path = os.path.join("images", i)
            prediction = load_and_predict_model(image_path, model)
            print(f"Prediction for {i}: {prediction}")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
