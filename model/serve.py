import os
import numpy as np
from mlserver import MLModel
from mlserver.codecs import decode_args

# from helpers import OBJECTS, detect_single_face, preprocessing

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf  # noqa: F401, E402
import tensorflow_hub as hub  # noqa: F401, E402


class EmotionModel(MLModel):
    async def load(self) -> bool:
        tf.config.set_visible_devices([], "GPU")
        self._model = hub.KerasLayer("model/")
        self.ready = True
        return self.ready

    # @decode_args
    async def predict(self, image: str) -> np.ndarray:
        print(f"Received {image} of type {type(image)}\n\n")
        # image = detect_single_face(b64_image)
        # if image is None:
        #     return np.array(["No face detected"])
        # else:
        #     print(f"Face detected in image of shape {image.shape}\n\n")
        # print("Preprocessing image\n\n")
        # image = preprocessing(image)
        # print("Predicting image\n\n")
        # prediction = self._model(image)

        # predictions_max = np.argmax(prediction, axis=1)
        # response_data = np.array([OBJECTS[i] for i in predictions_max])
        # return response_data
        return np.array([0.0])
