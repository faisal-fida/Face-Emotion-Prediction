from mlserver import MLModel
from mlserver.codecs import decode_args

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from helpers import OBJECTS, detect_single_face, preprocessing


class EmotionModel(MLModel):
    async def load(self) -> bool:
        tf.config.set_visible_devices([], "GPU")
        self._model = hub.KerasLayer(".")
        self.ready = True
        return self.ready

    @decode_args
    async def predict(self, image: np.ndarray) -> dict:
        image = detect_single_face(image)
        image = preprocessing(image)
        prediction = self._model(image)
        predictions_max = np.argmax(prediction, axis=1)
        response_data = {"prediction": OBJECTS[predictions_max[0]]}
        return response_data
