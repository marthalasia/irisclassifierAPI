import numpy
import pickle
import os
from pathlib import Path


class IrisClassifier:

    def __init__(self, model_name):
        cwd = os.path.dirname(os.path.abspath(os.path.join(__file__, "../..")))
        self.path = Path(cwd + "/models/" + model_name)

    def predict(self, data):
        data = numpy.array(data)
        model = self.load_model()
        prediction = []
        if model is not None:
            prediction = self.load_model().predict(data)
        return prediction

    def load_model(self):
        model = None
        if self.path.exists():
            with open(self.path, 'rb') as file:
                model = pickle.load(file)
        return model

