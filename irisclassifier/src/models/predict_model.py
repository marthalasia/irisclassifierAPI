import numpy
import pickle
import os.path


class IrisClassifier:

    def __init__(self):
        self.path = os.path.join(os.getcwd(), 'irisclassifier/models/iris_classifier.pickle')
        self.model = pickle.load(open(self.path, 'rb'))

    def predict(self, data):
        data = numpy.array(data)
        return self.model.predict(data)
