import numpy
import pickle
import os.path


class IrisClassifier:

    def __init__(self):
        self.cwd = os.getcwd()
        self.path = os.path.join(self.cwd, 'irisclassifierapi/models/iris_classifier.pickle')

        self.model = pickle.load(open(self.path, 'rb'))

    def predict(self, data):
        data = numpy.array(data)

        return self.model.predict(data)
