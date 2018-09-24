import unittest
from irisclassifier.src.models.predict_model import *
import json


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.data = ""

    def test_load_model_success(self):
        classifier = IrisClassifier("iris.pickle")
        classifier.path = Path("/Users/marthalasia/Python/irisclassifierAPI/irisclassifier/models/iris.pickle")
        model = classifier.load_model()
        self.assertIsNotNone(model)

    def test_load_model_fail(self):
        classifier = IrisClassifier("iris.pickle")
        classifier.path = Path("/Users/marthalasia/Python/irisclassifierAPI/irisclassifier/iris.pickle")
        model = classifier.load_model()
        self.assertIsNone(model)

    def test_predict_success(self):
        classifier = IrisClassifier("iris.pickle")
        classifier.path = Path("/Users/marthalasia/Python/irisclassifierAPI/irisclassifier/iris.pickle")
        new_data = json.loads(self.data)
        prediction = classifier.predict(new_data)
        self.assertIsNotNone(prediction)

    def test_predict_fail(self):
        classifier = IrisClassifier("iris.pickle")
        classifier.path = Path("/Users/marthalasia/Python/irisclassifierAPI/irisclassifier/iris.pickle")
        prediction = classifier.predict({"data": "null"})
        self.assertEqual(prediction, [])


if __name__ == '__main__':
    unittest.main()
