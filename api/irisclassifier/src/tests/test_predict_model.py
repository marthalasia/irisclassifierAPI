import unittest
from src.models import *
import json


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.data = json.loads(str([[5.1, 3.5, 1.4, 0.2]]))

    def test_load_model_success(self):
        classifier = IrisClassifier("iris.pickle")
        classifier.path = Path("/Users/marthalasia/Documents/Work/DataScienceProduction/irisclassifierAPI/api"
                               "/irisclassifier/models/iris.pickle")
        model = classifier.load_model()
        self.assertIsNotNone(model)

    def test_load_model_fail(self):
        classifier = IrisClassifier("iris.pickle")
        classifier.path = Path("/Users/marthalasia/Documents/Work/DataScienceProduction/irisclassifierAPI/api"
                               "/irisclassifier/iris.pickle")
        model = classifier.load_model()
        self.assertIsNone(model)

    def test_predict_success(self):
        classifier = IrisClassifier("iris.pickle")
        classifier.path = Path("/Users/marthalasia/Documents/Work/DataScienceProduction/irisclassifierAPI/api"
                               "/irisclassifier/models/iris.pickle")
        prediction = classifier.predict(self.data)
        self.assertEqual(prediction.tolist(), [0])

    def test_predict_fail_model_none(self):
        classifier = IrisClassifier("iris.pickle")
        classifier.path = Path("/Users/marthalasia/Documents/Work/DataScienceProduction/irisclassifierAPI/"
                               "api/irisclassifier/iris.pickle")
        prediction = classifier.predict(self.data)
        self.assertEqual(prediction, [])

    def test_predict_fail(self):
        classifier = IrisClassifier("iris.pickle")
        classifier.path = Path("/Users/marthalasia/Documents/Work/DataScienceProduction/irisclassifierAPI"
                               "/api/irisclassifier/iris.pickle")
        prediction = classifier.predict({"data": "null"})
        self.assertEqual(prediction, [])


if __name__ == '__main__':
    unittest.main()
