import json
from nameko.web.handlers import http
from irisclassifier.src import *


class IrisClassifierService(object):

    name = 'iris_classifier_service'

    def __init__(self):
        self.classifier = models.predict_model.IrisClassifier("iris.pickle")

    @http('POST', '/classify')
    def classify(self, request):
        try:
            new_data = request.get_data(as_text=True)
            new_data = json.loads(new_data)
            prediction = self.classifier.predict(new_data)
            return json.dumps({
                'status': 'success',
                'message': 'Classified input data',
                'predictions': prediction.tolist()
            })

        except ValueError:
            return json.dumps({
                'status': 'failed',
                'message': 'Input is not valid JSON data'
            })

        except Exception as e:
            return json.dumps({
                'status': 'failed',
                'message': 'Some error occured',
                'error': e
            })
