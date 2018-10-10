from flask import Flask, request
import json
import sys
sys.path.append('../../')
from irisclassifier import IrisClassifier

app = Flask(__name__)


@app.route('/classify', methods=['POST'])
def classify():
    try:
        new_data = request.get_data(as_text=True)
        new_data = json.loads(new_data)
        prediction = IrisClassifier("iris.pickle").predict(new_data)
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
            'message': 'Some error occurred',
            'error': e
        })


if __name__ == '__main__':
    app.run(host="0.0.0.0")


