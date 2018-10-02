from flask import Flask, request
from src import *
import json

app = Flask(__name__)


@app.route('/classify', methods=['POST'])
def classify():
    try:
        new_data = request.get_data(as_text=True)
        new_data = json.loads(new_data)
        prediction = src.models.predict_model.IrisClassifier("iris.pickle").predict(new_data)
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


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False)


