from flask import Flask, request, jsonify, render_template
import os
import base64
import numpy as np
import cv2
from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv()

application = Flask(__name__)  

rf = Roboflow(api_key=os.getenv('roboflow_apikey'))  
project = rf.workspace().project("-garbage")  
model = project.version("1").model  
@application.route('/')
def index():
    return render_template('index.html')

@application.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['image']
        img_data = base64.b64decode(data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  

        result = model.predict(img, confidence=30, overlap=10).json()

        for prediction in result.get("predictions", []):
            for key, value in prediction.items():
                if isinstance(value, np.ndarray): 
                    prediction[key] = value.tolist()

        return jsonify(result)  

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    application.run(debug=True)
