from flask import Flask, jsonify, request
import flask
import numpy as np
import PIL
from PIL import Image
from keras.models import load_model
import io
from keras.applications.resnet50 import decode_predictions
import base64
from sklearn.externals import joblib
from boto.s3.key import Key
import boto3
from boto.s3.connection import S3Connection
from flask import json
import botocore

BUCKET_NAME = 'blind-device-bucket0'
MODEL_NAME = 'resnet50_model.h5'
BUCKET_IMAGE_NAME = 'blinddevicebucketandroid'
IMAGE_PATH = 'public/s3Key.txt'

app = Flask(__name__)

@app.route('/hello', methods=['GET'])
def hello():
    return "Hello world"

#@app.route('/test', methods=['POST','GET'])
def test():
    if request.method == 'GET':
        data['gettest'] = "SUccessful GET request"
        return flask.jsonify(data)
    if request.method == 'POST':
        if request.get_json():
            params = request.get_json()
            name = params['name']
            data['posttest'] = "successful POST request" + name
            return flask.jsonify(data)


S3 = boto3.client('s3', region_name='ap-south-1')

@app.route('/', methods=['POST', 'GET'])
def predict_image():
#        payload = json.loads(request.get_data().decode('utf-8'))
       # Preprocess the image so that it matches the training input
    if request.method == 'POST':
        return "<h1>This is wrong type of request fam!!</h1>"
    if request.method == 'GET':
#            if request.files['image']:
           #imagefile = flask.request.files.get('imagefile', '')
    
            s3 = boto3.resource('s3')
            try:
                s3.Bucket(BUCKET_NAME).download_file(MODEL_NAME, 'resnet50_model.h5')
                s3.Bucket(BUCKET_IMAGE_NAME).download_file(IMAGE_PATH, 'image.jpeg')
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    print("The object does not exist.")
                else:
                    raise

#             image = request.files.get('image').read()
            data = {}
#                with open("imageToPredict.png", "wb") as fh:
#                        fh.write(base64.b64decode(image))

#                image = Image.open("imageToPredict.png")
            image = Image.open("image.jpeg")
            image = image.resize((224,224))
            image = np.asarray(image)
            image = image.reshape(1,224,224,3)
#                 print('hello')
#                 response = S3.get_object(Bucket='blind-device-bucket0', Key='resnet50_model.h5')



            model = load_model('resnet50_model.h5')
            print('model loaded')

#                model = load_model('resnet50_model.h5')
            model.compile(loss = 'binary_crossentropy', optimizer='adam')
            print('model compiled')
           # Use the loaded model to generate a prediction.

            print('Making Predictions...')

            pred = model.predict(image)

           # convert the probabilities to class labels
            label = decode_predictions(pred)
           # retrieve the most likely result, e.g. highest probability
            data["predictions"] = []

           # for (imagenetID, label, prob) in label[0]:
               # r = {"label": label, "probability": float(prob)}
               # data["predictions"].append(r)
            data["predictions"] = label[0][0][1] 

           # indicate that the request was a success
            data["status"] = "OK"

           # Prepare and send the response.
           # digit = np.argmax(pred)
           # prediction = {'digit':int(digit)}
            return flask.jsonify(data)

if __name__ == "__main__":
        app.run(host='0.0.0.0')
