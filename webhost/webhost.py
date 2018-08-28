import flask
from flask import Flask, jsonify, request
import numpy as np
import PIL
from PIL import Image
from keras.models import load_model
import io
from keras.applications.vgg16 import decode_predictions
import base64

app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict_image():

       # Preprocess the image so that it matches the training input
       if request.get_json()['image']:
       #imagefile = flask.request.files.get('imagefile', '')
               image = request.get_json()['image']
               data = {}
               with open("imageToPredict.png", "wb") as fh:
                       fh.write(base64.b64decode(image))

               image = Image.open("imageToPredict.png")
               image = image.resize((224,224))
               image = np.asarray(image)
               image = image.reshape(1,224,224,3)
              
               model = load_model('vgg16_model.h5')
               model.compile(loss = 'binary_crossentropy', optimizer='adam')
               # Use the loaded model to generate a prediction.
               pred = model.predict(image)

               # convert the probabilities to class labels
               label = decode_predictions(pred)
               # retrieve the most likely result, e.g. highest probability
               data["predictions"] = []

               for (imagenetID, label, prob) in label[0]:
                       r = {"label": label, "probability": float(prob)}
                       data["predictions"].append(r)

               # indicate that the request was a success
               data["status"] = "OK"

               # Prepare and send the response.
               # digit = np.argmax(pred)
               # prediction = {'digit':int(digit)}
               return flask.jsonify(data)

if __name__ == "__main__":
        app.run()

