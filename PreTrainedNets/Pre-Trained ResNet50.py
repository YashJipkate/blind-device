
# coding: utf-8

# In[53]:


import keras
from keras.applications import resnet50, vgg16, mobilenet, inception_v3
from keras.applications.imagenet_utils import decode_predictions
import cv2
import numpy as np
import pyttsx

engine = pyttsx.init()

resnet50_model = resnet50.ResNet50(weights='imagenet')


# In[50]:


cap = cv2.VideoCapture(0)

if cap.isOpened():
    i=1
    while True:
        ret, img = cap.read()
        img_input = cv2.resize(img, (224,224))
        img_input = np.array(img_input, dtype=np.float64)
        img_input = np.expand_dims(img_input, axis=0)

        img_input1 = vgg16.preprocess_input(img_input.copy())
        img_input2 = inception_v3.preprocess_input(img_input.copy())
        img_input3 = resnet50.preprocess_input(img_input.copy())
        img_input4 = mobilenet.preprocess_input(img_input.copy())


        predictions = resnet50_model.predict(img_input3)
        label = decode_predictions(predictions)
        
        if i%25 == 0:
            engine.say(label[0][0][1])
            engine.runAndWait()

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
            
else:
    print("Camera Not Found Probably")
        
cap.release()
cv2.destroyAllWindows()

