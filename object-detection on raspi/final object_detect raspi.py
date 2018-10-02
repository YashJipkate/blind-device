# coding: utf-8
# In[1]:
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
os.system('espeak hello')
#os.system('pulseaudio --start')
import tensorflow as tf
#os.system('espeak hello')
import zipfile
#os.system("bluetoothctl")
# In[2]:

os.system('espeak hello')
from collections import defaultdict
from io import StringIO
#from matplotlib import pyplot as plt
#from PIL import Image



# This is needed since the notebook is stored in the object_detection folder.
#sys.path.append("..")


# In[3]:

from utils import label_map_util
#from object_detection.utils import label_map_util
#from utils import visualization_utils as vis_util

# In[4]:

#import time

MODEL_NAME = "/home/pi/object_detection/ssd_mobilenet_v1_coco_2017_11_17"
#MODEL_NAME = "ssd_mobilenet_v1_coco_2017_11_17"
MODEL_FILE = MODEL_NAME
#MODEL_FILE = MODEL_NAME
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
#PATH_TO_CKPT = 'object_detection/'+MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_CKPT =MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = os.path.join("/home/pi/object_detection/data", "mscoco_label_map.pbtxt")
NUM_CLASSES = 90


# In[5]:


#os.system("bluetoothctl")
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# In[7]:


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
os.system('espeak hello')

# In[8]:


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# In[9]:


#PATH_TO_TEST_IMAGES_DIR = 'test_images'
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
i=0
#os.system("pulseaudio --start")
#os.system("bluetoothctl")
#os.system("power on")
#os.system("agent on")
#os.system("default-agent")
#os.system("pacmd list-cards")
#os.system('espeak hello')
from picamera.array import PiRGBArray
from picamera import PiCamera
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        camera = PiCamera()
        camera.vflip=True
        camera.resolution = (2592, 1944)
        camera.framerate = 16
        output= PiRGBArray(camera)
        #os.system("pacmd set-card-profile bluez_card.11_58_02_B8_02_50 a2dp_sink")
        #os.system("pacmd set-default-sink bluez_sink.11_58_02_B8_02_50.a2dp_sink")
        for frame in camera.capture_continuous(output, format="rgb"):
            i+=1
            #os.system("pacmd set-card-profile bluez_card.11_58_02_B8_02_50 a2dp_sink")
            #os.system("pacmd set-default-sink bluez_sink.11_58_02_B8_02_50.a2dp_sink")
            image_np = frame.array
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
                      [boxes, scores, classes, num_detections],
                       feed_dict={image_tensor: np.expand_dims(image_np, 0)})
                    #vis_util.visualize_boxes_and_labels_on_image_array(
                    #       image_np,
                    #       np.squeeze(boxes),
                    #       np.squeeze(classes).astype(np.int32),
                    #        np.squeeze(scores),
                    #      category_index,
                    #      use_normalized_coordinates=True,
                    #       line_thickness=8)
            classes=classes.reshape(classes.shape[1])
            scores=scores.reshape(scores.shape[1])
            d=classes[0]
            e=classes[1]
            a=category_index[d]['name']
            b=category_index[e]['name']
            print(category_index[d]['name'], category_index[e]['name'])
            #os.system('espeak "{}"'.format(a))
            #print(category_index[e]['name'])
            if(scores[1]>=0.5 and classes[1]!=classes[0]):
              os.system('espeak "{}"'.format(a))
              os.system('espeak and')
              os.system('espeak "{}"'.format(b))
            else:
              os.system('espeak "{}"'.format(a))
            output.truncate(0)
            if(i==5):
              break
