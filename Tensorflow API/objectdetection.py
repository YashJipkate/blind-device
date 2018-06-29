
# coding: utf-8

# In[1]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile


# In[2]:


from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image



# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# In[3]:



from utils import label_map_util

from utils import visualization_utils as vis_util


# In[4]:


MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# In[5]:



#opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())



# In[6]:


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


# In[8]:


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# In[9]:


PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
import win32com.client as wincl


# In[11]:

l=[]
import cv2
cap = cv2.VideoCapture(0)
i=0
from collections import Counter
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            import win32com.client as wincl
            speak = wincl.Dispatch("SAPI.SpVoice")
            import time
            ret, image_np = cap.read()
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np,axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: np.expand_dims(image_np, 0)})
      # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
             
            #for i in classes:
              #  speak.Speak(i)
            #speak.Speak([category_index.get(value)['name'] for index,value in enumerate(output_dict['detection_classes'][0]) if output_dict['detection_scores'][0,index] > 0.5])
        
            
            cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
            now = time.localtime(time.time())
            if(i%20 == 0):
                #g=np.argmax(np.bincount(np.squeeze(classes).astype(np.int32)))
                classes=classes.reshape(classes.shape[1])
                scores=scores.reshape(scores.shape[1])
                #t=np.argsort(scores)[-2:][::-1]
                d=classes[0]
                #e=classes[t[1]]
                e=classes[1]
                if(scores[1]>=0.5 and classes[1]!=classes[0]):
                    speak.Speak(str(category_index[d]['name'])+"and"+str(category_index[e]['name']))
                else:
                    speak.Speak(category_index[d]['name'])
                
                #g=max(list(classes),key=list(classes).count)
                #cnt=Counter(list(classes))
                #a=cnt.most_common(2)
                #g=a[0][0]
                #h=a[1][0]
                #k=np.squeeze(classes)[0]
                #if(i%60!=0):
                 #   l.append(np.squeeze(classes)[0])
                  #  if((len(l)==2 and l[0]==l[1]) or (len(l)==3 and l[1]==l[2])):
                   #     if l[0]==1:
                   #         speak.Speak(category_index[np.squeeze(classes)[1]]['name'])
                    #    else:
                    #        speak.Speak(category_index[g]['name'])
                    #else:
                    #    speak.Speak(category_index[k]['name'])
                            
                #else:
                 #   l=[]
                    #if(np.squeeze(classes)[0]==l[len(l)-1]):
                     #   if(k==1):
                      #      speak.Speak(category_index[np.squeeze(classes)[1]]['name'])
                      #  else:
                      #      speak.Speak(category_index[g]['name'])
                    #else:
                    #l.append(np.squeeze(classes)[0])
                    #speak.Speak(category_index[k]['name'])
                      
                            
                    
                   
                    


                #if(i%40==0):
                 #   speak.Speak(category_index[k]['name'])
                #else:
                 #   speak.Speak(category_index[h]['name'])
                print(classes)
                print(np.squeeze(scores))
                
            #print(classes.shape)
        
            #speak = wincl.Dispatch("SAPI.SpVoice")
            #speak.Speak()
            #g=np.argmax(np.bincount(np.squeeze(classes).astype(np.int32)))
            #speak.Speak(category_index[g]['name'])
            i+=1
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            
                
            #print(classes)                        #each
            
        #cap.release()
        #cv2.destroyAllWindows()


# ###### print(max(classes[0]))
