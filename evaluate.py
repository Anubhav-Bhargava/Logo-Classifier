import numpy as np
import keras
import argparse
import cv2
from keras.models import load_model
import matplotlib.pyplot as plt
from scipy.misc import *
import time

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())


def crop_image(img_path,x1,y1,x2,y2):
    img=imread(img_path)
    image=cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        
    h=y2-y1
    w=x2-x1
    crop_img = img[y1:y1+h, x1:x1+w]
    
    return crop_img

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])



sample_path=args["image"]
img=imread(sample_path)

model = load_model('models/my_model.h5')


(winW, winH) = (40, 40)

i=0
for (x, y, window) in sliding_window(img, stepSize=32, windowSize=(winW, winH)):
    # if the window does not meet our desired window size, ignore it
    if window.shape[0] != winH or window.shape[1] != winW:
        continue
 
   
    crop_img=crop_image(sample_path,x, y, x + winW, y + winH)
    crop_img=imresize(crop_img,(16,16))
    prediction=model.predict_classes(crop_img.reshape(1,16,16,3))
    if prediction==1:
    	pred='Pepsi'
    else:
    	pred='Not Pepsi'
    
   	
    clone = img.copy()
    cv2.putText(clone, pred, (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
    clone = cv2.cvtColor(clone,cv2.COLOR_BGR2RGB)
    cv2.imshow("Window", clone)
    #plt.imshow(clone)
    #plt.show()

    #filename = "/home/loop/Desktop/images/file_%d.jpg"%i
    #cv2.imwrite(filename, clone)
    
    i=i+1
    cv2.waitKey(1)
    time.sleep(0.5)


time.sleep(1.0)
cv2.destroyAllWindows()

