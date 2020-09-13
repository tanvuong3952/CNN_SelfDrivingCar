import serial
import time

import cv2
import numpy as np
import logging
import math
#import tensorflow as tf
import keras
from keras.models import Sequential  # V2 is tensorflow.keras.xxxx, V1 is keras.xxx
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.models import load_model


def move(angle):
    if (0 <= angle <= 180):
        ser.write(bytes([angle]))
    else:
        print("Servo angle must be an integer between 0 and 180.\n")

# Start the serial port to communicate with arduino
ser = serial.Serial('COM4', 115200, timeout=1)


def img_preprocess(image,steering_angle):
    height, _, _ = image.shape
    abc = display_heading_line(image, steering_angle)
    cv2.imshow("anh", abc)
    #cv2.imshow("anh", image)
    image = image[int(height/2):,:,:]  # remove top half of the image, as it is not relevant for lane following
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)  # Nvidia model said it is best to use YUV color space
    image = cv2.GaussianBlur(image, (3,3), 0)
    image = cv2.resize(image, (200,66)) # input image size (200,66) Nvidia model
    image = image / 255
    return image



def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5, ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    # figure out the heading line from steering angle
    # heading line (x1,y1) is always center bottom of the screen
    # (x2, y2) requires a bit of trigonometry

    # Note: the steering angle of:
    # 0-89 degree: turn left
    # 90 degree: going straight
    # 91-180 degree: turn right 
    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)
    return heading_image



############################
# Test Functions
############################

steering_angle=90
#image_check = cv2.imread("test_image16.png")
#cap = cv2.VideoCapture('output_tv15.avi')
cap = cv2.VideoCapture(0)
model = load_model('my_CNN_model.h5')
try:
    while(True):
        start = time.time()
        ret, img = cap.read()
        #img = cv2.resize(img, (320,240))
        preprocessed = img_preprocess(img,steering_angle)
        #cv2.imshow("anh", preprocessed)
        X = np.asarray([preprocessed])
        steering_angle = model.predict(X)[0]
        #logging.info("model=%3d" % (steering_angle))
        print('new steering angle: %s' % steering_angle)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
        end=time.time()
        seconds = end - start
        fps = 1 / seconds
        print("Frames per second : {0}".format(fps))
        try:
            move(int(steering_angle))
            print("Goc lai = " + steering_angle)
        except:
            print("Improper input")
            
    cap.release()
    cv2.destroyAllWindows()
#cv2.waitKey(0)
    
    
except KeyboardInterrupt:
   GPIO.cleanup()
