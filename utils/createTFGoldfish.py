
import numpy as np
import tensorflow as tf
import cv2

# name of goldfish data
cap = cv2.VideoCapture("goldfish.webm")

# number of frames per data point
FRAMES_NUM = 4

# create tf writer
writer = tf.python_io.TFRecordWriter("goldfish.tfrecords")

# the stored frames
frames = []

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

for i in xrange(FRAMES_NUM):
  ret, frame = cap.read()
  frames.append(frame)
   

while(True):

    

    # Capture frame-by-frame

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

