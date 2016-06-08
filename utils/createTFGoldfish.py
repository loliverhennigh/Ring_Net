

import numpy as np 
import tensorflow as tf 
import cv2 
# name of goldfish data 
cap = cv2.VideoCapture("test.webm") 
# number of frames per data point 
FRAMES_NUM = 4 
# resize shape 
SHAPE = (78,78)

# create tf writer
writer = tf.python_io.TFRecordWriter("goldfish.tfrecords")

# the stored frames
frames = np.zeros((SHAPE[0], SHAPE[1], FRAMES_NUM * 3))

# helper function
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

while(True):
  # create frames
  for i in xrange(FRAMES_NUM):
    ret, frame = cap.read()
    if not ret:
      print("starting over")
      cap.release()
      cap = cv2.VideoCapture("test.webm")
      ret, frame = cap.read()
      print(ret)
    frame = cv2.resize(frame, SHAPE, interpolation = cv2.INTER_CUBIC)
    frames[:, :, i:i+3] = frame 
  # process frame for saving
  frames_flat = frames.reshape([1,SHAPE[0]*SHAPE[1]*FRAMES_NUM*3])
    
   
  # Capture frame-by-frame
  ret, frame = cap.read()
  print(frame.shape)

  # Our operations on the frame come here
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Display the resulting frame
  cv2.imshow('frame',gray)
  if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

