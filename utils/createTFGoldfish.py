

import numpy as np 
import tensorflow as tf 
import cv2 
# name of goldfish data 
video_file = "goldfish.webm"
cap = cv2.VideoCapture(video_file) 

# seq length
SEQ_LENGTH = 2

# number of frames per data point 
FRAMES_NUM = 4 

# resize shape 
SHAPE = (28,28)

# create tf writer
record_filename = "goldfish_28x28_" + str(SEQ_LENGTH) + ".tfrecords"
writer = tf.python_io.TFRecordWriter(record_filename)

# the stored frames
frames = np.zeros((SHAPE[0], SHAPE[1], FRAMES_NUM))

# helper function
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# num frames
ind = 0

# end of file
end = False 

while(not end):
  # create frames
  for i in xrange(FRAMES_NUM):
    ret, frame = cap.read()
    if not ret:
      end = True 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, SHAPE, interpolation = cv2.INTER_CUBIC)
    frames[:, :, i] = frame/255.0
  # process frame for saving
  frames_flat = frames.reshape([1,SHAPE[0]*SHAPE[1]*FRAMES_NUM])
  frame_raw = frames_flat.tostring()
  # create example and write it
  example = tf.train.Example(features=tf.train.Features(feature={
    'image': _bytes_feature(frame_raw)})) 
  writer.write(example.SerializeToString()) 

  # Display the resulting frame
  #cv2.imshow('frame',frames[:,:,0])
  #if cv2.waitKey(1) & 0xFF == ord('q'):
  #    break
  ind = ind + 1
  print(ind)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

