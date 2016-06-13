

import numpy as np 
import tensorflow as tf 
import cv2 

# helper function
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def generate_tfrecords(video_file, seq_length, shape, frame_num)
  # make video cap
  cap = cv2.VideoCapture(video_file) 

  # create tf writer
  record_filename = video_file.replace('.', '_') + '_seq_' + str(seq_length) + '_size_' + str(shape[0]) + 'x' + str(shape[1]) + 'x' + str(frame_num) + '.tfrecords')
 
  # check to see if file alreay exists 
  tfrecord_filename = glb('../data/tfRecords/'+FLAGS.video_dir+'/*') 
  if record_filename in tfrecord_filename:
    print('already a tfrecord there! I will skip this one')
    return 
 
  writer = tf.python_io.TFRecordWriter(record_filename)

  # the stored frames
  frames = np.zeros((shape[0], shape[1], frame_num))

  # num frames
  ind = 0

  # end of file
  end = False 
  
  print('now generating tfrecords for ' + video_file)
  while(not end):
    # create frames
    for i in xrange(frame_num):
      ret, frame = cap.read()
      if not ret:
        end = True 
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      frame = cv2.resize(frame, shape, interpolation = cv2.INTER_CUBIC)
      frames[:, :, i] = frame/255.0
    # process frame for saving
    frames_flat = frames.reshape([1,shape[0]*shape[1]*frames_num])
    frame_raw = frames_flat.tostring()
    # create example and write it
    example = tf.train.Example(features=tf.train.Features(feature={
      'image': _bytes_feature(frame_raw)})) 
    writer.write(example.SerializeToString()) 

    # Display the resulting frame
    cv2.imshow('frame',frames[:,:,0])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ind = ind + 1
    print(ind)

  # When everything done, release the capture
  cap.release()
  cv2.destroyAllWindows()

