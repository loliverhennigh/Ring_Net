

import numpy as np 
import tensorflow as tf 
import cv2 
from glob import glob as glb

FLAGS = tf.app.flags.FLAGS

# helper function
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_converted_frame(cap, shape, color):
  ret, frame = cap.read()
  frame = cv2.resize(frame, shape, interpolation = cv2.INTER_CUBIC)
  if color:
    return frame
  else:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def generate_tfrecords(video_file, seq_length, shape, frame_num, color):
  # make video cap
  cap = cv2.VideoCapture(video_file) 

  # calc number of frames in video
  total_num_frames =float(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
  
  # create tf writer
  video_file_name = video_file.split('/')[-1]
  record_filename = '../data/tfrecords/' + FLAGS.video_dir + '/' + video_file_name.replace('.', '_') + '_seq_' + str(seq_length) + '_size_' + str(shape[0]) + 'x' + str(shape[1]) + 'x' + str(frame_num) + '_color_' + str(color) + '.tfrecords'
 
  # check to see if file alreay exists 
  tfrecord_filename = glb('../data/tfrecords/'+FLAGS.video_dir+'/*')
  if record_filename in tfrecord_filename:
    print('already a tfrecord there! I will skip this one')
    return 
 
  writer = tf.python_io.TFRecordWriter(record_filename)

  # the stored frames
  if color:
    frames = np.zeros((shape[0], shape[1], frame_num*3))
    seq_frames = np.zeros((seq_length, shape[0], shape[1], frame_num*3))
  else:
    frames = np.zeros((shape[0], shape[1], frame_num))
    seq_frames = np.zeros((seq_length, shape[0], shape[1], frame_num))

  # num frames
  ind = 0

  # end of file
  end = False 
  
  print('now generating tfrecords for ' + video_file + ' and saving to ' + record_filename)

  while(not end):
    # create frames
    for s in xrange(seq_length):
      if ind == 0:
        for i in xrange(frame_num):
          if color:
            frames[:,:,i*3:(i+1)*3] = get_converted_frame(cap, shape, color)
          else:
            frames[:,:,i] = get_converted_frame(cap, shape, color)
      else:
        if color:
          frames[:,:,0:frame_num*3-3] = frames[:,:,3:frame_num*3]
          frames[:,:,(frame_num-1)*3:frame_num*3] = get_converted_frame(cap, shape, color)
        else:
          frames[:,:,0:frame_num-1] = frames[:,:,1:frame_num]
          frames[:,:,frame_num-1] = get_converted_frame(cap, shape, color)
      seq_frames[s, :, :, :] = frames[:,:,:]

    #print(seq_frames.shape)

    # process frame for saving
    seq_frames = np.uint8(seq_frames)
    if color:
      seq_frames_flat = seq_frames.reshape([1,seq_length*shape[0]*shape[1]*frame_num*3])
    else:
      seq_frames_flat = seq_frames.reshape([1,seq_length*shape[0]*shape[1]*frame_num])
    
    seq_frame_raw = seq_frames_flat.tostring()
    # create example and write it
    example = tf.train.Example(features=tf.train.Features(feature={
      'image': _bytes_feature(seq_frame_raw)})) 
    writer.write(example.SerializeToString()) 

    # Display the resulting frame
    #cv2.imshow('frame',seq_frames[0,:,:,0:3])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
    # print status
    ind = ind + 1
    if ind%10000 == 0:
      print('percent converted = ' + str(100.0 * float(ind*seq_length) / float(total_num_frames)))

  # When everything done, release the capture
  cap.release()
  cv2.destroyAllWindows()

