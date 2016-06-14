
import os
import numpy as np
import tensorflow as tf
import utils.createTFRecords as createTFRecords
from glob import glob as glb


FLAGS = tf.app.flags.FLAGS

# Constants describing the training process.
tf.app.flags.DEFINE_string('video_dir', 'goldfish',
                           """ dir containing the video files """)

def video_inputs(batch_size, seq_length):
  """Construct input for ring net. given a video_dir that contains videos this will check to see if there already exists tf recods and makes them. Then returns batchs
  Args:
    batch_size: Number of images per batch.
    seq_length: seq of inputs.
  Returns:
    images: Images. 4D tensor. Possible of size [batch_size, 78x78x4].
  """

  video_filename = glb('../data/videos/'+FLAGS.video_dir+'/*') 

  for f in video_filename:
    if model == "fully_connected_84x84x4":
      generateTFRecords.generate_tfrecords(f, seq_length, (84,84), 4)
     

 
  return x_1_inputs[0]
