
"""Builds the ring network.

Summary of available functions:

  # Compute pics of the simulation runnig.
  
  # Create a graph to train on.
"""

import ring_net_input

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 50,
                            """the batch size""")

# Constants describing the training process.
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,
                          """The decay to use for the moving average""")
tf.app.flags.DEFINE_float('momentum', 0.9,
                          """momentum of learning rate""")
tf.app.flags.DEFINE_float('alpha', 0.1,
                          """Leaky RElu param""")
tf.app.flags.DEFINE_float('weight_decay', 0.0005,
                          """ """)
tf.app.flags.DEFINE_float('dropout_hidden', 0.5,
                          """ dropout on hidden """)
tf.app.flags.DEFINE_float('dropout_input', 0.8,
                          """ dropout on input """)

def inputs():
  return ring_net_input.inputs(FLAGS.batch_size)


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    weight_decay.set_shape([])
    tf.add_to_collection('losses', weight_decay)
  return var


def inference(images):
  """Build the ring network for 28x28 dynamical systems.
  Args:
    inputs: input not sure yet
  """
  def conv_layer(inputs, kernel_size, stride, num_features, idx):
    with tf.variable_scope('{0}_conv'.format(idx)) as scope:
      input_channels = inputs.get_shape()[3]

      weights = _variable_with_weight_decay('weights', shape=[kernel_size,kernel_size,input_channels,num_features],stddev=0.1, wd=FLAGS.weight_decay)
      biases = _variable_on_cpu('biases',[num_features],tf.constant_initializer(0.1))

      conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME')
      conv_biased = tf.nn.bias_add(conv, biases)
      #Leaky ReLU
      conv_rect = tf.maximum(FLAGS.alpha*conv_biased,conv_biased,name='{0}_conv'.format(idx))
      return conv_rect
  
  def transpose_conv_layer(inputs, kernel_size, stride, num_features, idx):
    with tf.variable_scope('{0}_trans_conv'.format(idx)) as scope:
      input_channels = inputs.get_shape()[3]
      
      weights = _variable_with_weight_decay('weights', shape=[kernel_size,kernel_size,num_features,input_channels], stddev=0.1, wd=FLAGS.weight_decay)
      biases = _variable_on_cpu('biases',[num_features],tf.constant_initializer(0.1))
      batch_size = tf.shape(x_1_image)[0]
      output_shape = tf.pack([tf.shape(inputs)[0], tf.shape(inputs)[1]*stride, tf.shape(inputs)[2]*stride, num_features]) 
      conv = tf.nn.conv2d_transpose(inputs, weights, output_shape, strides=[1,stride,stride,1], padding='SAME')
      conv_biased = tf.nn.bias_add(conv, biases)
      #Leaky ReLU
      conv_rect = tf.maximum(FLAGS.alpha*conv_biased,conv_biased,name='{0}_transpose_conv'.format(idx))
      return conv_rect
       
  
  def fc_layer(inputs, hiddens, idx, flat = False, linear = False):
    with tf.variable_scope('fc{0}'.format(idx)) as scope:
      input_shape = inputs.get_shape().as_list()
      if flat:
        dim = input_shape[1]*input_shape[2]*input_shape[3]
        inputs_processed = tf.reshape(inputs, [-1,dim])
      else:
        dim = input_shape[1]
        inputs_processed = inputs
      
      weights = _variable_with_weight_decay('weights', shape=[dim,hiddens],stddev=0.01, wd=FLAGS.weight_decay)
      biases = _variable_on_cpu('biases', [hiddens], tf.constant_initializer(0.01))
      if linear:
        return tf.add(tf.matmul(inputs_processed,weights),biases,name=str(idx)+'_fc')
    
      ip = tf.add(tf.matmul(inputs_processed,weights),biases)
      return tf.maximum(FLAGS.alpha*ip,ip,name=str(idx)+'_fc')

  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  x_1 = images
  #x_1 = tf.placeholder(tf.float32, [None, 784])
  #x_2 = tf.placeholder(tf.float32, [None, 784])
  #y_1 = tf.placeholder(tf.float32, [None, 64])
  #y_2 = tf.placeholder(tf.float32, [None, 64])
 
  # normalize and formate the first layer
  #keep_prob = tf.placeholder("float") # do a little dropout to normalize
  x_1_norm = tf.nn.dropout(x_1, .08)
  x_1_image = tf.reshape(x_1_norm, [-1, 28, 28, 1])
  # Need the batch size for the transpose layers.
  batch_size = tf.shape(x_1_image)[0]
 
  # conv1
  conv1 = conv_layer(x_1_image, 5, 1, 32, 1)
  # conv2
  conv2 = conv_layer(conv1, 2, 2, 32, 2)
  # conv3
  conv3 = conv_layer(conv2, 5, 1, 64, 3)
  # conv4
  conv4 = conv_layer(conv3, 2, 2, 64, 4)
  # conv5
  conv5 = conv_layer(conv4, 1, 1, 5, 5)
  # conv6
  conv6 = transpose_conv_layer(conv5, 1, 1, 64, 6)
  # conv7
  conv7 = transpose_conv_layer(conv6, 2, 2, 64, 7)
  # conv8
  conv8 = transpose_conv_layer(conv7, 5, 1, 32, 8)
  # conv9 
  conv9 = transpose_conv_layer(conv8, 2, 2, 32, 9)
  # conv10 
  conv10 = transpose_conv_layer(conv9, 5, 1, 1, 10)
  # reshape
  
  _x_2 = tf.reshape(conv10, [-1, 784])
  
  #error = tf.nn.l2_loss(x_2 - _x_2)
  #train_step = tf.train.AdamOptimizer(1e-4).minimize(error) # I made the learning rate smaller then normal
  #accuracy = tf.nn.l2_loss(x_2 - _x_2)
  return _x_2

def loss(output, correct_output):
  error = tf.nn.l2_loss(x_2 - _x_2)
  return error
  
def train(total_loss):
   lr = 1e-4
   train_op = tf.trainAdamOptimizer(lr).minimize(total_loss)
   return train_op

'''
for i in range(20000):
  x_input, y_input = k.generate_28x28(1,50)
  if i%20 == 0:
    #train_accuracy = accuracy.eval(feed_dict={
    #    x:x_input[0], y_:x_input[0], keep_prob: 1.0})
    #print("step %d, training accuracy %g"%(i, train_accuracy))
    print("Saving test image to new_run_1.png")
    #new_im = y_conv.eval(feed_dict={x: x_input[0,10:11], y_: x_input[0,10:11], keep_prob: 1.0})
    #new_im = _x_2.eval(feed_dict={x: x_input[0,10:11], keep_prob: 1.0})
    #plt.imshow(new_im.reshape((28,28)))
    #plt.savefig('new_run_1.png')
    #print("Saved")
  train_step.run(feed_dict={x: x_input[0], y_: x_input[0], keep_prob: 0.8})

'''
