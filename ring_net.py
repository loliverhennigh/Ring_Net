
"""Builds the ring network.

Summary of available functions:

  # Compute pics of the simulation runnig.
  
  # Create a graph to train on.
"""

import ring_net_input

import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS


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

def _conv_layer(inputs, kernel_size, stride, num_features, idx):
  with tf.variable_scope('{0}_conv'.format(idx)) as scope:
    input_channels = inputs.get_shape()[3]

    weights = _variable_with_weight_decay('weights', shape=[kernel_size,kernel_size,input_channels,num_features],stddev=0.1, wd=FLAGS.weight_decay)
    biases = _variable_on_cpu('biases',[num_features],tf.constant_initializer(0.1))

    conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME')
    conv_biased = tf.nn.bias_add(conv, biases)
    #Leaky ReLU
    conv_rect = tf.maximum(FLAGS.alpha*conv_biased,conv_biased,name='{0}_conv'.format(idx))
    return conv_rect

def _transpose_conv_layer(inputs, kernel_size, stride, num_features, idx):
  with tf.variable_scope('{0}_trans_conv'.format(idx)) as scope:
    input_channels = inputs.get_shape()[3]
    
    weights = _variable_with_weight_decay('weights', shape=[kernel_size,kernel_size,num_features,input_channels], stddev=0.1, wd=FLAGS.weight_decay)
    biases = _variable_on_cpu('biases',[num_features],tf.constant_initializer(0.1))
    batch_size = tf.shape(inputs)[0]
    output_shape = tf.pack([tf.shape(inputs)[0], tf.shape(inputs)[1]*stride, tf.shape(inputs)[2]*stride, num_features]) 
    conv = tf.nn.conv2d_transpose(inputs, weights, output_shape, strides=[1,stride,stride,1], padding='SAME')
    conv_biased = tf.nn.bias_add(conv, biases)
    #Leaky ReLU
    conv_rect = tf.maximum(FLAGS.alpha*conv_biased,conv_biased,name='{0}_transpose_conv'.format(idx))
    return conv_rect
     

def _fc_layer(inputs, hiddens, idx, flat = False, linear = False):
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

def encoding(inputs, keep_prob):
  """Builds encoding part of ring net.
  Args:
    inputs: input to encoder
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice x_1 -> y_1
  x_1_image = inputs 
 
  # normalize and formate the first layer
  #keep_prob = tf.placeholder("float") # do a little dropout to normalize
  #x_1_image = tf.reshape(x_1, [-1, 28, 28, 1])
  # conv1
  conv1 = _conv_layer(x_1_image, 5, 1, 32, 1)
  # conv2
  conv2 = _conv_layer(conv1, 2, 2, 32, 2)
  # conv3
  conv3 = _conv_layer(conv2, 5, 1, 64, 3)
  # conv4
  conv4 = _conv_layer(conv3, 2, 2, 64, 4)
  # fc5 
  fc5 = _fc_layer(conv4, 512, 5, True, False)
  # dropout maybe
  fc5_dropout = tf.nn.dropout(fc5, keep_prob)
  # y_1 
  y_1 = _fc_layer(fc5_dropout, 64, 6, False, False)

  return y_1 

def markov_encoding(inputs, keep_prob):
  """Builds encoding part of ring net.
  Args:
    inputs: input to encoder
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice x_1 -> y_1
  x_1_image = inputs 
 
  # normalize and formate the first layer
  #keep_prob = tf.placeholder("float") # do a little dropout to normalize
  #x_1_image = tf.reshape(x_1, [-1, 28, 28, 1])
  # conv1
  conv1 = _conv_layer(x_1_image, 5, 1, 32, 1)
  # conv2
  conv2 = _conv_layer(conv1, 2, 2, 32, 2)
  # conv3
  conv3 = _conv_layer(conv2, 5, 1, 64, 3)
  # conv4
  conv4 = _conv_layer(conv3, 2, 2, 64, 4)
  # fc5 
  fc5 = _fc_layer(conv4, 512, 5, True, False)
  # dropout maybe
  fc5_dropout = tf.nn.dropout(fc5, keep_prob)
  # y_1 
  y_1 = tf.nn.softmax(_fc_layer(fc5_dropout, 64, 6, False, True))

  return y_1 


def fully_connected_compression(inputs, keep_prob):
  """Builds compressed dynamical system part of the net.
  Args:
    inputs: input to system
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_1 -> y_2
  y_1 = inputs 
 
  # (start indexing at 10) -- I will change this in a bit
  # fc11
  fc11 = _fc_layer(y_1, 512, 11, False, False)
  # fc12
  fc12 = _fc_layer(fc11, 512, 12, False, False)
  # dropout maybe
  fc12_dropout = tf.nn.dropout(fc12, keep_prob)
  # y_2 
  y_2 = _fc_layer(fc12_dropout, 64, 13, False, False)

  return y_2 

def markov_compression(inputs):
  """Builds compressed dynamical system part of the net.
  Args:
    inputs: input to system
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_1 -> y_2
  y_1 = inputs 
 
  # y_2 
  y_2 = tf.nn.softmax(_fc_layer(y_1, 64, 13, False, True))

  return y_2 

def decoding(inputs):
  """Builds decoding part of ring net.
  Args:
    inputs: input to decoder
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_2 -> x_2
  y_2 = inputs 
 
  # fc21
  fc21 = _fc_layer(y_2, 512, 21, False, False)
  # fc23
  fc22 = _fc_layer(fc21, 64*7*7, 22, False, False)
  conv22 = tf.reshape(fc22, [-1, 7, 7, 64])
  # conv23
  conv23 = _transpose_conv_layer(conv22, 2, 2, 64, 23)
  # conv24
  conv24 = _transpose_conv_layer(conv23, 5, 1, 32, 24)
  # conv25
  conv25 = _transpose_conv_layer(conv24, 2, 2, 32, 25)
  # conv26
  conv26 = _transpose_conv_layer(conv25, 5, 1, 4, 26)
  # x_2 
  x_2 = tf.reshape(conv26, [-1, 28, 28, 4])

  return x_2 

def loss(output, correct_output):
  error = tf.nn.l2_loss(output - correct_output)
  return error

def markov_loss(output, correct_output):
  index_max = tf.argmax(correct_output, 1)
  correct_output_one_hot = tf.one_hot(index_max, 64, 1.0, 0.0)
  error = tf.reduce_mean(-tf.reduce_sum(correct_output_one_hot * tf.log(correct_output), reduction_indices=[1]))
  return error
  
  
def train(total_loss, lr):
   train_op = tf.train.AdamOptimizer(lr).minimize(total_loss)
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
