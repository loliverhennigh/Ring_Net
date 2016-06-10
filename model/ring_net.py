
"""Builds the ring network.

Summary of available functions:

  # Compute pics of the simulation runnig.
  
  # Create a graph to train on.
"""

#import ring_net_input

import tensorflow as tf
import numpy as np
import architecture
import unwrap_helper

FLAGS = tf.app.flags.FLAGS


# Constants describing the training process.
tf.app.flags.DEFINE_string('model', 'fully_connected_28x28x4',
                           """ model name to train """)
tf.app.flags.DEFINE_string('system', 'video',
                           """ system to compress """)
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
# possible models to train are
# markov_28x28x4
# fully_connected_28x28x4

def inputs(batch_size, seq_length):
  """makes input vector
  Return:
    x: input vector, may be filled 
  """
  if FLAGS.model == "fully_connected_28x28x4" or FLAGS.model == "markov_28x28x4":
    x = tf.placeholder(tf.float32, [batch_size, seq_length, 28, 28, 4])
  return x

def encoding(inputs, keep_prob):
  """Builds encoding part of ring net.
  Args:
    inputs: input to encoder
    keep_prob: dropout layer
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice x_1 -> y_1
  if FLAGS.model == "fully_connected_28x28x4": 
    y_1 = architecture.encoding_28x28x4(inputs, keep_prob)
  elif FLAGS.model == "markov_28x28x4": 
    y_1 = architecture.markov_encoding_28x28x4(inputs, keep_prob)

  return y_1 

def compression(inputs, keep_prob):
  """Builds compressed dynamical system part of the net.
  Args:
    inputs: input to system
    keep_prob: dropout layer
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_1 -> y_2
  if FLAGS.model == "fully_connected_28x28x4": 
    y_2 = architecture.compression_28x28x4(inputs, keep_prob)
  elif FLAGS.model == "markov_28x28x4": 
    y_2 = architecture.markov_compression_28x28x4(inputs)

  return y_2 

def decoding(inputs):
  """Builds decoding part of ring net.
  Args:
    inputs: input to decoder
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_2 -> x_2
  if FLAGS.model == "fully_connected_28x28x4": 
    x_2 = architecture.decoding_28x28x4(inputs)
  elif FLAGS.model == "markov_28x28x4": 
    x_2 = architecture.decoding_28x28x4(inputs)

  return x_2 

def unwrap(inputs, keep_prob, seq_length):
  """Unrap the system for training.
  Args:
    inputs: input to system, should be [minibatch, seq_length, image_size]
    keep_prob: dropout layers
    seq_length: how far to unravel 
 
  Return: 
    output_t: calculated y values from iterating t'
    output_g: calculated x values from g
    output_f: calculated y values from f 
  """

  if FLAGS.model == "fully_connected_28x28x4": 
    output_t, output_g, output_f = unwrap_helper.fully_connected_unwrap(inputs, keep_prob, seq_length)
  elif FLAGS.model == "markov_28x28x4": 
    output_t, output_g, output_f = unwrap_helper.markov_unwrap(inputs, keep_prob, seq_length)

  return output_t, output_g, output_f 

def loss(inputs, output_t, output_g, output_f):
  """Calc loss for unrap output.
  Args.
    inputs: true x values
    output_t: calculated y values from iterating t'
    output_g: calculated x values from g
    output_f: calculated y values from f 

  Return:
    error: loss value
  """
  if FLAGS.model == "fully_connected_28x28x4": 
    error_tf = tf.mul(50.0, tf.nn.l2_loss(output_f - output_t)) # scaling by 50 right now but this will depend on what network I am training. requires further investigation
    error_xg = tf.nn.l2_loss(output_g - inputs)
    tf.scalar_summary('error_tf', error_tf)
    tf.scalar_summary('error_xg', error_xg)
    error = tf.cond(error_tf > error_xg, lambda: error_tf, lambda: error_xg)
  elif FLAGS.model == "markov_28x28x4": 
    error_tf = tf.mul(500.0, cross_entropy_loss(output_t, output_f))
    error_ft = tf.mul(500.0, cross_entropy_loss(output_f, output_t))
    error_xg = tf.nn.l2_loss(output_g - inputs)
    tf.scalar_summary('error_tf', error_tf)
    tf.scalar_summary('error_ft', error_ft)
    tf.scalar_summary('error_xg', error_xg)
    error = tf.add_n([error_tf, error_ft, error_xg])
  tf.scalar_summary('error', error)
  return error

def l2_loss(output, correct_output):
  """Calcs the loss for the model"""
  error = tf.nn.l2_loss(output - correct_output)
  return error

def cross_entropy_loss(output, correct_output):
  """ cross entropy loss by converting correcte_output to a one hot vector"""
  correct_output_one_hot = one_hot(correct_output)
  error = tf.reduce_mean(-tf.reduce_sum(correct_output_one_hot * tf.log(correct_output), reduction_indices=[1]))
  return error

def one_hot(inputs):
  batch_size = inputs.get_shape().as_list()[0]
  max_value = tf.reduce_max(inputs, reduction_indices=[1])
  inv_max_value = tf.div(1.0, max_value)
  inv_max_value = tf.expand_dims(inv_max_value, 1)
  inputs = tf.mul(inv_max_value, inputs)
  inputs = tf.pow(inputs, 10)
  return inputs
 
def train(total_loss, lr):
   train_op = tf.train.AdamOptimizer(lr).minimize(total_loss)
   return train_op

