
import os.path

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 

import cannon as cn

import ring_net

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/ring_train_store',
                            """dir to store trained net""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """number of batches to run""")


def train():
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # make dynamic system
    k = cn.Cannon()
    # init in and out
    x_0 = tf.placeholder(tf.float32, [None, 28, 28, 4])
    x_1 = tf.placeholder(tf.float32, [None, 28, 28, 4])
    x_2 = tf.placeholder(tf.float32, [None, 28, 28, 4])

    # possible input dropout 
    input_keep_prob = tf.placeholder("float")
    x_0_norm = tf.nn.dropout(x_1, input_keep_prob)
    x_1_norm = tf.nn.dropout(x_1, input_keep_prob)
    x_2_norm = tf.nn.dropout(x_1, input_keep_prob)
 
    # ok, because tensorflow is picky first I will define the graph for
    #  x_1_m and then I will set reuse = true and get the other needed 
    # values to train.

    # encoding
    y_1_m = ring_net.encoding(x_1_norm)
    
    # dynamic system
    y_2_m = ring_net.dynamic_compression(y_1_m)
 
    # decoding 
    x_2_m = ring_net.decoding(y_2_m)

    # set reuse to true
    tf.get_variable_scope().reuse_variables()

    # get the other needed values for training
    # 0 and 2 encoding
    y_0_t = ring_net.encoding(x_0_norm)
    y_2_b = ring_net.encoding(x_2_norm)
    # just need 0 dynamic_compression
    y_1_t = ring_net.dynamic_compression(y_0_t)

    # add up errors
    error_t = tf.mul(50.0, ring_net.loss(y_1_m, y_1_t))
    error_m = ring_net.loss(x_2, x_2_m)
    error_b = tf.mul(50.0, ring_net.loss(y_2_m, y_2_b))
    error = tf.add_n([error_t, error_m, error_b])
 
    """
    ########### x_1 time ###########
    # encoding
    y_1_m = ring_net.encoding(x_1_norm)

    # dynamic system
    y_2_m = ring_net.dynamic_compression(y_1_m)

    # decoding 
    x_2_m = ring_net.decoding(y_2_m)
    
    # set reuse to true
    tf.get_variable_scope().reuse_variables()

    ########### x_0 time ###########
    # encoding
    y_0_t = ring_net.encoding(x_0_norm)

    # dynamic system
    y_1_t = ring_net.dynamic_compression(y_0_t)
    y_2_t = ring_net.dynamic_compression(y_1_t)

    # decoding 
    x_2_t = ring_net.decoding(y_2_t)
 
    ########### x_2 time ###########
    # encoding
    y_2_b = ring_net.encoding(x_2_norm)

    # decoding 
    x_2_b = ring_net.decoding(y_2_b)

    # add up errors
    error_t = ring_net.loss(x_2, x_2_t)
    error_m = ring_net.loss(x_2, x_2_m)
    error_b = ring_net.loss(x_2, x_2_b)
    error = tf.add_n([error_t, error_m, error_b])
    """    

    # train hopefuly 
    train_op = ring_net.train(error)

    # List of all Variables
    variables = tf.all_variables()

    # Build a saver
    saver = tf.train.Saver(tf.all_variables())   
    for i, variable in enumerate(variables):
	print '----------------------------------------------'
	print variable.name[:variable.name.index(':')]
 
    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session()
    sess.run(init)

    for step in xrange(FLAGS.max_steps):
      x_0_true, x_1_true, x_2_true = k.generate_28x28(1,50)
      x_0_true, x_1_true, x_2_true = convert_1frame_to_4frame(x_0_true, x_1_true, x_2_true) 
      _ , loss_value = sess.run([train_op, error],feed_dict={x_0:x_0_true[0], x_1:x_1_true[0], x_2:x_2_true[0], input_keep_prob:.8})
      print(loss_value)

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step%1000 == 0:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("Saved")

def convert_1frame_to_4frame(x_0, x_1, x_2):
    x_0_new = np.zeros([x_0.shape[0], x_0.shape[1] - 4, x_0.shape[2], x_0.shape[3], 4])
    x_1_new = np.zeros([x_0.shape[0], x_0.shape[1] - 4, x_0.shape[2], x_0.shape[3], 4])
    x_2_new = np.zeros([x_0.shape[0], x_0.shape[1] - 4, x_0.shape[2], x_0.shape[3], 4])
    for i in xrange(x_0.shape[1]-4):
        for j in xrange(4):
            x_0_new[:, i, :, :, j]  = x_0[:, i+j, :, :]
            x_1_new[:, i, :, :, j]  = x_1[:, i+j, :, :]
            x_2_new[:, i, :, :, j]  = x_2[:, i+j, :, :]
       
    return x_0_new, x_1_new, x_2_new

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()
