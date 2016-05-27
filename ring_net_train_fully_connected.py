
import os.path

import numpy as np
import tensorflow as tf
#import matplotlib
#matplotlib.use("Agg")
#import matplotlib.pyplot as plt 
#import matplotlib.animation as animation 

import cannon as cn

import ring_net

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/home/hennigho/git_things/Ring_Net/fully_connected_ring_train_store',
                            """dir to store trained net""")
tf.app.flags.DEFINE_float('t_weight', '1.0',
                          """the weight or the t error""")
tf.app.flags.DEFINE_float('m_weight', '1.0',
                          """the weight or the m error""")
tf.app.flags.DEFINE_float('b_weight', '1.0',
                          """the weight or the b error""")
tf.app.flags.DEFINE_float('xi_weight', '1.0',
                          """the weight or the xi error""")
tf.app.flags.DEFINE_float('yi_weight', '1.0',
                          """the weight or the yi error""")

CURRICULUM_STEPS = [10001, 10001, 10001, 10001]
CURRICULUM_SEQ = [2, 3, 4, 5]
CURRICULUM_BATCH_SIZE = [50, 30, 20, 20]


def train(iteration):
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # make dynamic system
    k = cn.Cannon()
    # init x 
    x = tf.placeholder(tf.float32, [CURRICULUM_BATCH_SIZE[iteration], CURRICULUM_SEQ[iteration], 28, 28, 4])
    keep_prob = tf.placeholder("float")

    # possible input dropout 
    input_keep_prob = tf.placeholder("float")
    x_drop = tf.nn.dropout(x, input_keep_prob)

    # make a list for collecting T' mappings
    output_t = []

    # make list for collecting g mappings 
    output_g = [] 

    # make list for collecting f outputs
    output_f = []

    # first I will run once to create the graph and then set reuse to true so there is weight sharing when I roll out t
    # do f
    y_0 = ring_net.encoding(x_drop[:, 0, :, :, :],keep_prob) 
    # do g
    x_0 = ring_net.decoding(y_0) 
    # do T' 
    y_1 = ring_net.fully_connected_compression(y_0, keep_prob) 
    # set weight sharing   
    tf.get_variable_scope().reuse_variables()
 
    # append these to the lists (I dont need output f. there will be seq_length elements of output_g and seq_length-1 of output_t and output_f)
    output_g.append(x_0)
    output_t.append(y_1)

    # loop throught the seq
    for i in xrange(CURRICULUM_SEQ[iteration] - 1):
      # calc f for all in seq 
      y_f_i = ring_net.encoding(x_drop[:, i+1, :, :, :],keep_prob)
      output_f.append(y_f_i)
      # calc g for all in seq
      x_g_i = ring_net.decoding(y_1) 
      output_g.append(x_g_i)
      # calc t for all in seq
      if i != (CURRICULUM_SEQ[iteration] - 2):
        y_1 = ring_net.fully_connected_compression(y_1,keep_prob)
        output_t.append(y_1)
      
    # calc error from (output_t - output_f)
    output_f = tf.pack(output_f)
    output_t = tf.pack(output_t)
    error_tf = tf.mul(50.0, ring_net.loss(output_f, output_t)) # scaling by 50 right now but this will depend on what network I am training. requires further investigation

    # calc error from (x - output_g)
    output_g = tf.pack(output_g)
    output_g = tf.transpose(output_g, perm=[1,0,2,3,4]) # this will make it look like x (I should check to see if transpose is not flipping or doing anything funny)
    error_xg = ring_net.loss(output_g, x)

    # add up errors
    error = tf.add_n([error_tf, error_xg])

    # train hopefuly 
    train_op = ring_net.train(error, 5e-4)

    # List of all Variables
    variables = tf.all_variables()

    # Build a saver
    saver = tf.train.Saver(tf.all_variables())   
    for i, variable in enumerate(variables):
	print '----------------------------------------------'
	print variable.name[:variable.name.index(':')]
 
    # Build an initialization operation to run below.
    if iteration == 0:
      init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session()

    # init if this is the very time training
    if iteration == 0: 
      sess.run(init)
 
    # restore if iteration is not 0
    if iteration != 0:
      variables_to_restore = tf.all_variables()
      saver = tf.train.Saver(variables_to_restore)
      ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("restored file from " + ckpt.model_checkpoint_path)
      else:
        print("no chekcpoint file found, this is an error")

    for step in xrange(CURRICULUM_STEPS[iteration]):
     # x_batch = k.generate_28x28x4(FLAGS.batch_size,FLAGS.seq_length)
      x_batch = k.generate_28x28x4(CURRICULUM_BATCH_SIZE[iteration],CURRICULUM_SEQ[iteration])
      _ , loss_value, loss_tf, loss_xg = sess.run([train_op, error, error_tf, error_xg],feed_dict={x:x_batch, keep_prob:.8, input_keep_prob:0.85})
      print(loss_value)
      print(loss_tf, loss_xg)

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step%1000 == 0:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("Saved")

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  for i in xrange(len(CURRICULUM_STEPS)):
    train(i)

if __name__ == '__main__':
  tf.app.run()
