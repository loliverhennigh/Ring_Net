
import os.path

import numpy as np
import tensorflow as tf
#import matplotlib
#matplotlib.use("Agg")
#import matplotlib.pyplot as plt 
#import matplotlib.animation as animation 

#import Ring_Net.systems.cannon as cn
import sys
sys.path.append('../')
import systems.cannon as cn
import model.ring_net as ring_net

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '../checkpoints/train_store_',
                            """dir to store trained net""")
CURRICULUM_STEPS = [100001, 100001]
CURRICULUM_SEQ = [2, 3]
CURRICULUM_BATCH_SIZE = [50, 50]


def train(iteration):
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # make dynamic system
    k = cn.Cannon()
    # make inputs
    x = ring_net.inputs(CURRICULUM_BATCH_SIZE[iteration], CURRICULUM_SEQ[iteration]) 
    # possible input dropout 
    input_keep_prob = tf.placeholder("float")
    x_drop = tf.nn.dropout(x, input_keep_prob)
    # possible dropout inside
    keep_prob = tf.placeholder("float")
    # create and unrap network
    output_t, output_g, output_f = ring_net.unwrap(x_drop, keep_prob, CURRICULUM_SEQ[iteration]) 
    # calc error
    error = ring_net.loss(x, output_t, output_g, output_f)
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
      ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir+FLAGS.model)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("restored file from " + ckpt.model_checkpoint_path)
      else:
        print("no chekcpoint file found, this is an error")

    for step in xrange(CURRICULUM_STEPS[iteration]):
      x_batch = k.generate_28x28x4(CURRICULUM_BATCH_SIZE[iteration],CURRICULUM_SEQ[iteration])
      _ , loss_value = sess.run([train_op, error],feed_dict={x:x_batch, keep_prob:1.0, input_keep_prob:1.0})
      print(loss_value)

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step%1000 == 0:
        checkpoint_path = os.path.join(FLAGS.train_dir + FLAGS.model, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("saved!")
        print(loss_value)

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir + FLAGS.model):
    tf.gfile.DeleteRecursively(FLAGS.train_dir + FLAGS.model)
  tf.gfile.MakeDirs(FLAGS.train_dir + FLAGS.model)
  for i in xrange(len(CURRICULUM_STEPS)):
    train(i)

if __name__ == '__main__':
  tf.app.run()
