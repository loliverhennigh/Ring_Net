



import math

import numpy as np
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/ring_net_eval_store',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/rint_net_train_store',
                           """Directory where to read model checkpoints.""")

def eval_once(saver, input_test, y_1_loop, y_2_loop, x_2_loop):
  """Run Eval on input_test.
  Args:
    saver: Saver.
    summary_writer: input.
    y_1_loop: first y_1 
    y_2_loop: second y_2 
    x_2_loop: generated image 
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return
    ims_generated = []
    y_2_approx = y_2_loop.eval(session = sess, feed_dict={x_1=input_test[0,0:1,:,:,:]})
    for step in xrange(input_test.shape[1]):
      # calc image from y_2
      im.append(x_2_m.eval(session=sess, feed_dict={y_2_m=y_2_approx}))
      # use first step to calc next
      y_2_approx = y_2_loop.eval(session=sess, feed_dict={y_2_m=y_2_approx})
    return ims_generated

def evaluate():
  """ Eval the system"""
  with tf.Graph().as_default():
    # make dynamic system
    k = cn.Cannon()
    # init in and out
    x_1 = tf.placeholder(tf.float32, [None, 28, 28, 4])

    # encoding
    y_1_m = ring_net.encoding(x_1_norm)
    
    # dynamic system
    y_2_m = ring_net.dynamic_compression(y_1_m)
 
    # decoding 
    x_2_m = ring_net.decoding(y_2_m)
  
    eval_once(saver


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
