



import math

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 

import cannon as cn

import ring_net

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/home/hennigho/git_things/Ring_Net/ring_net_eval_store',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/hennigho/git_things/Ring_Net/markov_ring_train_store',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('video_name', 'new_video_1.mp4',
                           """name of the video you are saving""")

writer = animation.writers['ffmpeg'](fps=30)


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



def eval_once(saver, input_test, x_1_loop, y_1_loop, y_2_loop, x_2_loop, keep_prob):
  """Run Eval on input_test.
  Args:
    saver: Saver.
    summary_writer: input.
    y_1_loop: first y_1 
    y_2_loop: second y_2 
    x_2_loop: generated image 
  """
  with tf.Session() as sess:
    print FLAGS.checkpoint_dir
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
    fig = plt.figure()
    y_2_approx = y_2_loop.eval(session = sess, feed_dict={x_1_loop:input_test[0,0:1,:,:,:], keep_prob:1.0})
    x_2_full_1 = x_2_loop.eval(session = sess, feed_dict={x_1_loop:input_test[0,0:1,:,:,:], keep_prob:1.0})

    x_2_full_2 = x_2_loop.eval(session = sess, feed_dict={x_1_loop:input_test[0,0:1,:,:,:], keep_prob:1.0})
    
    for step in xrange(input_test.shape[1]-1):
      # calc image from y_2
      print(step)
      x_2_out = x_2_loop.eval(session=sess, feed_dict={y_2_loop:y_2_approx, keep_prob:1.0})
      
      #new_im = np.concatenate((x_2_full_2[:,:,:,0].reshape((28,28))/x_2_full_2.max(),x_2_out[:,:,:,0].reshape((28,28))/x_2_out.max(), x_2_full_1[:,:,:,0].reshape((28,28))/x_2_full_1.max(),input_test[0,step+1:step+2,:,:,0].reshape((28,28))/input_test.max()), axis=0)
      new_im = np.concatenate((x_2_out[:,:,:,0].reshape((28,28))/x_2_out.max(), input_test[0,step+1:step+2,:,:,0].reshape((28,28))/input_test.max()), axis=0)
      

      x_2_full_1 = x_2_loop.eval(session=sess, feed_dict={x_1_loop:x_2_full_1, keep_prob:1.0})

      # calc for f(g())
      y_1_fg = y_1_loop.eval(session=sess, feed_dict={x_1_loop:x_2_full_2, keep_prob:1.0})
      x_2_full_2 = x_2_loop.eval(session=sess, feed_dict={y_2_loop:y_1_fg, keep_prob:1.0})

      

      ims_generated.append((plt.imshow(new_im),))
      # use first step to calc next
      y_2_approx = y_2_loop.eval(session=sess, feed_dict={y_1_loop:y_2_approx, keep_prob:1.0})
    m_ani = animation.ArtistAnimation(fig, ims_generated, interval= 5000, repeat_delay=3000, blit=True)
    print(FLAGS.video_name)
    m_ani.save(FLAGS.video_name, writer=writer)
       

def evaluate():
  """ Eval the system"""
  with tf.Graph().as_default():
    # make dynamic system
    k = cn.Cannon()
    x_0_true, x_1_true, x_2_true = k.generate_28x28(1,500)
    x_0_true, x_1_true, x_2_true = convert_1frame_to_4frame(x_0_true, x_1_true, x_2_true) 

    # init in and out
    x_1 = tf.placeholder(tf.float32, [None, 28, 28, 4])
    keep_prob = tf.placeholder("float")

    # encoding
    y_1_m = ring_net.markov_encoding(x_1, keep_prob)
    
    # dynamic system
    y_2_m = ring_net.markov_compression(y_1_m)
 
    # decoding 
    x_2_m = ring_net.decoding(y_2_m)
 
    # Restore the moving average cersion of the learned variables for eval
    #variable_averages = tf.train.ExponentialMovingAverage(
    #    FLAGS.moving_average_decay)
    #variables_to_restore = variable_averages.variables_to_restore()
    variables_to_restore = tf.all_variables()
    saver = tf.train.Saver(variables_to_restore)

    # eval now
    eval_once(saver, x_1_true, x_1, y_1_m, y_2_m, x_2_m, keep_prob)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
