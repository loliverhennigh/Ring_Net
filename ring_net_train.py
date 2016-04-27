
import numpy as np
import tensorflow as tf

import cannon as cn

import ring_net

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """number of batches to run""")

def train():
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # Get images and labels for CIFAR-10.
    k = cn.Cannon()
    x_1 = tf.placeholder(tf.float32, [None, 784])
    x_2 = tf.placeholder(tf.float32, [None, 784])
    # ring_net.inputs() 

    # encoding
    y_1 = ring_net.encoding(x_1)
    # dynamic system
    y_2 = ring_net.dynamic_compression(y_1)
    # decoding 
    x_2_out = ring_net.decoding(y_2)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    error = ring_net.loss(x_2, x_2_out)
    train_op = ring_net.train(error)

    # List of all Variables
    variables = tf.all_variables()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session()
    sess.run(init)

    # Start the queue runners.
    #tf.train.start_queue_runners(sess=sess)

    #graph_def = sess.graph.as_graph_def(add_shapes=True)
    #summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
    #                                        graph_def=graph_def)

    for step in xrange(FLAGS.max_steps):
      ins, outs = k.generate_28x28(1,50)
      _ , loss_value = sess.run([train_op, error],feed_dict={x_inputs:ins[0], x_outputs:ins[0]})
      print(loss_value)

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
      if step%20 == 0:
        #new_im = x_outputs.eval(feed_dict={x_inputs: ins[0,10:11], x_outputs: ins[0,10:11]})
        #for j in range(15):
        #  new_im = y_conv.eval(feed_dict={x: new_im, keep_prob: 1.0})
        #plt.imshow(new_im.reshape((28,28)))
        #plt.savefig('new_run_1.png')
        print("Saved")

      #if step % 100 == 0:
      #  summary_str = sess.run(summary_op,feed_dict={keep_prob:FLAGS.dropout_hidden,keep_prob_input:FLAGS.dropout_input,lr:learning_rate})
      #  summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      #if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
      #  checkpoint_path = os.path.join(FLAGS.train_dir, 'YoloNeovision.ckpt')
      #  saver.save(sess, checkpoint_path, global_step=step)

def main(argv=None):  # pylint: disable=unused-argument
  train()

if __name__ == '__main__':
  tf.app.run()
