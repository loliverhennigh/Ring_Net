
import numpy as np
import tensorflow as tf

import cannon as cn

import ring_net

def train():
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # Get images and labels for CIFAR-10.
    k = cn.Cannon()
    x_inputs = ring_net.inputs() 

    # Build a Graph that computes the logits predictions from the
    # inference model.
    output = ring_net.inference(x_inputs)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    error = ring_net.loss(output)
    train_op = ring_net.train(error)

    # List of all Variables
    variables = tf.all_variables()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                            graph_def=graph_def)

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      learning_rate = neovision.RATES[bisect.bisect(neovision.CUM_ITERS_AT_RATE,step)]
      _ , loss_value = sess.run([train_op, loss],feed_dict={keep_prob:FLAGS.dropout_hidden,keep_prob_input:dropout_input,lr:learning_rate})
#      ret_value, df_dx_value, loss_value = sess.run([ret, df_dx, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op,feed_dict={keep_prob:FLAGS.dropout_hidden,keep_prob_input:FLAGS.dropout_input,lr:learning_rate})
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'YoloNeovision.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

def main(argv=None):  # pylint: disable=unused-argument
  train()

if __name__ == '__main__':
  tf.app.run()
