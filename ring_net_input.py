

import tensorflow as tf
import cannon as cn

def inputs(batch_size):
  """Construct input for ring net.
  Args:
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, 784] size.
  """
  k = cn.Cannon()
  x_1_inputs, x_2_inputs = k.generate_28x28(1, batch_size)
  
  return x_1_inputs[0]
