## Snippets of code taken or inspired to original authors' code at [https://github.com/ginobilinie/medSynthesis](https://github.com/ginobilinie/medSynthesis)

### PSNR function
Original code at [https://github.com/ginobilinie/medSynthesis/blob/master/3dganversion/utils.py](https://github.com/ginobilinie/medSynthesis/blob/master/3dganversion/utils.py).
Our own code gets the PSNR function exactly as it is.

```python
def psnr(ct_generated,ct_GT):
  print ct_generated.shape
  print ct_GT.shape

  mse=np.sqrt(np.mean((ct_generated-ct_GT)**2))
  print 'mse ',mse
  max_I=np.max([np.max(ct_generated),np.max(ct_GT)])
  print 'max_I ',max_I
  return 20.0*np.log10(max_I/mse)
```

### LP loss and GDL
Original code at [https://github.com/ginobilinie/medSynthesis/blob/master/3dganversion/loss_functions.py](https://github.com/ginobilinie/medSynthesis/blob/master/3dganversion/loss_functions.py).

```python
def lp_loss(ct_generated, gt_ct, l_num, batch_size_tf):
  """
  Calculates the sum of lp losses between the predicted and ground truth frames.
  @param ct_generated: The predicted ct
  @param gt_ct: The ground truth ct
  @param l_num: 1 or 2 for l1 and l2 loss, respectively).
  @return: The lp loss.
  """
  lp_loss=tf.reduce_sum(tf.abs(ct_generated - gt_ct)**l_num)/(2*tf.cast(batch_size_tf,tf.float32))
  #print 'lp_loss ',gt_ct.get_shape()
  tf.add_to_collection('losses', lp_loss)

  loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
  return loss
  ```

```python

def gdl_loss(gen_frames, gt_frames, alpha):
  """
  Calculates the sum of GDL losses between the predicted and ground truth frames.
  @param gen_frames: The predicted frames at each scale.
  @param gt_frames: The ground truth frames at each scale
  @param alpha: The power to which each gradient term is raised.
  @return: The GDL loss for 2d.
  """
  # calculate the loss for each scale
  scale_losses = []
  for i in xrange(len(gen_frames)):
    # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
    pos = tf.constant(np.identity(1), dtype=tf.float32)
    neg = -1 * pos
    filter_x = tf.expand_dims(tf.pack([neg, pos]), 0)  # [-1, 1]
    filter_y = tf.pack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]
    strides = [1, 1, 1, 1]  # stride of (1, 1)
    padding = 'SAME'

    gen_dx = tf.abs(tf.nn.conv2d(gen_frames[i], filter_x, strides, padding=padding))
    gen_dy = tf.abs(tf.nn.conv2d(gen_frames[i], filter_y, strides, padding=padding))
    gt_dx = tf.abs(tf.nn.conv2d(gt_frames[i], filter_x, strides, padding=padding))
    gt_dy = tf.abs(tf.nn.conv2d(gt_frames[i], filter_y, strides, padding=padding))

    grad_diff_x = tf.abs(gt_dx - gen_dx)
    grad_diff_y = tf.abs(gt_dy - gen_dy)

    scale_losses.append(tf.reduce_sum((grad_diff_x ** alpha + grad_diff_y ** alpha)))

  # condense into one tensor and avg
  return tf.reduce_mean(tf.pack(scale_losses))
```
    
While this is how we have implemented both loss functions in one customized loss function.
    
```python
# This is a part of the whole generator loss function that will be used in the GAN model
# Indeed, the adversarial loss L_ADV(X) is added in the GAN's definition

# The percentages of L_G and L_GDL losses to use in the combined loss
lambda_2 = 1.0
lambda_3 = 1.0

def custom_loss(y_true, y_pred):
  # L1/L2 Loss: set tf.abs or tf.square respectively
  lp_loss = tf.reduce_sum(tf.abs(y_true - y_pred)) / tf.cast(BATCH_SIZE, tf.float32)

  # Image Gradient Difference Loss
  dPredicted_x, dPredicted_y = tf.image.image_gradients(tf.reshape(y_pred, shape=(BATCH_SIZE * 32, 32, 32, 1)))
  dTrue_x, dTrue_y = tf.image.image_gradients(tf.reshape(y_true, shape=(BATCH_SIZE * 32, 32, 32, 1)))
  gdl = tf.reduce_sum(tf.square(tf.abs(dTrue_x) - tf.abs(dPredicted_x)) + tf.square(tf.abs(dTrue_y) - tf.abs(dPredicted_y))) / tf.cast(BATCH_SIZE, tf.float32)

  return lambda_2 * lp_loss + lambda_3 * gdl
```
