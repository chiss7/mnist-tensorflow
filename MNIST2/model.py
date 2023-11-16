import tensorflow as tf
import numpy as np

class Model(tf.Module):
  def __init__(self, filters, kernel_size, strides, fc_arch, fc_input, conv_activation_functions, fc_activation_functions):
    # Conv layer attributes
    self.conv_arch = filters
    self.conv_activation_functions = conv_activation_functions
    self.strides = strides
    self.kernel_size = kernel_size

    # Dense layer attributes
    self.fc_arch = fc_arch
    self.fc_activation_functions = fc_activation_functions
    self.fc_input = fc_input

    self.title = f"""Model architecture:
    Input Image: (batch_size, 28, 28, 1)
    {len(self.conv_arch)} Conv Layers: {self.conv_arch}
    {len(self.fc_arch)} Dense Layers: {self.fc_arch}
    Output Dense Layer: 10 Neurons"""
    self.built = False

  
  def get_fans(self, shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out


  def xavier_init(self, shape):  # GlorotUniform
    # The weights should also be initialized properly to prevent activation outputs from 
    # becoming too large or small.
    # Computes the xavier initialization values for a weight matrix
    in_dim, out_dim = self.get_fans(shape)
    xavier_lim = tf.sqrt(6.) / tf.sqrt(tf.cast(in_dim + out_dim, tf.float32))
    weight_vals = tf.random.uniform(shape=shape, minval=-xavier_lim, maxval=xavier_lim, seed=22)
    return weight_vals


  """
  Init weights and biases
  args:
    in_channels: image channels
  """
  def build(self, in_channels):
    # Init w with xavier scheme
    w_init = self.xavier_init
    # Init b
    b_init = tf.zeros

    self.conv_weights = []
    self.conv_biases = []
    self.fc_weights = []
    self.fc_biases = []

    ## Loop through Convolutional Layers
    for out_channels in self.conv_arch:
      self.conv_weights.append(tf.Variable(w_init(shape=(self.kernel_size[0], self.kernel_size[1], in_channels, out_channels))))
      self.conv_biases.append(tf.Variable(b_init(shape=(out_channels,))))
      in_channels = out_channels

    ## Loop through Dense Layers
    for dim in self.fc_arch:
      self.fc_weights.append(tf.Variable(w_init(shape=(self.fc_input, dim))))
      self.fc_biases.append(tf.Variable(b_init(shape=(dim,))))
      self.fc_input = dim
    
    ## Final Dense Layer [10 neurons]:
    # Init weights for final layer
    self.fc_weights.append(tf.Variable(w_init(shape=(dim, 10))))
    # Init biases for final layer
    self.fc_biases.append(tf.Variable(b_init(shape=(10))))

    self.conv_Variables = self.conv_weights + self.conv_biases
    self.fc_Variables = self.fc_weights + self.fc_biases
    self.Variables = self.conv_Variables + self.fc_Variables


  """
  Feedforward function
  args:
    x: input tensor (batch_size, 28, 28, 1)
  return:
    y: output tensor (batch_size, 10)
  """
  @tf.function # computational graph, (better performance) 
  def __call__(self, x):
    # Check if run for first time
    if not self.built:
      self.build(x.shape[3])
      self.built = True

    # Loop through Convolutional Layers
    for w, b, act_f in zip(self.conv_weights, self.conv_biases, self.conv_activation_functions):
      conv = tf.nn.conv2d(x, w, strides=[1, self.strides[0], self.strides[1], 1], padding="VALID")
      z = tf.nn.bias_add(conv, b)
      a = act_f(z)
      a = tf.nn.max_pool2d(a, 2, 2, padding="VALID")
      x = a

    # Reshaping the data (reshape the feature matrices to be 2-dimensional by flattening the image)
    x = tf.reshape(x, shape=[-1, x.shape[1] * x.shape[2] * x.shape[3]])

    # Loop through Dense Layers
    for w, b, act_f in zip(self.fc_weights, self.fc_biases, self.fc_activation_functions):
      z = tf.add(tf.matmul(x, w), b)
      a = act_f(z)
      x = a

    return a


  """
  Loss function (Mean Squared Error)
  args:
    y_pred: input tensor 
    y_true: input tensor
  return:
    loss: output tensor
  """
  def compute_loss(self, y_pred, y_true):
    return tf.reduce_sum((y_pred - y_true) ** 2) / y_true.shape[0]


  """
  Loss function (cross-entropy loss)
  args:
    y_pred: input tensor 
    y_true: input tensor
  return:
    loss: output tensor
  """
  def cross_entropy_loss(self, y_pred, y_true):
    # Compute cross entropy loss with a sparse operation
    # This function does not require the model's last layer to apply the softmax activation 
    # function nor does it require the class labels to be one hot encoded
    sparse_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return tf.reduce_mean(sparse_ce)
  

  """
  Acc function
  args:
    y_pred: input tensor 
    y_true: input tensor
  return:
    acc: output tensor
  """
  def compute_accuracy(self, y_pred, y_true):
    is_equal = tf.equal(y_true, tf.argmax(tf.nn.softmax(y_pred), axis=1))
    return tf.reduce_mean(tf.cast(is_equal, tf.float32))

