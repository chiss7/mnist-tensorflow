import tensorflow as tf
import numpy as np

class Model(tf.Module):
  def __init__(self, *args):
    self.arch = args[0]
    self.title = f"Model architecture -> input[28x28], hidden{self.arch}, output[10]"
    self.activation_functions = [tf.nn.relu, tf.nn.relu, tf.identity]
    self.built = False


  def xavier_init(self, shape):
    # The weights should also be initialized properly to prevent activation outputs from 
    # becoming too large or small.
    # Computes the xavier initialization values for a weight matrix
    in_dim, out_dim = shape
    xavier_lim = tf.sqrt(6.) / tf.sqrt(tf.cast(in_dim + out_dim, tf.float32))
    weight_vals = tf.random.uniform(shape=(in_dim, out_dim), minval=-xavier_lim, maxval=xavier_lim, seed=22)
    return weight_vals

  """
  Init weights and biases
  args: 
    input_len: number of features of x
  """
  def build(self, input_len):
    # Init w with xavier scheme
    w_init = self.xavier_init
    # Init b
    b_init = tf.zeros

    self.weights = []
    self.biases = []

    ## Loop through layers
    for dim in self.arch:
      self.weights.append(tf.Variable(w_init(shape=(input_len, dim))))
      self.biases.append(tf.Variable(b_init(shape=(dim,))))
      input_len = dim
    
    ## Final Layer [10 neuron]:
    # Init weights for final layer
    self.weights.append(tf.Variable(w_init(shape=(dim, 10))))
    # Init biases for final layer
    self.biases.append(tf.Variable(b_init(shape=(10))))

    self.Variables = self.weights + self.biases


  """
  Feedforward function
  args:
    x: input tensor (128, 784)
  return:
    y: output tensor (128, 10)
  """
  @tf.function # computational graph, (better performance) 
  def __call__(self, x):
    # Check if run for first time
    if not self.built:
      self.build(x.shape[1])
      self.built = True

    for w, b, act_f in zip(self.weights, self.biases, self.activation_functions):
      z = tf.add(tf.matmul(x, w), b)
      a = act_f(z)
      x = a

    return a


  """
  Loss function (Mean Squared Error)
  args:
    y_pred: input tensor 
    y_true: np array
  return:
    loss: output tensor
  """
  def compute_loss(self, y_pred, y_true):
    return tf.reduce_sum((y_pred - y_true) ** 2) / y_true.shape[0]


  """
  Loss function (cross-entropy loss)
  args:
    y_pred: input tensor 
    y_true: np array
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
    y_true: np array
  return:
    acc: output tensor
  """
  def compute_accuracy(self, y_pred, y_true):
    is_equal = tf.equal(y_true, tf.argmax(tf.nn.softmax(y_pred), axis=1))
    return tf.reduce_mean(tf.cast(is_equal, tf.float32))

