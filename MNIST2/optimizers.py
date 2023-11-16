import tensorflow as tf

class GradientDescent:
  def __init__(self, learning_rate=1e-3):
    # Initialize parameters
    self.learning_rate = learning_rate
    self.title = f"Gradient descent optimizer: learning rate = {self.learning_rate}"


  """
  Init weights and biases
  args: 
    variables: 
    grads:
  return: new grads
  """
  def apply_gradients(self, variables, grads):
    # Update variables
    return [v.assign_sub(delta * self.learning_rate) for v, delta in zip(variables, grads)]


class Momentum(tf.Module):
  def __init__(self, learning_rate=1e-3, momentum=0.7):
    # Initialize parameters
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.change = 0.
    self.title = f"Gradient descent optimizer w/Momentum: learning rate = {self.learning_rate}, momentum={self.momentum}"

  
  def apply_gradients(self, variables, grads):
    # Update variables 
    updated_vars = []
    for v, delta in zip(variables, grads):
      curr_change = (self.learning_rate * delta) + (self.momentum * self.change)
      print(f"current_change: {curr_change}")
      v.assign_sub(curr_change)
      updated_vars.append(v)
      self.change = curr_change
    return updated_vars


class Adam:
  def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, ep=1e-7):
    # Initialize the Adam parameters
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.learning_rate = learning_rate
    self.ep = ep
    self.t = 1.
    self.v_dvar, self.s_dvar = [], []
    self.title = f"Adam: learning rate = {self.learning_rate}"
    self.built = False


  def apply_gradients(self, variables, grads):
    updated_vars = []
    # Set up moment and RMSprop slots for each variable on the first call
    if not self.built:
      for var in variables:
        v = tf.Variable(tf.zeros(shape=var.shape))
        s = tf.Variable(tf.zeros(shape=var.shape))
        self.v_dvar.append(v)
        self.s_dvar.append(s)
      self.built = True
    # Perform Adam updates
    for i, (d_var, var) in enumerate(zip(grads, variables)):
      # Moment calculation
      self.v_dvar[i] = self.beta_1*self.v_dvar[i] + (1-self.beta_1)*d_var
      # RMSprop calculation
      self.s_dvar[i] = self.beta_2*self.s_dvar[i] + (1-self.beta_2)*tf.square(d_var)
      # Bias correction
      v_dvar_bc = self.v_dvar[i]/(1-(self.beta_1**self.t))
      s_dvar_bc = self.s_dvar[i]/(1-(self.beta_2**self.t))
      # Update model variables
      var.assign_sub(self.learning_rate*(v_dvar_bc/(tf.sqrt(s_dvar_bc) + self.ep)))
      updated_vars.append(var)
    # Increment the iteration counter
    self.t += 1.
    return updated_vars
