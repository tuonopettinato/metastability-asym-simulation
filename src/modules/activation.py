"""
modules/activation.py contains activation functions used in the neural network dynamics simulations.
"""
import numpy as np


def threshold_function(x, amplitude=1.0):
  """Step/threshold function"""
  return amplitude * np.sign(x)

def sigmoid_function(x, r_m=1.0, beta=1.0, x_r=0.0):
  """
  Sigmoid function with adjustable maximum firing rate, slope at inflection point, 
  inflection point or threshold (x_r)

  Parameters:
  -----------
  x : array-like
      Input values
  r_m : float
      Maximum firing rate (amplitude)
  beta : float
      Slope parameter at the inflection point
  x_r : float
      Inflection point location (threshold)

  Returns:
  --------
  y : array-like
      Output values between 0 and r_m
  """
  # Clip input to prevent overflow in exp function
  clipped_input = np.clip(-beta * (x - x_r), -500, 500)
  return r_m / (1.0 + np.exp(clipped_input))

def relu_function(x, amplitude=1.0):
  """
  Standard ReLU function without clipping

  Parameters:
  -----------
  x : array-like
      Input values
  amplitude : float
      Scaling factor applied to the output (gain)

  Returns:
  --------
  y : array-like
      Output values, max(0, x) scaled by amplitude
  """
  return amplitude * np.maximum(0, x)

def step_function(x, q_f=1.0, x_f=0.0):
  """
  Step function of the form: f = q_f for x > x_f, -(1-q_f) elsewhere
  
  Parameters:
  -----------
  x : array-like
      Input values
  q_f : float
      Value when x > x_f
  x_f : float
      Threshold value
  
  Returns:
  --------
  y : array-like
      Output values: q_f if x > x_f, q_f - 1 otherwise
  """
  x = np.asarray(x)
  y = np.where(x > x_f, q_f, q_f - 1.0)
  return y

def inverse_sigmoid_function(x, r_m=1.0, beta=1.0, x_r=0.0):
    """
    Inverse of the 3-parameter sigmoid: g(x) = 1 / (1 + exp(-a(x - theta)))
    """
    x = np.clip(x, 1e-6, r_m - 1e-6)  # Avoid division by zero with clipping
    return x_r + (1 / beta) * np.log(x / (r_m - x))

def inverse_relu_function(x, amplitude=1.0):
    """
    Inverse of the ReLU function: g(x) = x / amplitude for x >= 0, 0 for x < 0
    """
    x = np.asarray(x)
    return np.where(x >= 0, x / amplitude, 0.0)

def derivative_sigmoid_function(x, r_m=1.0, beta=1.0, x_r=0.0):
    """
    Derivative of the 3-parameter sigmoid: 
    g(x) = r_m / (1 + exp(-beta(x - x_r)))
    g'(x) = beta * g(x) * (1 - g(x)/r_m)
    """
    g_x = r_m / (1 + np.exp(-beta * (x - x_r)))
    return beta * g_x * (1 - g_x / r_m)

def derivative_relu_function(x, amplitude=1.0):
    """
    Derivative of the ReLU function:
    g(x) = max(0, x) / amplitude
    g'(x) = 1/amplitude for x > 0, 0 otherwise
    """
    x = np.asarray(x)
    return np.where(x > 0, 1.0 / amplitude, 0.0)