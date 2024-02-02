import numpy as np
from Layer import Layer

class ActivationLayer(Layer):

    input : np.ndarray
    activation : callable
    activation_derivative : callable

    def __init__(self, activation : callable , activation_derivative : callable):
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward_pass(self, inputs : np.ndarray) -> np.ndarray:
        self.input = inputs
        return self.activation(inputs)
    
    def backward_pass(self, output_error : np.ndarray, learning_rate : float) -> np.ndarray:
        return self.activation_derivative(self.input) * output_error
