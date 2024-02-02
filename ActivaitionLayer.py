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
        pass
    
    def backward_pass(self, output_error : np.ndarray, learning_rate : float) -> np.ndarray:
        pass
