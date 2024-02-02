import numpy as np
from Layer import Layer

class FullyConnectedLayer(Layer):

    weights : np.ndarray
    input : np.ndarray
    output : np.ndarray

    def __init__(self, input_size : int, output_size : int):
        self.weights = np.random.randn(input_size, output_size) * 0.1


    def forward_pass(self, inputs : np.ndarray) -> np.ndarray:
        pass
    
    def backward_pass(self, output_error, learning_rate):
        pass
