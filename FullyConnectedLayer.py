import numpy as np
from Layer import Layer

class FullyConnectedLayer(Layer):

    weights : np.ndarray
    input : np.ndarray
    output : np.ndarray

    def __init__(self, input_size : int, output_size : int):
        self.weights = np.random.randn(input_size, output_size) * 0.1


    def forward_pass(self, inputs : np.ndarray) -> np.ndarray:
        self.input = inputs
        self.output = np.dot(inputs, self.weights)
        return self.output
    
    def backward_pass(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        #Calculate Error for each Weight
        weights_error = np.dot(self.input.T, output_error)
        #Adjust Each Weight by its Error and Learning Rate
        self.weights -= learning_rate * weights_error
        
        #Distribute Error to Previous Layer
        return input_error
