import numpy as np
from Layer import Layer
from LossFunctions import mean_square_error, mean_square_error_derivative

class Network():

    layers : list[Layer]
    loss : callable
    loss_prime : callable
    error_list : list[float]
    accuracy_list : list[float]

    def __init__(self):
        self.layers = []
        self.error_list = []
        self.accuracy_list = []
        self.loss = None
        self.loss_prime = None
        self.loss = mean_square_error
        self.loss_prime = mean_square_error_derivative

    def addLayer(self, layer : Layer):
        self.layers.append(layer)

    def inference(self, input_data : np.ndarray) -> np.ndarray:
        pass
    
    def set_evaluation_data(self, x_test : np.ndarray, y_test : np.ndarray):
        self.x_test = x_test
        self.y_test = y_test

    def evaluate(self):
        pass
    
    def fit(self, x_train : np.ndarray, y_train : np.ndarray, epochs : int, learning_rate : float):
        pass