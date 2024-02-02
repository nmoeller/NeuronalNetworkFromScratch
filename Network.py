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
        output : np.ndarray = input_data
        for layer in self.layers:
            output = layer.forward_pass(output)
        return output
    
    def set_evaluation_data(self, x_test : np.ndarray, y_test : np.ndarray):
        self.x_test = x_test
        self.y_test = y_test

    def evaluate(self):
        samples = len(self.x_test)
        correct = 0
        for i in range(samples):
            x = self.x_test[i]
            y = self.y_test[i]
            prediction = self.inference(x)
            if np.argmax(prediction) == np.argmax(y):
                correct += 1
        accuracy = correct / samples
        self.accuracy_list.append(accuracy)
    
    def fit(self, x_train : np.ndarray, y_train : np.ndarray, epochs : int, learning_rate : float):
        samples = len(x_train)

        for i in range(epochs):
            err = 0
            for j in range(samples):

                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_pass(output)

                err += self.loss(y_train[j], output)

                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_pass(error, learning_rate)

            err /= samples
            if self.x_test is not None and self.y_test is not None:
                self.evaluate()
            self.error_list.append(err)
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))