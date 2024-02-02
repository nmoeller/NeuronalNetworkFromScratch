from ActivationFunctions import sigmoid, sigmoid_derivative
import numpy as np
from Network import Network
from FullyConnectedLayer import FullyConnectedLayer
from ActivaitionLayer import ActivationLayer

from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

raw_test_image = x_test

x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
y_train = to_categorical(y_train)

x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = to_categorical(y_test)

# network
net = Network()
net.addLayer(FullyConnectedLayer(28*28, 100))
net.addLayer(ActivationLayer(sigmoid, sigmoid_derivative))
net.addLayer(FullyConnectedLayer(100, 50))
net.addLayer(ActivationLayer(sigmoid, sigmoid_derivative))
net.addLayer(FullyConnectedLayer(50, 10))
net.addLayer(ActivationLayer(sigmoid, sigmoid_derivative))

# set evaluation data
net.set_evaluation_data(x_test[:100], y_test[:100])
# train
net.fit(x_train[:4000], y_train[:4000], epochs=10, learning_rate=0.5)

# infrence
test_image_index = 0
result = net.inference(x_test[test_image_index])

plt.subplot(2, 2, 1)
plt.plot(net.error_list,label='Loss')
plt.legend() 
plt.subplot(2, 2, 2)
plt.plot(net.accuracy_list,label='Accuracy') 
plt.legend() 
plt.subplot(2, 2, 3)
plt.imshow(np.array(raw_test_image[test_image_index], dtype='float'),cmap='gray')
plt.title("Model Result: " + str(np.argmax(result)))
plt.show()
