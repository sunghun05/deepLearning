
import numpy as np
from layers.TwoLayers import *
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

network1 = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
network2 = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]
print(x_batch.shape)
grad_numerical = network1.numerical_gradient(x_batch, t_batch)
grad_backdrop = network2.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backdrop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))