import sys, os
sys.path.append(os.pardir)
import numpy as np
from neuralNetwork.modules.neural_network import NeuralNetwork
from neuralNetwork.modules.train import loss

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)
    def predict(self, x):
        return x @ self.W
    def Ploss(self, x, t):
        z = self.predict(x)
        y = Nnet.softmax(z)
        set_loss = mlos.cross_entropy_error(y, t)
        return set_loss

if __name__ == '__main__':
    Nnet = NeuralNetwork()
    net = simpleNet()
    mlos = loss()
    print(net.W)
    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)
    print(np.argmax(p))
    t = np.array([0,0,1])
    print(net.Ploss(x, t))
