
from Layers.Layer import *
from collections import OrderedDict

class Net:
    def __init__(self, inputSize, hiddenSize, outputSize, w_init=0.01, learningRate=0.1):
        self.params = {}
        self.params['W1'] = w_init * np.random.randn(inputSize, hiddenSize)
        self.params['b1'] = np.zeros(1, hiddenSize)
        self.params['W2'] = w_init * np.random.randn(hiddenSize, outputSize)
        self.params['b2'] = np.zeros(1, outputSize)
        self.t = None

        self.cnt = 0

        self.newNet = OrderedDict()
        self.init()

    def init(self): # renews parameters
        self.newNet['Affine1'] = Affine(self.params['W1'],self.params['b1'])
        self.newNet['Relu1'] = Relu()
        self.newNet['Affine2'] = Affine(self.params['W2'],self.params['b2'])
        self.newNet['Relu2'] = Relu()

    def predict(self, x):

        for layer in self.newNet:
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        loss_input = softMaxCee()
        loss = loss_input.forward(x, t)
        return loss

    def slope_by_BPG(self, x):
        pass

    def slope_by_grad(self, x):
        grads = {}
        for key in ('W1', 'b1', 'W2', 'b2'):
            grads[key] = self.partial_diff(x, key)

        return grads

    def partial_diff(self, x, key): # becareful if it is using same batch data
        h = 1e-7
        x0 = x + h
        x1 = x - h
        self.params[key] = x0
        self.init()
        fx0 = self.predict(x)
        fx0 = self.loss(fx0)
        self.params[key] = x1
        self.init()
        fx1 = self.predict(x)
        fx1 = self.loss(fx1)

        grad = (fx0 - fx1) / (2*h)

        return grad


        