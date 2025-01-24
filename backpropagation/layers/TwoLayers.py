# TwoLayers.py

from layers.Layers import *
from layers.Layers import functions as f
import numpy as np
from collections import OrderedDict

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.1):
        # init parameters
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # allocate parameters to layers instance
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.lastLayer = Softmax_L()

    def init(self):
        self.params = {}
        self.params['W1'] = 
        self.params['b1'] = 
        self.params['W2'] = 
        self.params['b2'] = 

        # allocate parameters to layers instance
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.lastLayer = Softmax_L()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x
    
    def loss(self, x, t):
        y = self.predict(x)

        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y==t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        
        for key in ('W1', 'b1', 'W2', 'b2'):
            
    
    def gradient(self, x, t):

        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads

    def numerical_diff(self, xw, t):

        h = 1e-4  # 0.0001
        # grad = np.zeros_like(xw)
        x0 = np.zeros_like(xw)
        x1 = np.zeros_like(xw)

        it = np.nditer(xw, flags=['multi_index'], op_flags=['readwrite'])
        print(f"it : {it}")

        while not it.finished:
            idx = it.multi_index
            tmp_val = xw[idx]
            
            w0 = float(tmp_val) + h
            x0[idx] = w0
            # TL.layers[key].W = w0
            #print(fxh1)
            # fxh1 = f(x[idx])  # f(x+h)
            
            w1 = float(tmp_val) + h
            x1[idx] = w1
            # TL.layers[key].W = w1
            #print(fxh2)
            # fxh2 = f(x[idx])  # f(x-h)
            # grad[idx] = 

            xw[idx] = tmp_val  # 값 복원
            it.iternext()
        # grad = w1 - w0
        # print("gradient : ")
        # print(grad)
        # print(f"^^^^^^^^^^^^^^{grad.shape}")
            
        return w0, w1
    
