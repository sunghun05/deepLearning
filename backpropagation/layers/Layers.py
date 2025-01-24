# Layers.py
import numpy as np
from layers.TwoLayers import TwoLayerNet as TL
# from functions.funcs import *

class AddLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x+y
    
    def backward(self, dout):
        dx = dout
        dy = dout
        return dx, dy

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        
        return self.x, self.y
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy
    
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0

        return out
        
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
        
class sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = functions.sigmoid_batch(x)
        self.out = out

        return self.out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
    
# for batch
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(self.x , self.W) + self.b
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, np.transpose(self.W))
        self.dW = np.dot(np.transpose(self.x), dout) 
        self.db = np.sum(dout, axis=0) # bias' backpropagation is gotten by summing its elements
        return dx

class Softmax_L:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = functions.softmax(x)
        self.loss = functions.cee(self.y, self.t)
        return self.loss
        
    def backward(self, dout=1): # unused value
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size 
        return dx

class functions:
    def __init__(self):
        pass
        
    def cee(self, y, t):
    # if data is not for batch
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        batch_size = y.shape[0]
        return -np.sum(t * np.log(y + 1e-7)) / batch_size

    def sigmoid_batch(self, x0):
        y = 1 / (1 + np.exp(-x0))
        batch_size = x0.shape[0]
        return y / batch_size

    def softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)  # prevent from being overflowed
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

    def numerical_gradient(self, xw, t, key):
        h = 1e-3  # 0.001
        grad = np.zeros_like(xw)

        it = np.nditer(xw, flags=['multi_index'], op_flags=['readwrite'])
        print(f"it : {it}")

        while not it.finished:
            idx = it.multi_index
            tmp_val = xw[idx]
            
            w0 = float(tmp_val) + h
            TL.layers[key].W = w0
            #print(fxh1)
            # fxh1 = f(x[idx])  # f(x+h)
            
            w1 = float(tmp_val) + h
            TL.layers[key].W = w1
            #print(fxh2)
            # fxh2 = f(x[idx])  # f(x-h)

            xw[idx] = tmp_val  # 값 복원
            it.iternext()
        print("gradient : ")
        print(grad)
        print(f"^^^^^^^^^^^^^^{grad.shape}")

        return grad