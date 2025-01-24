
import numpy as np

class Affine:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z
    
    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        dw = np.dot(self.x.T, dout)
        db = dout
        return dx, dw, db
    
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
    
class softMaxCee:
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, a, t):
        self.t = t
        self.y = functions.softmax(a)
        return functions.cee(self.y, t)

    def backward(self, dout=1):
        return self.y - self.t

class functions:
    def softmax(self, x):
        x -= x.max()
        return np.exp(x) / np.sum(np.exp(-x))
    
    def cee(self, y, t):
        alpha = 1e-7
        if y.ndim == 1:
            y = y.reshape(1, y.size)
            t = t.reshape(1, t.size)

        batch_size = t.shape[0]
        return -np.sum(t * np.log(y + alpha)) / batch_size

