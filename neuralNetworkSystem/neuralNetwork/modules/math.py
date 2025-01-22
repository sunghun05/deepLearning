
import numpy as np

class pmath:
    def __init__(self):
        pass
    def diff(self, f, x):
        h = 1e-4
        return (f(x + h) - f(x - h)) / (2 * h)      # 중앙 차분법
    def gradient(self, f, x): # function parameter
        h = 1e-4
        grad = np.zeros_like(x)
        for i in range(x.size):
            tmp_val = x[i]
            x[i] = tmp_val + h
            fxh1 = f(x)
            #f(x+h)

            #f(x-h)
            x[i] = tmp_val - h
            fxh2 = f(x)

            grad[i] = (fxh1 - fxh2) / (2 * h)
            x [i] = tmp_val

        return grad
     