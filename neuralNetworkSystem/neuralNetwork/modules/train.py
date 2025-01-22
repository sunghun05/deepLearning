import numpy as np
from neuralNetwork.modules import math


class loss:
    def __init__(self):
        pass
    def sum_of_squared_error(self, y, t):
        return 0.5 * np.sum((y-t)**2)
    def cross_entropy_error(self, y, t):
        delta = 1e-7
        return -np.sum(t*np.log(y+delta)) # because log0 does not exist
    def gradient_method(self, f, init_x, lr=0.01, step_num=100):
        x = init_x

        for i in range(step_num):
            grad = math.gradient(f, x)
            x -= lr * grad
        return x
    