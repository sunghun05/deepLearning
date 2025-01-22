import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    # c = np.max(x)
    # exp_x = np.exp(x-c)
    # sum_exp_x = np.sum(exp_x)
    # y = exp_x / sum_exp_x
    # return y
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def cross_entropy_error(y, t):
    delta = 1e-7
    return (-1)*(np.sum(t*np.log(y+delta)))

def numercial_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        idx = it.multi_index
        tmp_val = x[idx]

        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
        print(idx)
        it.iternext()

    return grad
if __name__ == "__main__.py":
    pass