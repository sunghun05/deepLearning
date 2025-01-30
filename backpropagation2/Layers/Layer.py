# from wsgiref.simple_server import demo_app
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class Affine:

    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None

    def forward(self, x):
        self.x = x
        z = np.dot(x, self.w) + self.b
        #graph(self.w, self.b)
        return z

    def backward(self, dout, x):
        # print(f"affine x, w, b : \n{self.x}, \n{self.w}, \n{self.b}")
        dx = np.dot(dout, self.w.T)
        dw = np.dot(np.transpose(self.x), dout)
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
        self.y = softmax(a)
        # print(f"softmax bf cee : {self.y}")
        return cee(self.y, t)

    def backward(self, dout=1):
        return self.y - self.t


def softmax(x):
    x -= x.max(axis=1, keepdims=True)
    denom = np.sum(np.exp(x), axis=1)
    nom = np.exp(x)
    denom = denom.reshape(-1, 1)
    res = nom / denom
    return res

def cee(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    # if t.size == y.size:
    #     t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def graph(w, b):
    w = w.flatten()
    b = b.flatten()
    count_w = Counter(w)
    count_b = Counter(b)
    plt.bar(count_w.keys(), count_w.values(), color="blue")
    plt.xticks(np.unique(w))  # x축 눈금을 고유값으로 설정
    plt.show()

