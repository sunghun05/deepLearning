
import numpy as np

class Affine:

    def __init__(self, w, b):
        self.w = w
        self.b = b

    def forward(self, x):
        # print(f"x , shape : {x} \n{x.shape}")
        # print(f"w : {self.w}")
        # print(f"b : {self.b}")
        #for batch
        # batch_size = x.shape[0]

        #print(f"x, w, b : {x.shape}, {self.w.shape}, {self.b.shape}")
        z = np.dot(x, self.w) + self.b
        return z
    
    def backward(self, dout, x):
        self.x = x
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
    x -= x.max()
    return np.exp(x) / np.sum(np.exp(-x))

def cee(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

