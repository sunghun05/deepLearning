
from Layers.Layer import *
from collections import OrderedDict

class Net:
    def __init__(self, inputSize, hiddenSize, outputSize, w_init=0.01, learningRate=0.1):
        self.params = {}
        self.params['W1'] = w_init * np.random.randn(inputSize, hiddenSize)
        self.params['b1'] = np.zeros(hiddenSize)
        self.params['W2'] = w_init * np.random.randn(hiddenSize, outputSize)
        self.params['b2'] = np.zeros(outputSize)
        self.t = None

        #print(self.params['W1'])

        self.cnt = 0

        self.newNet = OrderedDict()
        self.init()
        self.lastLayer = softMaxCee()

    def init(self): # renews parameters
        self.newNet['Affine1'] = Affine(self.params['W1'],self.params['b1'])
        self.newNet['Relu1'] = Relu()
        self.newNet['Affine2'] = Affine(self.params['W2'],self.params['b2'])
        self.newNet['Relu2'] = Relu()
        self.lastLayer = softMaxCee()

    def predict(self, x):
        # print(f"x pre : {x}")
        for key in self.newNet.keys():
            x = self.newNet[key].forward(x)

        return x

    def loss(self, x, t):
        loss_input = softMaxCee()
        loss = loss_input.forward(x, t)
        return loss

    def slope_by_BPG(self, x, t):

        cp_x = x
        grad = {}
        self.newNet['Affine1'].x = x
        self.newNet['Affine2'].x = x
        # self.init()
        #forward
        #print(self.newNet)
        for key in self.newNet.keys():
            #print(key)
            x = self.newNet[key].forward(x)
        self.lastLayer.forward(x, t)

        # refresh x
        x = cp_x

        #print(f"start backward, x : {x}")
        dx = self.lastLayer.backward()   # default dout = 1
        dx = self.newNet['Relu2'].backward(dx)
        self.init()
        (dx, dw1, db1) = self.newNet['Affine2'].backward(dx, x)
        dx = self.newNet['Relu1'].backward(dx)
        self.init()
        (grad['dx'], grad['dw'], grad['db']) = self.newNet['Affine1'].backward(dx, x)

        return grad


    def slope_by_grad(self, x, t):
        grads = {}
        #print(f"params : {self.params}")
        for key in ('W1', 'b1', 'W2', 'b2'):
            # print(f"x preprepre : {x.shape}")
            np.array(self.params[key])
            grads[key] = self.partial_diff(x, t, self.params[key], key)

        return grads

    def partial_diff(self, x, t, param, key): # becareful if it is using same batch data
        # print(f"x prepre : {x}")
        grad = np.zeros_like(param)
        tmp0 = np.zeros_like(param)
        tmp1 = np.zeros_like(param)

        h = 1e-7
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
#           print(key)
            idx = it.multi_index
#           print(idx)
            x0 = param[idx] + h
            x1 = param[idx] - h
            #tmp = param[idx]
            # print(f"x0, x1 : {x0}, {x1}")

            tmp0[idx] = x0
            self.params[key] = tmp0
            self.init()

            fx0 = self.predict(x)
            fx0 = self.loss(fx0, t)

            tmp1[idx] = x1
            self.params[key] = tmp1
            self.init()

            fx1 = self.predict(x)
            fx1 = self.loss(fx1, t)

            #print(f"fx0, fx1 : {fx0}, {fx1}")

            grad[idx] = (fx0 - fx1) / (2*h)

            # print(f"grad : {grad[idx]}")
            # print(f"tmp0, tmp1 : {tmp0}\n {tmp1}")

            #self.params[key] = tmp
            it.iternext()

        return grad


        