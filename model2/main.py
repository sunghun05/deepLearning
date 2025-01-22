
import numpy as np
from dataset.mnist import load_mnist
from packages.functions import *

class TwoLayerNet:
    def __init__(self, input, output, hidden, lr, weight_init=0.01): # lr : learning, rate
        self.params = {}
        self.params['W1'] = weight_init * np.random.randn(input, hidden)
        self.params['b1'] = np.zeros(hidden)
        self.params['W2'] = weight_init * np.random.randn(hidden, output)
        self.params['b2'] = np.zeros(output)
        self.lr = lr
        self.lossRes = []

    def predict(self, input):
        # input parameter's shape must be (28, 28)
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(input, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2

        y = softmax(a2)

        return y

    def loss(self, pr, t):
        x = self.predict(pr)
        return cross_entropy_error(x, t)

    def gradient_des(self, input, t):
        # print(input.shape)
        # t : answer label
        loss_w = lambda w: self.loss(input, t)

        # get slope of w, b in loss function
        slopes = {}
        slopes['W1'] = diff(loss_w, self.params['W1'])
        slopes['b1'] = diff(loss_w, self.params['b1'])
        slopes['W2'] = diff(loss_w, self.params['W2'])
        slopes['b2'] = diff(loss_w, self.params['b2'])
        return slopes


if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, one_hot_label=True)

    train_loss_list = []

    # hyper parameter
    iters_num = 100
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.01

    net = TwoLayerNet(input=784, hidden=50, output=10, lr=learning_rate)

    for i in range(iters_num):
        print(f"===================={i}====================")
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        slopes = net.gradient_des(x_batch, t_batch)

        # modify parameters
        for key in ('W1', 'b1', 'W2', 'b2'):
            net.params[key] -= learning_rate * slopes[key]

        loss = net.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        print(train_loss_list[i])
