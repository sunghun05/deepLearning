# activation functions for neural network

import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self):
        pass

    def step_function(self, x):
        res = []
        if type(x) == list or type(x) == np.ndarray:
            for i in range(len(x)):
                if x[i]>0:
                    res.append(1)
                else:
                    res.append(0)
        else:
            if x > 0:
                res = 1
            else:
                res = 0
        return res

    def sigmoid(self, x):

        res = 1 / (1 + np.exp(-x))

        return res

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def graph(self, x):

        x_axis = np.arange(-9, 9, 0.1)

        # step function
        plt.subplot(3, 1, 1)
        plt.title("step function")
        plt.plot(x, self.step_function(x), 'ro', label='input')
        plt.plot(x_axis, self.step_function(x_axis), 'blue', label='step function', linewidth=0.9)
        plt.legend(loc='lower right')

        # sigmoid
        plt.subplot(3, 1, 2)
        plt.title("sigmoid function")
        plt.plot(x, self.sigmoid(x), 'ro', label='input')
        plt.plot(x_axis, self.sigmoid(x_axis), 'green', label='sigmoid', linewidth=0.9)

        # relu
        plt. subplot(3, 1, 3)
        plt.title("relu function")
        plt.plot(x, self.relu(x), 'ro', label='input')
        plt.plot(x_axis, self.relu(x_axis), 'black', label='relu', linewidth=0.9)

        # draw
        plt.plot(x, 'black', linewidth=0.8)

        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

class foward(NeuralNetwork):

    def __init__(self, x:np.ndarray, w:np.ndarray, b:np.ndarray, layer:int):
        self.x = x
        self.w = w
        self.b = b
        self.layer = layer

    # compute hidden layer by sigmoid
    def hidden(self):
        try:
            a = self.x @ self.w + self.b
            return self.sigmoid(a)
        except ValueError:
            a = self.x @ np.transpose(self.w) + self.b
            return self.sigmoid(a)

    def identity(self):
        return self.x


# if __name__ == '__main__':
#     net = NeuralNetwork()
#     first = foward(
#                     np.array([1.0, 0.5]),
#                    np.array([[0.1, 0.3, 0.5],
#                              [0.2, 0.4, 0.6]]),
#                    np.array([0.1, 0.2, 0.3]), 1
#     )
#     second = foward(
#                     first.hidden(),
#                     np.array([[0.1, 0.4],
#                               [0.2, 0.5],
#                               [0.3, 0.6]]),
#                     np.array([0.1, 0.2]), 2
#     )
#     res = foward(
#         second.hidden(),
#         np.array([]),
#         np.array([]), 3
#     )
#     print(f"first : {first.hidden()}")
#     print(f"2nd : {second.hidden()}")
#     print(f"res : {res.identity()}")
#
#     A = np.array([[1, 2], [3, 4], [5, 6]])
#     B = np.array([[4, 3, 2], [0, 2, 1]])
#
#     print(f"A : \n{A}")
#     print(f"B : \n{B}")
#
#     print(f"A shape : {A.shape}")
#     print(f"B shape : {B.shape}")
#     print(f"np dim A : {np.ndim(A)}")
#     print(f"np dim B : {np.ndim(B)}")
#
#     print(f"A @ B : \n{A @ B}")
#     print(f"np.dot(A, B) : \n{np.dot(A, B)}")

    #net.graph(1)

