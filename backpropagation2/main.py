import time
import numpy as np
from Layers.Layer import *
from Layers.Network import *
from dataset.mnist import load_mnist

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

iters_num = 100
batch_size = 100
train_size = x_train.shape[0]

grad_bpg = Net(inputSize=784, hiddenSize=50, outputSize=10)
grad_slope = Net(inputSize=784, hiddenSize=50, outputSize=10)

grad_bpg_list = {}
grad_slope_list = {}

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    # print(batch_mask)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad_slope_list = grad_slope.slope_by_grad(x_batch, t_batch)
    #grad_bpg_list = grad_bpg.slope_by_BPG(x_batch, t_batch)
    print(f"gradient by differentiation {grad_slope_list}")



