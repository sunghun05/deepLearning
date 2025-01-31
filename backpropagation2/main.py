import time

import matplotlib.pyplot as plt
import numpy as np
import pickle
from Layers.Layer import *
from collections import Counter
from Layers.Network import *
from dataset.mnist import load_mnist

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

iters_num = 10000
batch_size = 300
train_size = x_train.shape[0]

grad_bpg = Net(inputSize=784, hiddenSize=50, outputSize=10)
grad_slope = Net(inputSize=784, hiddenSize=50, outputSize=10)

grad_bpg_list = {}

for i in range(iters_num):
    grad_slope_dict = {}
    batch_mask = np.random.choice(train_size, batch_size)

    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #grad_slope_dict = grad_slope.slope_by_grad(x_batch, t_batch)
    grad_bpg_list = grad_bpg.slope_by_BPG(x_batch, t_batch)

    #y = grad_slope.predict(x_batch)
    #loss = grad_slope.loss(y, t_batch)
    loss = grad_bpg.predict(x_batch)
    loss = grad_bpg.loss(loss, t_batch)
    print(f"loss : {loss}")
    for key in ('W1', 'b1', 'W2', 'b2'):
        #grad_slope.renew_params(grad_slope_dict[key], key)
        grad_bpg.renew_params(grad_bpg_list[key], key)

# save parameters

# tmp = grad_bpg.params
# with open('params.pickle', 'wb') as f:
#     pickle.dump(tmp, f, pickle.HIGHEST_PROTOCOL)

# graph

bpg_params = {}

bpg_params['W1'] = grad_bpg.params['W1'].flatten()
bpg_params['W2'] = grad_bpg.params['W2'].flatten()
bpg_params['z1'] = grad_bpg.newNet['Relu1'].activation.flatten()
bpg_params['z2'] = grad_bpg.newNet['Relu2'].activation.flatten()

w1 = list(bpg_params['W1'])
w2 = list(bpg_params['W2'])
z1 = list(bpg_params['z1'])
z2 = list(bpg_params['z2'])

for i in range(len(w1)):
    w1[i] = round(w1[i], 2)
bpg_params['W1'] = w1

for i in range(len(w2)):
    w2[i] = round(w2[i], 1)
bpg_params['W2'] = w2

for i in range(len(z1)):
    z1[i] = round(z1[i], 1)
bpg_params['z1'] = z1

for i in range(len(z2)):
    z2[i] = round(z2[i], 1)
bpg_params['z2'] = z2

w1_cnt = Counter(bpg_params['W1'])
w2_cnt = Counter(bpg_params['W2'])
z1_cnt = Counter(bpg_params['z1'])
z2_cnt = Counter(bpg_params['z2'])

plt.suptitle(f"distribution of weight & activation value\n"
             f"batch size : {batch_size} learning rate : {grad_bpg.lr}",
             y = 0.95)
plt.subplots_adjust(hspace=0.8, top=0.8)
plt.subplot(2, 2, 1)
plt.title('distribution of W1')
plt.bar(w1_cnt.keys(), w1_cnt.values(), color="blue")
plt.subplot(2, 2, 2)
plt.title('distribution of W2')
plt.bar(w2_cnt.keys(), w2_cnt.values(), color="blue")
plt.subplot(2, 2, 3)
plt.title('distribution of z1')
plt.ylim(0, 1000)
plt.bar(z1_cnt.keys(), z1_cnt.values(), color="blue")
plt.subplot(2, 2, 4)
plt.title('distribution of z2')
plt.ylim(0, 100)
plt.bar(z2_cnt.keys(), z2_cnt.values(), color="blue")

#plt.xticks(np.unique(w1_cnt.keys()))  # x축 눈금을 고유값으로 설정
plt.show()




