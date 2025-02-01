import matplotlib.pyplot as plt
import numpy as np
tmp0 = np.zeros(10001)
tmp1 = np.zeros(344)
with open('loss_log.rtf', 'r') as f:
    li = f.read().split('loss : ')

    tmp0 = tmp0.astype(type(''))
    for i in range(len(li)):
        li[i] = li[i].split('\\par\n')
    for i in range(len(li)):
        tmp0[i] = li[i][0]
    tmp0 = list(tmp0)
    tmp0.pop(0)
    tmp0 = list(map(float, tmp0))
with open('log_gradient.rtf', 'r') as f:
    li = f.read().split('loss : ')

    tmp1 = tmp1.astype(type(''))
    for i in range(len(li)):
        li[i] = li[i].split('\\par\n')
    for i in range(len(li)):
        tmp1[i] = li[i][0]
    tmp1 = list(tmp1)
    tmp1.pop(0)
    tmp1 = list(map(float, tmp1))
    print(tmp1)

plt.title('change in Loss')
plt.plot(tmp0, label='by backpropagation')
plt.plot(tmp1, color='r', label='by gradient')
plt.legend()
plt.xlim(0, 10000)
plt.ylim(-0.1, 2.5)
plt.xlabel('number')
plt.ylabel('LOSS')
plt.show()