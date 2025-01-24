# # funcs.py
# import numpy as np

# def cee(y, t):
#     # if data is not for batch
#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)

#     batch_size = y.shape[0]
#     return -np.sum(t * np.log(y + 1e-7)) / batch_size

# def sigmoid_batch(x):
#     y = 1 / (1 + np.exp(-x))
#     batch_size = x.shape[0]
#     return y / batch_size

# def softmax(x):
#     x = x - np.max(x, axis=-1, keepdims=True)  # prevent from being overflowed
#     return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

# def numerical_gradient(f, x):
#     h = 1e-3  # 0.001
#     grad = np.zeros_like(x)

#     it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
#     print(f"it : {it}")

#     while not it.finished:
#         idx = it.multi_index
#         tmp_val = x[idx]
        
#         x[idx] = float(tmp_val) + h
#         fxh1 = f(x)  # f(x+h)
#         #print(fxh1)
#         # fxh1 = f(x[idx])  # f(x+h)

#         x[idx] = float(tmp_val) - h
#         fxh2 = f(x)  # f(x-h)
#         #print(fxh2)
#         # fxh2 = f(x[idx])  # f(x-h)
#         grad[idx] = (fxh1 - fxh2) / (2 * h)

#         x[idx] = tmp_val  # 값 복원
#         it.iternext()
#     print("gradient : ")
#     print(grad)
#     print(f"^^^^^^^^^^^^^^{grad.shape}")

#     return grad