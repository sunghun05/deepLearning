#import sys, os
import numpy as np
import pickle
#from keras.src.backend.jax.nn import softmax
# sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image
from modules.neural_network import NeuralNetwork


def img_show(img):
    pil_image = Image.fromarray(np.uint8(img))
    pil_image.show()

def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=True, one_hot_label=False)
    print(x_train)
    return x_test, t_test, x_train

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = Net.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = Net.sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = Net.softmax(a3)

    return y

if __name__ == '__main__':
    Net = NeuralNetwork()
    x, t, xt = get_data()
    network = init_network()
    img = xt[0]
    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)
        if p == t[i]:
            accuracy_cnt += 1

    print(img)
    print(img.shape)
    img *= 255
    # print(img)
    print(img.reshape(28, 28).shape)
    img_show(img.reshape(28, 28))
    y1 = predict(network, img)
    print(f"y1 : {np.argmax(y1)}")

    print("Accuracy: " + str(float(accuracy_cnt)/len(x)))