#import Fashion-MNIST data set with TorchVision
# which is Zalando's black&white image dataset

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use('Agg')

training_data = datasets.FashionMNIST(
    root='data', # path that learning/testing data is saved
    train=True,  # specify whether the dataset is for learning or testing
    download=True, # if data does not exist in root, download it
    transform=ToTensor() # transform and target_transform specify feature, label
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: 'T-shirt',
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
    print(sample_idx)
plt.show()

#print(training_data[0])  # This should print a tensor and the corresponding label