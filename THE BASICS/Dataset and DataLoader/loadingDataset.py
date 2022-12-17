"""
pytorch에서는 모듈화를 위해 데이터관련 클래스 제공
Dataset은 샘플과 label들을 저장.
DataLoader는 데이터셋을 iterable로 감싸 샘플에 쉽게 반환할 수 있도록 함.
"""

import torch
from torch.utils.data import Dataset  # remember torch.utils.data
from torch.utils.data import DataLoader
from torchvision import datasets  # pytorch provide a pre-loaded datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data",  # path where data is stored
    train=True,
    download=True,
    transform=ToTensor()  # transform data to tensor
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
# if you run this code, download starts automatically


# let's check out samples in training data
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]  # Dataset can index like a list
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()


# While training a model, pass samples in "mini batches"
# reshuffle the data at every epoch to reduce model overfitting
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()  # first sample from the mini batches(64)
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
