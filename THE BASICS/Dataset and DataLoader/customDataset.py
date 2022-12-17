"""
커스텀 데이터셋은 Dataset클래스를 상속 받아 구현할 수 있다.
__init__ , __len__, __getitem__을 구현해야 한다.
"""

import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    # holding raw data (feature and label)
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    # __len__ have to return number of samples
    def __len__(self):
        return len(self.img_labels)

    # return one sample (with label) at the given index
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
