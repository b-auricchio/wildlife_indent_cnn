from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torch
import csv

stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

test_transform = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])


class BinaryBirds(Dataset):
    def __init__(self, split, transform = None):
        if split == 'train':
            self.data_path = 'inat_birds_train.csv'
        if split == 'val':
            self.data_path = 'inat_birds_test.csv'

        self.transform = transform

        self.data = []

        with open(self.data_path, newline='') as f:
            reader = csv.reader(f)
            for line in reader:
                self.data.append(line)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data[index][0]
        label = int(self.data[index][1])

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        return image, label



