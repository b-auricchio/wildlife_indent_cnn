from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import csv
import os

stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(232),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

test_transform = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

class Flowers(Dataset):
    def __init__(self, split, image_root, transform = None):
        self.num_classes = 102-10
        if split == 'train':
            self.data_path = './datasets/flowers_train.csv'
        if split == 'val':
            self.data_path = './datasets/flowers_val.csv'
        if split == 'test':
            self.data_path = './datasets/flowers_test.csv'

        self.image_root = image_root
        self.transform = transform

        self.data = []

        with open(self.data_path, newline='') as f:
            reader = csv.reader(f)
            for line in reader:
                self.data.append(line)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.data[index][0])
        label = int(self.data[index][1])-1 #classes start at 0 instead of 1

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        return image, label