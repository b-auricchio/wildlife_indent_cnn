import os

from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import tarfile
from torchvision import transforms
import csv
from PIL import Image

#hypermarameters
label_smoothing = 0
image_size = 224
lr = 1e-2
batch_size = 64
num_known_classes = 92

root = './data/flowers102/'
image_root = os.path.join(root, 'jpg')

def download_data():
    img_filename = '102flowers.tgz'
    labels_filename = 'imagelabels.mat'
    image_url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
    labels_url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat'
    download_url(image_url, root, img_filename)

    with tarfile.open(os.path.join(root, img_filename), "r:gz") as tar:
        tar.extractall(path=root)

    download_url(labels_url, root, labels_filename)

stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
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
    def __init__(self, split, transform = None):
        self.num_classes = 102-10
        self.split = split
        if split == 'train':
            self.data_path = os.path.join(root,"train.csv")
        if split == 'val':
            self.data_path = os.path.join(root,"val.csv")
        if split == 'test_known':
            self.data_path = os.path.join(root,"test_known.csv")
        if split == 'test_unknown':
            self.data_path = os.path.join(root,"test_unknown.csv")

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
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

def get_datasets(download=True):
    if download:
        download_data()

    train_dataset = Flowers(split = 'train', transform=train_transform)
    val_dataset = Flowers(split = 'val', transform=test_transform)
    test_dataset_known = Flowers(split = 'test_known', transform=test_transform)
    test_dataset_unknown = Flowers(split = 'test_unknown', transform=test_transform)
    
    
    all_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test_known': test_dataset_known,
        'test_unknown': test_dataset_unknown,
    }

    return all_datasets