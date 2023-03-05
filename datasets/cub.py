import os

from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import tarfile
from torchvision import transforms
import csv
from PIL import Image
import config as cfg

#hyperparameters
label_smoothing = 0.3
batch_size = 32
image_size = cfg.img_size
num_known_classes = 160

root = './data/cub/'
image_root = os.path.join(root, 'CUB_200_2011/images')

def download_data():
    filename = 'CUB_200_2011.tgz'
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
    download_url(url, root, filename)

    with tarfile.open(os.path.join(root, filename), "r:gz") as tar:
        tar.extractall(path=root)

stats = ((0.4893, 0.5014, 0.4416), (0.2284, 0.2235, 0.2612))
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(int(image_size*1.2)),
    transforms.RandomCrop(image_size),
    transforms.RandAugment(num_ops=2, magnitude=2),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

test_transform = transforms.Compose([
    transforms.Resize(int(image_size*1.2)),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

class CUB(Dataset):
    def __init__(self, split, transform = None):
        self.num_classes = num_known_classes
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

    train_dataset = CUB(split = 'train', transform=train_transform)
    val_dataset = CUB(split = 'val', transform=test_transform)
    test_dataset_known = CUB(split = 'test_known', transform=test_transform)
    test_dataset_unknown = CUB(split = 'test_unknown', transform=test_transform)
    
    
    all_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test_known': test_dataset_known,
        'test_unknown': test_dataset_unknown,
    }

    return all_datasets