import os

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

#hyperparameters
label_smoothing = 0.3
num_known_classes = 160

split_root = './datasets/litter'
root = './data/litter/'
image_root = root

stats = ((0.4337, 0.4380, 0.4429),(0.2283, 0.2284, 0.22851))

class Litter(Dataset):
    def __init__(self, transform = None):
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return 3499

    def __getitem__(self, index):
        filename = str(index+1).zfill(6) + '.jpg'
        image_path = os.path.join(self.image_root, filename)
        label = index #classes start at 0 instead of 1

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

def get_dataset(image_size):
    transform = transforms.Compose([
        transforms.Resize(int(image_size*1.2)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    return Litter(transform=transform)