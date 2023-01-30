from torchvision import transforms
from torch.utils.data import Dataset
from scipy.io import loadmat
import PIL.Image as Image
import torch

root = './data/102flowers/'
labels_root = './data/102flowers/imagelabels.mat'
images_root = './data/102flowers/jpg/'

stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(250),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

class Flowers102(Dataset):
    def __init__(self, labels_root, images_root_dir, transform):

        self.labels_root = labels_root
        self.images_root_dir = images_root_dir
        self.transform = transform

        labels = loadmat(labels_root)
        self.labels = labels['labels'].flatten()

    def __len__(self):
        return 8189

    def indextopath(self, i):
        index = str(i+1).zfill(5)
        filename = 'image_'+ index + '.jpg'
        return self.images_root_dir + filename

    def __getitem__(self, index):
        root = self.indextopath(index)
        image = Image.open(root)
        image = transform(image)
        label = torch.tensor(self.labels[index])
        return image, label