import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functions import show_batch, ToDeviceLoader
from datasets.openflowers import Flowers, test_transform
from models.ResNet import ResNet, ResBlock, ResBottleneckBlock
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

test_data = Flowers(
    split = 'test',
    image_root = './data/flowers-102/jpg/',
    transform = test_transform
)

test_dl = DataLoader(test_data, batch_size=64,shuffle=False)
test_dl = ToDeviceLoader(test_dl, device)

model = ResNet(3, ResBlock, [3,4,6,3], useBottleneck=False, outputs=test_data.num_classes).to(device)
model.load_state_dict(torch.load('./dict/'+'resnet_20230219_013837'+'.pt'))

tau = 0.5

model.eval()
for X,y in test_dl:
    with torch.no_grad():
        pred = model(X)
        pred = F.softmax(pred, dim=1).cpu().detach().numpy()
        max_softmax = np.max(pred, axis=1)
        labels = y.cpu().detach().numpy()
        print(labels[1])
        
