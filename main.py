import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100, Flowers102
from torch.utils.data.dataloader import DataLoader
import time
from IPython.display import display


from functions import fit
from functions import ToDeviceLoader
from models.ResNet import ResNet, ResBlock, ResBottleneckBlock
import processing


start_time = time.time()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

num_classes = 102
batch_size = 64
epochs = 50

stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(250),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

train_data = Flowers102(
    root = './data/',
    download = False,
    split = 'test',
    transform=transform
    )

val_data = Flowers102(
    root = './data/',
    download = False,
    split = 'val',
    transform=transform
    )



train_dl = DataLoader(train_data, batch_size=batch_size,shuffle=True)
test_dl = DataLoader(val_data, batch_size=batch_size)

train_dl = ToDeviceLoader(train_dl,device)
test_dl = ToDeviceLoader(test_dl,device)

model = ResNet(3, ResBlock, [3,4,6,3], useBottleneck=False, outputs=102).to(device)

max_lr=0.01
grad_clip = 0.1
weight_decay = 1e-4

loss_fn = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(),weight_decay=weight_decay,lr=max_lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=max_lr, steps_per_epoch=len(train_dl), epochs=epochs)

history = fit(batch_size,epochs,train_dl,test_dl,model,loss_fn,optimiser,grad_clip=grad_clip, scheduler=scheduler)

print(f"--- {(time.time() - start_time):>0.1f} seconds ---")

history = processing.to_dataframe(history)

filename = model.name + '_' + time.strftime("%Y%m%d_%H%M%S")

history.to_csv((filename+'.csv'), encoding='utf-8', index=False)
torch.save(model.state_dict(), './dict/'+ filename +'.pt')

display(history)