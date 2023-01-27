import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as tt
from torchvision.datasets import CIFAR100
from torch.utils.data.dataloader import DataLoader
import time
import pandas as pd
import csv
from IPython.display import display


from functions import fit
from functions import ToDeviceLoader
from models.ResNet50 import Model
import processing


start_time = time.time()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

num_classes = 100
batch_size = 64
epochs = 1

#LOADING DATA
stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
traintransform = tt.Compose([
    tt.RandomHorizontalFlip(),
    tt.RandomCrop(32,padding=4,padding_mode="reflect"),
    tt.ToTensor(),
    tt.Normalize(*stats)    
])

valtransform = tt.Compose([
    tt.ToTensor(),
    tt.Normalize(*stats) 
])


train_data = CIFAR100(
    download=False,
    train=True,
    root='data',
    transform=traintransform
)

test_data = CIFAR100(
    download=False,
    train=False,
    root='data',
    transform=valtransform
)

train_dl = DataLoader(train_data, batch_size, shuffle=True)
test_dl = DataLoader(test_data, batch_size, shuffle=True)

train_dl = ToDeviceLoader(train_dl,device)
test_dl = ToDeviceLoader(test_dl,device)

model = Model(3,100).to(device)
max_lr=0.01
grad_clip = 0.1
weight_decay = 1e-4

loss_fn = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(),weight_decay=weight_decay,lr=max_lr)
scheduler = scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=max_lr, steps_per_epoch=len(train_dl), epochs=epochs)

history = fit(batch_size,epochs,train_dl,test_dl,model,loss_fn,optimiser,scheduler,grad_clip)

print(f"--- {(time.time() - start_time):>0.1f} seconds ---")

history = processing.to_dataframe(history)

filename = Model.name + '_' + time.strftime("%Y%m%d_%H%M%S")

history.to_csv((filename+'.csv'), encoding='utf-8', index=False)
torch.save(model.state_dict(), './dict/'+ filename +'.pt')

display(history)

