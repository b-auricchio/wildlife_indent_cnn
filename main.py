import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102
import time
from IPython.display import display

from datasets.inaturalist import Birds, train_transform, test_transform
from functions import fit
from functions import ToDeviceLoader
from models.ResNet import ResNet, ResBlock, ResBottleneckBlock
import processing

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

batch_size = 32
accum_iter = 4
epochs = 90

train_data = Birds(
    split = 'train',
    transform = train_transform
)

test_data = Birds(
    split = 'val',
    transform = test_transform
)

train_dl = DataLoader(train_data, batch_size=batch_size,shuffle=True)
test_dl = DataLoader(test_data, batch_size=batch_size)

train_dl = ToDeviceLoader(train_dl,device)
test_dl = ToDeviceLoader(test_dl,device)

model = ResNet(3, ResBottleneckBlock, [3,4,6,3], useBottleneck=True, outputs=126).to(device)

max_lr=0.01
grad_clip = 0.1
weight_decay = 1e-4

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
optimiser = optim.Adam(model.parameters(),weight_decay=weight_decay,lr=max_lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=max_lr, steps_per_epoch=len(train_dl), epochs=epochs)

start_time = time.time()

history = fit(batch_size,epochs,train_dl,test_dl,model,loss_fn,optimiser,accum_iter,grad_clip=grad_clip, scheduler=scheduler)

print(f"--- {(time.time() - start_time):>0.1f} seconds ---")

history = processing.to_dataframe(history)

filename = model.name + '_' + time.strftime("%Y%m%d_%H%M%S")

history.to_csv((filename+'.csv'), encoding='utf-8', index=False)
torch.save(model.state_dict(), './dict/'+ filename +'.pt')

display(history)