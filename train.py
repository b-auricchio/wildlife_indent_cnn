import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import time
from IPython.display import display

from datasets import flowers
from functions import fit
from functions import ToDeviceLoader
from models.ResNet import ResNet, ResBlock, ResBottleneckBlock
import processing

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

batch_size = 64
epochs = 50

num_classes = flowers.num_known_classes

datasets = flowers.get_datasets()

train_dl = DataLoader(datasets['train'], batch_size=batch_size,shuffle=True)
val_dl = DataLoader(datasets['val'], batch_size=batch_size)

train_dl = ToDeviceLoader(train_dl,device)
val_dl = ToDeviceLoader(val_dl,device)

model = ResNet(3, ResBlock, [3,4,6,3], useBottleneck=False, outputs=num_classes).to(device)

max_lr=0.01
grad_clip = 0.1
weight_decay = 1e-4

loss_fn = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(),weight_decay=weight_decay,lr=max_lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=max_lr, steps_per_epoch=len(train_dl), epochs=epochs)

start_time = time.time()
history = fit(batch_size,epochs,train_dl,val_dl,model,loss_fn,optimiser,grad_clip=grad_clip, scheduler=scheduler)

print(f"--- {(time.time() - start_time):>0.1f} seconds ---")

history['time_elapsed'] = time.time() - start_time
history = processing.to_dataframe(history)

filename = model.name + '_' + time.strftime("%Y%m%d_%H%M%S")

history.to_csv(('./output/'+filename+'.csv'), encoding='utf-8', index=False)
torch.save(model.state_dict(), './dict/'+ filename +'.pt')

display(history)