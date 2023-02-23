import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import time
from IPython.display import display
import config as cfg
import os

from datasets import flowers, cub
import utils
from utils import fit
from utils import ToDeviceLoader
from models import models
import misc

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

if cfg.DEBUG:
    epochs = cfg.debug_epochs
else: epochs = cfg.epochs

dataset = eval(cfg.dataset)

batch_size = dataset.batch_size
num_classes = dataset.num_known_classes
datasets = dataset.get_datasets(cfg.download)

train_dl = DataLoader(datasets['train'], batch_size=batch_size,shuffle=True)
val_dl = DataLoader(datasets['val'], batch_size=batch_size)
train_dl = ToDeviceLoader(train_dl,device)
val_dl = ToDeviceLoader(val_dl,device)

model = models.get_model(cfg.model, in_channels=3, num_classes=num_classes, width_scaling=cfg.width_scaling).to(device)

max_lr = dataset.eta
grad_clip = cfg.grad_clip
weight_decay = cfg.weight_decay
label_smoothing = dataset.label_smoothing

loss_fn = nn.CrossEntropyLoss(label_smoothing)
optimiser = optim.Adam(model.parameters(),weight_decay=weight_decay,lr=max_lr)
scheduler = utils.get_scheduler(optimiser, cfg)

print("\nTraining \n ----------------------")
print(f"Model: {cfg.model}")
print(f"Dataset: {cfg.dataset}")
print("\nHyperparameters \n ----------------------")
print(f"ETA: {max_lr}")
print(f"Batch size: {batch_size}")
print(f"Image size: {dataset.image_size}")
print("\n")

start_time = time.time()
history = fit(batch_size,epochs,train_dl,val_dl,model,loss_fn,optimiser,grad_clip=grad_clip, scheduler=scheduler, print_freq = cfg.print_freq)

print(f"--- {(time.time() - start_time):>0.1f} seconds ---")

history['time_elapsed'] = time.time() - start_time
history = misc.to_dataframe(history)

acc = history['accuracy'].tolist()[-1]*100

filename = f'{cfg.model}_{cfg.dataset}_width{cfg.width_scaling}_size{cfg.img_size}_{cfg.scheduler}'

try:
    history.to_csv(os.path.join(cfg.dict_path, filename +'.csv'), encoding='utf-8', index=False)
    torch.save(model.state_dict(), os.path.join(cfg.dict_path, filename +'.pt'))
    print(f'Saved to {cfg.dict_path}')
except:
    print('Local save directory not found!')
    pass

try:
    history.to_csv(os.path.join(cfg.cloud_dict_path, filename +'.csv'), encoding='utf-8', index=False)
    torch.save(model.state_dict(), os.path.join(cfg.cloud_dict_path ,filename +'.pt'))
    print(f'Saved to {cfg.cloud_dict_path}')
except:
    print('Cloud save directory not found!')
    pass

display(history)
misc.plot_history(history)