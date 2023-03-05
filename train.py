import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import time
from IPython.display import display
import config as cfg
import os
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Training script', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('model', type=str, help='model name (REQUIRED)')
parser.add_argument('img_size', type=int, help='image size (REQUIRED)')
parser.add_argument('-n', type=int, help='depth scaling')
parser.add_argument('-k', type=int, help='width scaling')
parser.add_argument('-b', dest='batch', type=int, default=32, help='batch size')
parser.add_argument('-e', dest='epochs', type=int,default=100, help='number of epochs')
parser.add_argument('-l', dest='lr', type=float, default=1.5e-3, help='learning rate')
parser.add_argument('--dataset', type=str, default='cub', help='dataset')
parser.add_argument('--download', type=bool, default=False, help='download dataset')
args = parser.parse_args()

if args.model == 'wrn' and (args.n is None or args.k is None):
    parser.error('wrn requires -n and -k argument')


cfg.model = args.model
cfg.img_size = args.img_size
cfg.width_scaling = args.k
cfg.depth_scaling = args.n

cfg.batch_size = args.batch
cfg.epochs = args.epochs
cfg.lr = args.lr
cfg.dataset = args.dataset
cfg.download = args.download


from datasets import flowers, cub
import utils
from utils import fit
from utils import ToDeviceLoader
from models import models
import misc

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
torch.cuda.empty_cache()

if cfg.DEBUG:
    epochs = cfg.debug_epochs
else: epochs = cfg.epochs

dataset = eval(cfg.dataset)

batch_size = cfg.batch_size
num_classes = dataset.num_known_classes
datasets = dataset.get_datasets(cfg.download)

train_dl = DataLoader(datasets['train'], batch_size=batch_size,shuffle=True)
val_dl = DataLoader(datasets['val'], batch_size=batch_size)
train_dl = ToDeviceLoader(train_dl,device)
val_dl = ToDeviceLoader(val_dl,device)

model = models.get_model(cfg, in_channels=3, num_classes=num_classes).to(device)
if cfg.load_dict == True:
    dictpath = './output/' + cfg.train_filename + '.pt'
    model.load_state_dict(torch.load(dictpath))


lr = cfg.lr
grad_clip = cfg.grad_clip
weight_decay = cfg.weight_decay
label_smoothing = dataset.label_smoothing

loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
optimiser = optim.Adam(model.parameters(),weight_decay=weight_decay,lr=lr)

scheduler = utils.get_scheduler(optimiser, cfg, epochs, len(train_dl))

print("\nTraining \n ----------------------")
print(f"Model: {cfg.model}")
print(f"Dataset: {cfg.dataset}")
print(f"Epochs: {cfg.epochs}")
print(f"Scheduler: {cfg.scheduler}")
print("\nHyperparameters \n ----------------------")
print(f"LR: {lr}")
print(f"Batch size: {batch_size}")
print(f"Image size: {dataset.image_size}")
print("\n")

start_time = time.time()
history = fit(batch_size,epochs,train_dl,val_dl,model,loss_fn,optimiser,cfg,grad_clip=grad_clip, scheduler=scheduler, print_freq=cfg.print_freq)

print(f"--- {(time.time() - start_time):>0.1f} seconds ---")

history['time_elapsed'] = time.time() - start_time
history = misc.to_dataframe(history)

acc = history['val_acc'].tolist()[-1]*100

filename = f'{cfg.model}_{cfg.dataset}_size{cfg.img_size}_{cfg.scheduler}'
if cfg.load_dict == True:
    filename = filename + '_tuned'

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