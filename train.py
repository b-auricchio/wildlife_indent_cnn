import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import time
from IPython.display import display
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
parser.add_argument('-l', dest='lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('-m', dest='randmag', type=int, default=2, help='magnitude of RandAugment')
parser.add_argument('--dataset', type=str, default='cub', help='dataset')
parser.add_argument('--download', type=bool, default=False, help='download dataset')
parser.add_argument('--scheduler', type=str, default='onecycle', help='scheduler')
parser.add_argument('--freq', type=int, default=50, help='print frequency')
parser.add_argument('--optim', type=str, default='adam', help='optimiser')

dict_path = './output'
cloud_dict_path = '../drive/MyDrive/RP3'

args = parser.parse_args()

if args.model == 'wrn' and (args.n is None or args.k is None):
    parser.error('wrn requires -n and -k argument')


from datasets import flowers, cub
import utils
from utils import fit
from utils import ToDeviceLoader
from models import models
import misc

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
torch.cuda.empty_cache()

dataset = eval(args.dataset)

batch_size = args.batch
num_classes = dataset.num_known_classes
datasets = dataset.get_datasets(args.img_size, args.randmag, download=args.download)

train_dl = DataLoader(datasets['train'], batch_size=batch_size,shuffle=True)
val_dl = DataLoader(datasets['val'], batch_size=batch_size)
train_dl = ToDeviceLoader(train_dl,device)
val_dl = ToDeviceLoader(val_dl,device)

model = models.get_model(args, in_channels=3, num_classes=num_classes).to(device)

epochs = args.epochs
lr = args.lr
grad_clip = 0.1
weight_decay = 1e-4
label_smoothing = dataset.label_smoothing

loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
if args.optim == 'adamw': optimiser = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay, amsgrad=True, betas=[0.9, 0.99])
if args.optim == 'adam': optimiser = optim.Adam(model.parameters(),weight_decay=weight_decay,lr=lr)
if args.optim == 'sgd': optimiser = optim.SGD(model.parameters(),lr=args.lr, momentum=0.9,weight_decay=weight_decay)

scheduler = utils.get_scheduler(optimiser, args, epochs, len(train_dl))

print("\nTraining \n ----------------------")
print(f"Model: {args.model}")
print(f"Dataset: {args.dataset}")
print(f"Epochs: {args.epochs}")
print(f"Scheduler: {args.scheduler}")
print(f"Optimiser: {args.optim}")
print("\nHyperparameters \n ----------------------")
print(f"LR: {lr}")
print(f"Batch size: {batch_size}")
print(f"Image size: {args.img_size}")
print("\n")

start_time = time.time()
history = fit(batch_size,epochs,train_dl,val_dl,model,loss_fn,optimiser,args,grad_clip=grad_clip, scheduler=scheduler, print_freq=args.freq)

print(f"--- {(time.time() - start_time):>0.1f} seconds ---")

history['time_elapsed'] = time.time() - start_time
history = misc.to_dataframe(history)

acc = history['val_acc'].tolist()[-1]*100

filename = f'{args.model}_{args.dataset}_size{args.img_size}_{args.scheduler}'

try:
    history.to_csv(os.path.join(dict_path, filename +'.csv'), encoding='utf-8', index=False)
    torch.save(model.state_dict(), os.path.join(dict_path, filename +'.pt'))
    print(f'Saved to {dict_path}')
except:
    print('Local save directory not found!')
    pass

try:
    history.to_csv(os.path.join(cloud_dict_path, filename +'.csv'), encoding='utf-8', index=False)
    torch.save(model.state_dict(), os.path.join(cloud_dict_path ,filename +'.pt'))
    print(f'Saved to {cloud_dict_path}')
except:
    print('Cloud save directory not found!')
    pass

display(history)
misc.plot_history(history)