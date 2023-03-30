import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import argparse
from torch_lr_finder import LRFinder
from datasets import flowers, cub, seals
from utils.utils import ToDeviceLoader
from models import models

parser = argparse.ArgumentParser(description='Training script', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('model', type=str, help='model name (REQUIRED)')
parser.add_argument('img_size', type=int, help='image size (REQUIRED)')
parser.add_argument('-n', type=int, help='depth scaling')
parser.add_argument('-k', type=int, default=1,help='width scaling')
parser.add_argument('-l', dest='lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('-b', dest='batch', type=int, default=32, help='batch size')
parser.add_argument('--dataset', type=str, default='cub', help='dataset')
parser.add_argument('--download', type=bool, default=False, help='download dataset')
parser.add_argument('--optim', type=str, default='adam', help='optimiser')

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
torch.cuda.empty_cache()


dataset = eval(args.dataset)

batch_size = 32
num_classes = dataset.num_known_classes
datasets = dataset.get_datasets(args.img_size, 1, args.download)

train_dl = DataLoader(datasets['train'], batch_size=batch_size,shuffle=True)
val_dl = DataLoader(datasets['val'], batch_size=batch_size)


model = models.get_model(args.model, args.k, in_channels=3, num_classes=num_classes).to(device)

max_lr = args.lr
grad_clip = 0.1
weight_decay = 1e-4
label_smoothing = dataset.label_smoothing
loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
if args.optim == 'adamw': optimiser = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay, amsgrad=True, betas=[0.9, 0.99])
if args.optim == 'adam': optimiser = optim.Adam(model.parameters(),weight_decay=weight_decay,lr=args.lr)
if args.optim == 'sgd': optimiser = optim.SGD(model.parameters(),lr=args.lr, momentum=0.9,weight_decay=weight_decay)

lr_finder = LRFinder(model, optimiser, loss_fn, device=device)
lr_finder.range_test(train_dl, end_lr=100, num_iter=100)
lr_finder.plot() # to inspect the loss-learning rate graph
lr_finder.reset() # to reset the model and optimizer to their initial state