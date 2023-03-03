import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import config as cfg

from torch_lr_finder import LRFinder
from datasets import flowers, cub
from utils import ToDeviceLoader
from models import models



device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
torch.cuda.empty_cache()


dataset = eval(cfg.dataset)

batch_size = cfg.batch_size
num_classes = dataset.num_known_classes
datasets = dataset.get_datasets(cfg.download)

train_dl = DataLoader(datasets['train'], batch_size=batch_size,shuffle=True)
val_dl = DataLoader(datasets['val'], batch_size=batch_size)


model = models.get_model(cfg, in_channels=3, num_classes=num_classes).to(device)
if cfg.load_dict == True:
    dictpath = './output/' + cfg.train_filename + '.pt'
    model.load_state_dict(torch.load(dictpath))


max_lr = dataset.eta
grad_clip = cfg.grad_clip
weight_decay = cfg.weight_decay
label_smoothing = dataset.label_smoothing

loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
optimiser = optim.Adam(model.parameters(),weight_decay=weight_decay,lr=max_lr)

lr_finder = LRFinder(model, optimiser, loss_fn, device=device)
lr_finder.range_test(train_dl, end_lr=100, num_iter=100)
lr_finder.plot() # to inspect the loss-learning rate graph
lr_finder.reset() # to reset the model and optimizer to their initial state


