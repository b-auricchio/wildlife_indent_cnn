import pandas as pd
import misc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import show_batch, ToDeviceLoader
from models import models
from datasets import cub, flowers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import config as cfg
from tqdm import tqdm

filename = cfg.filename

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

dataset = eval(cfg.dataset)
datasets = dataset.get_datasets(download=False)

known = DataLoader(datasets['test_known'], batch_size=128,shuffle=False)
unknown = DataLoader(datasets['test_unknown'], batch_size=128,shuffle=False)
known = ToDeviceLoader(known, device)
unknown = ToDeviceLoader(unknown, device)

model = models.get_model(cfg.model, in_channels=3, num_classes=dataset.num_known_classes).to(device)
dictpath = './output/' + filename + '.pt'
model.load_state_dict(torch.load(dictpath))

def get_binary_predictions_softmax(tau):
    model.eval()

    in_softmax_sum = 0
    out_softmax_sum = 0

    binary_preds = []
    targets = []

    for X,y in tqdm(known):
        with torch.no_grad():
            pred = model(X)
            pred = F.softmax(pred, dim=1)
            max_softmax = torch.max(pred, dim=1)[0]
            bin = [1 if i >= tau else 0 for i in max_softmax] # 1=correctly identified as in-set (sm>tau)
            in_softmax_sum += torch.sum(max_softmax).item()
            binary_preds.extend(bin)
            targets.extend([1]*len(bin))
            
    for X,y in tqdm(unknown):
        with torch.no_grad():
            pred = model(X)
            pred = F.softmax(pred, dim=1)
            max_softmax = torch.max(pred, dim=1)[0]
            bin = [1 if i >= tau else 0 for i in max_softmax] # 0=correctly identified as out-set (sm<tau)
            out_softmax_sum += torch.sum(max_softmax).item()
            binary_preds.extend(bin)
            targets.extend([0]*len(bin))

    in_mean = in_softmax_sum/len(datasets['test_known'])
    out_mean = out_softmax_sum/len(datasets['test_unknown'])

    return binary_preds, targets, in_mean, out_mean

def get_binary_predictions_logits(tau):
    model.eval()

    in_logits_sum = 0
    out_logits_sum = 0

    binary_preds = []
    targets = []

    for X,y in tqdm(known):
        with torch.no_grad():
            pred = model(X)
            max_logits = torch.max(pred, dim=1)[0]
            bin = [1 if i >= tau else 0 for i in max_logits] # 1=correctly identified as in-set (sm>tau)
            in_logits_sum += torch.sum(max_logits).item()
            binary_preds.extend(bin)
            targets.extend([1]*len(bin))
            
    for X,y in tqdm(unknown):
        with torch.no_grad():
            pred = model(X)
            max_logits = torch.max(pred, dim=1)[0]
            bin = [1 if i >= tau else 0 for i in max_logits] # 0=correctly identified as out-set (sm<tau)
            out_logits_sum += torch.sum(max_logits).item()
            binary_preds.extend(bin)
            targets.extend([0]*len(bin))

    in_mean = in_logits_sum/len(datasets['test_known'])
    out_mean = out_logits_sum/len(datasets['test_unknown'])

    return binary_preds, targets, in_mean, out_mean

def roc(predictions, targets, plot=False):
    false_positive_rate, true_positive_rate, threshold = roc_curve(targets, predictions)

    auroc = roc_auc_score(targets, predictions)

    if plot == True:
        plt.title('Receiver Operating Characteristic')
        plt.plot(false_positive_rate, true_positive_rate)
        plt.plot([0, 1], ls="--")
        plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        plt.show()

    return auroc

for tau in np.linspace(cfg.tau_min, cfg.tau_max, cfg.tau_num_steps):
    print(f'\n Threshold: {tau}\n ----------------------')
    predictions, targets, in_mean, out_mean = get_binary_predictions_softmax(tau)
    auroc = roc(targets, predictions)
    print('AUROC: ', auroc)

print(f'In-set mean: {in_mean}')
print(f'Out-set mean: {out_mean}')