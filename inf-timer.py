import pandas as pd
import misc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import models
from tqdm import tqdm
import argparse
from datasets import cub, flowers
import time

parser = argparse.ArgumentParser(description='Open-set testing script', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('model', type=str, help='model name (REQUIRED)')
parser.add_argument('img_size', type=int, help='image size (REQUIRED)')
parser.add_argument('filename', type=str, help='filename of dict (REQUIRED)')
parser.add_argument('-b', type=int, default = 128, help='batch size')
parser.add_argument('-n', type=int, help='depth scaling')
parser.add_argument('-k', type=int, help='width scaling')
parser.add_argument('--dataset', type=str, default='cub', help='dataset')

args = parser.parse_args()

if args.model == 'wrn' and (args.n is None or args.k is None):
    parser.error('wrn requires -n and -k argument')

filename = args.filename

dataset = eval(args.dataset)
datasets = dataset.get_datasets(args.img_size, 0, download=False)

batch_size = args.b

known = DataLoader(datasets['test_known'], batch_size=batch_size,shuffle=False)
unknown = DataLoader(datasets['test_unknown'], batch_size=batch_size,shuffle=False)

model = models.get_model(args, in_channels=3, num_classes=dataset.num_known_classes).to('cpu')

dictpath = './output/' + filename + '.pt'
model.load_state_dict(torch.load(dictpath))

start_time = time.time()
print(f"Inferring on {len(known.dataset)} images" )
for X,y in tqdm(known):
    with torch.no_grad():
        pred = model(X)
        pred = F.softmax(pred, dim=1)
        max_softmax = torch.max(pred, dim=1)[0]
time_elapsed = time.time() - start_time
print(f"--- {(time_elapsed):>0.1f} seconds ---")
print(f"=> {time_elapsed/len(known.dataset)} seconds per image")

 