import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import show_batch, ToDeviceLoader
from models import models
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm
import argparse
from datasets import cub, flowers

# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='Open-set testing script', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-model', type=str, help='model name')
parser.add_argument('-k', default=1.0, type=float, help='width scaling')
args = parser.parse_args()

# if args.model == 'wrn' and (args.n is None or args.k is None):
#     parser.error('wrn requires -n and -k argument')



filenames = ['448/resnet18_cub_size448_onecycle_k1.0_m15', '448/resnet34_cub_size448_onecycle_k1.0', '448/resnet50_cub_size448_onecycle_k1.0', '448/resnet101_cub_size448_onecycle_k1.0_m15']
ks = [1,1,1,1]
model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101']


# filenames = ['224/resnet18_cub_size224_onecycle', '224/resnet18_cub_size224_onecycle_k1.5_m15', '224/resnet18_cub_size224_onecycle_k2.0_m15', '224/resnet18_cub_size224_onecycle_k2.5_m15']
# ks = [1, 1.5, 2, 2.5]
# model_names = ['resnet18','resnet18', 'resnet18','resnet18']

img_size = 448
dataset = 'cub'

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

dataset = eval(dataset)
datasets = dataset.get_datasets(img_size, 0, download=False)

known = DataLoader(datasets['test_known'], batch_size=64,shuffle=False)
unknown = DataLoader(datasets['test_unknown'], batch_size=64,shuffle=False)
known = ToDeviceLoader(known, device)
unknown = ToDeviceLoader(unknown, device)

ax = plt.subplot(1,1,1)

for [filename, model_name, k] in zip(filenames, model_names, ks):
    args.model = model_name
    args.k = k
    model = models.get_model(args, in_channels=3, num_classes=dataset.num_known_classes).to(device)
    dictpath = './output/' + filename + '.pt'
    model.load_state_dict(torch.load(dictpath))
    print(f"{args.model} k{args.k} loaded!\nGetting softmax:")

    sm_known = torch.tensor([])
    sm_unknown = torch.tensor([])
    labels = torch.tensor([])

    for X,y in tqdm(known):
        with torch.no_grad():
            pred = F.softmax(model(X), dim=1)
            pred = torch.max(pred, dim=1)[0]
            sm_known = torch.cat((sm_known, pred.cpu()))
            labels = torch.cat((labels, torch.ones(y.size(dim=0))))
            
    for X,y in tqdm(unknown):
        with torch.no_grad():
            pred = F.softmax(model(X), dim=1)
            pred = torch.max(pred, dim=1)[0]
            sm_unknown = torch.cat((sm_unknown, pred.cpu()))
            labels = torch.cat((labels, torch.zeros(y.size(dim=0))))

    sm = torch.cat((sm_known,sm_unknown))

    labels = labels.numpy()
    sm = sm.numpy()

    #positive = in set, negative = out-of-set

    fpr, tpr, thresholds = roc_curve(labels, sm)
    auroc = roc_auc_score(labels, sm)
    print('AUROC: ', auroc)

    plt.plot(fpr, tpr)


ax.legend([model_names[0]+'_'+str(ks[0]), model_names[1]+'_'+str(ks[1]), model_names[2]+'_'+str(ks[2]), model_names[3]+'_'+str(ks[3])])


plt.title('Receiver Operating Characteristic')
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
ax.set_aspect('equal', adjustable='box')
# plt.savefig('roc_curve.pgf')
plt.show()

    




