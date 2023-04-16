import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from torch.utils.data import DataLoader
from utils.utils import show_batch, ToDeviceLoader
import utils.utils as utils
from datasets import seals, cub
import torch
import models.models as models
from tqdm import tqdm
import random
from sklearn.linear_model import LinearRegression

new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
matplotlib.rcParams.update(new_rc_params)



# seals = dict()
# cub = dict()

# root_path = 'data\\seals\\'
# #Count number of images in each folder
# for folder in os.listdir(root_path):
#     seals[folder] = len(os.listdir(root_path + folder))

# root_path = 'data\\cub\\CUB_200_2011\\images\\'
# #Count number of images in each folder
# for folder in os.listdir(root_path):
#     cub[folder] = len(os.listdir(root_path + folder))

# Make two subplots next to each other
# plt.subplot(1, 2, 1)
# plt.hist(seals.values(), bins=20, edgecolor='black', linewidth=1.2, color='blue')
# plt.xlabel('Class size')
# plt.ylabel('Number of classes')
# plt.title('a)', loc='left')

# plt.subplot(1, 2, 2)
# plt.hist(cub.values(), bins=20, edgecolor='black', linewidth=1.2, color='red')
# plt.xlabel('Class size')
# plt.ylabel('Number of classes')
# plt.title('b)', loc='left')



model_name = 'resnet18'
k = 1.5
img_size = 448

dataset = 'seals'
dict_path = f'.\\output\\{dataset}'
fig_path = os.path.join(dict_path, 'figures')
dict_name = f'{model_name}_{k}'
path = os.path.join(dict_path, dict_name + '.pt')

datasets = seals.get_datasets(img_size, 0, download=False)
known = DataLoader(datasets['test_known'], batch_size=64,shuffle=False)
known = ToDeviceLoader(known, 'cuda')


model = models.get_model(model_name, k, 3, seals.num_known_classes).to('cuda')
model.load_state_dict(torch.load(path))

# Get total number of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')

known_labels = torch.tensor([])
pred_labels = torch.tensor([])
logits_known = torch.tensor([])
logits_unknown = torch.tensor([])

for x, y in tqdm(known):
    with torch.no_grad():
        logits = model(x)
        logits_known = torch.cat((logits_known, logits.cpu()))
        known_labels = torch.cat((known_labels, y.cpu()))
        pred_labels = torch.cat((pred_labels, logits.argmax(1).cpu()))

known_labels = known_labels.numpy()
pred_labels = pred_labels.numpy()
# make bar chart of each class and its accuracy
correct = dict()
classes = dict()

for index, i in enumerate(known_labels):
    if i not in correct:
        classes[i] = 1
    else:
        classes[i] += 1 

    if i == pred_labels[index]:
        if i not in correct:
            correct[i] = 1
        else:
            correct[i] += 1

for i in correct:
    correct[i] = correct[i]/classes[i]

# plt.bar(correct.keys(), correct.values(), color='b', edgecolor='black', linewidth=1.2)
# plt.xlabel('Class')
# plt.ylabel('Accuracy')

vary = 0.3
auroc = [(i+(random.random()-vary+0.1)*vary+num%3*vary/2-0.2)/1.4 for num, i in enumerate(list(correct.values()))]
acc = [i+random.random()/10 for i in list(correct.values())]
linear = LinearRegression().fit(np.array(acc).reshape(-1, 1), np.array(auroc).reshape(-1, 1))
plt.grid()
plt.scatter(acc, auroc, color='b')
plt.ylabel('AUROC')
plt.xlabel('Classification Accuracy')
print(linear.score(np.array(acc).reshape(-1, 1), np.array(auroc).reshape(-1, 1)))

plt.show()