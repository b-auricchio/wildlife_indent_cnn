from mirror import mirror
from mirror.visualisations.web import *
from PIL import Image
from models.models import get_model
from torchvision.transforms import ToTensor, Resize, Compose
import os
import torch
from datasets import seals, litter, cub

model_name = 'resnet34'
k = 1
img_size = 224
dataset = 'cub'

dict_path = f'.\\output\\{dataset}'
fig_path = os.path.join(dict_path, 'figures')
dict_name = f'{model_name}_{k}'
path = os.path.join(dict_path, dict_name + '.pt')

model = get_model(model_name, k, 3, cub.num_known_classes)
model.load_state_dict(torch.load(path))

cub_data = cub.get_datasets(img_size, 0, download=False)
seal, label = cub_data['test_known'][1535]

litter_data = litter.get_dataset(img_size)
beach, label = litter_data[24]

# call mirror with the inputs and the model
mirror([seal], model, visualisations=[GradCam])