from torchsummary import summary
from torchvision import models
from models import models
import torch
import argparse

parser = argparse.ArgumentParser(description='Training script', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('model', type=str, help='model name (REQUIRED)')

args = parser.parse_args()


torch.cuda.empty_cache()
model = models.get_model(args, 3, 100).to('cuda')
summary(model, (3, 448, 448))
