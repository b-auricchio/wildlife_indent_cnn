from torchsummary import summary
from torchvision import models
from models import models
import torch
import argparse

parser = argparse.ArgumentParser(description='Training script', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('model', type=str, help='model name (REQUIRED)')
parser.add_argument('-n', type=int, help='depth scaling')
parser.add_argument('-k', type=int, help='width scaling')

args = parser.parse_args()

if args.model == 'wrn' and (args.n is None or args.k is None):
    parser.error('wrn requires -n and -k argument')


torch.cuda.empty_cache()
model = models.get_model(args, 3, 100).to('cuda')
summary(model, (3, 448, 448))
