from torchsummary import summary
from torchvision import models
from models import models
import torch
import argparse
from models.resnet import ResNet, ResBlock, ResBottleneckBlock

parser = argparse.ArgumentParser(description='Training script', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('model', type=str, help='model name (REQUIRED)')
parser.add_argument('-n', type=int, help='depth scaling')
parser.add_argument('-k', default=1.0, type=float, help='width scaling')

args = parser.parse_args()

if args.model == 'wrn' and (args.n is None or args.k is None):
    parser.error('wrn requires -n and -k argument')



torch.cuda.empty_cache()
model = models.get_model(args, 3, 100).to('cuda')
#model = ResNet(3, ResBlock, [2,2,2,2], useBottleneck=False, outputs=100, k=1.25).to('cuda')

summary(model, (3, 448, 448))
