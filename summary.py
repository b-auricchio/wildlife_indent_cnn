from torchsummary import summary
from torchvision import models
import config as cfg
from models import models
import torch

torch.cuda.empty_cache()
model = models.get_model(cfg, 3, 100).to('cuda')
summary(model, (3, 448, 448))
