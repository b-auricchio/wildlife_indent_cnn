from torchsummary import summary
from torchvision import models
import config as cfg
from models import models

model = models.get_model(cfg, 3, 100).to('cuda')
summary(model, (3, 224, 224))
