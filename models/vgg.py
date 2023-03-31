from torchvision.models import vgg11, vgg13, vgg16, vgg19
import torch.nn as nn
import torch.nn.functional as F

class VGG(nn.Module):
        def __init__(self, num_classes, init_weights=True):
            super(VGG, self).__init__()
            self.model = vgg11()
            in_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(in_ftrs, num_classes)
            
            if init_weights:
                self._initialize_weights(self.model)
    
        def forward(self, x):
            x = self.model(x)
            return x
    
        def _initialize_weights(self, model):
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)