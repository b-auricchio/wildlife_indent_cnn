from models.resnet import ResNet, ResBlock, ResBottleneckBlock
from models.wideresnet import WideResNet

def get_model(cfg, in_channels, num_classes):
    name = cfg.model
    try:
        #ResNets 
        if name == 'resnet18':
            model = ResNet(in_channels, ResBlock, [2,2,2,2], useBottleneck=False, outputs=num_classes)
        if name == 'resnet34':
            model = ResNet(3, ResBlock, [3,4,6,3], useBottleneck=False, outputs=num_classes)
        if name == 'resnet50':
            model = ResNet(3, ResBottleneckBlock, [3,4,6,3], useBottleneck=True, outputs=num_classes)
        if name == 'resnet101':
            model = ResNet(3, ResBottleneckBlock, [3,4,23,3], useBottleneck=True, outputs=num_classes)
        if name == 'resnet152':
            model = ResNet(3, ResBottleneckBlock, [3,8,36,3], useBottleneck=True, outputs=num_classes)
        if name == 'wrn':
            model = WideResNet(3, cfg.depth_scaling, cfg.width_scaling, num_classes, drop_rate=0)
            cfg.model = f'wrn_{cfg.depth_scaling*6+4}_{cfg.width_scaling}'
    except:
        raise Exception('model not found')

    return model
