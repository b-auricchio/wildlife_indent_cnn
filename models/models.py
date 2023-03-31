from models.resnet import ResNet, ResBlock, ResBottleneckBlock
from models.wideresnet import WideResNet
from models.vgg import VGG

def get_model(filename, k, in_channels, num_classes, n=1):

    try:
        #ResNets 
        if filename == 'resnet10':
            model = ResNet(in_channels, ResBlock, [1,1,1,1], useBottleneck=False, outputs=num_classes, k=k)
        if filename == 'resnet18':
            model = ResNet(in_channels, ResBlock, [2,2,2,2], useBottleneck=False, outputs=num_classes, k=k)
        if filename == 'resnet34':
            model = ResNet(in_channels, ResBlock, [3,4,6,3], useBottleneck=False, outputs=num_classes, k=k)
        if filename == 'resnet50':
            model = ResNet(in_channels, ResBottleneckBlock, [3,4,6,3], useBottleneck=True, outputs=num_classes, k=k)
        if filename == 'resnet101':
            model = ResNet(in_channels, ResBottleneckBlock, [3,4,23,3], useBottleneck=True, outputs=num_classes, k=k)
        if filename == 'resnet152':
            model = ResNet(in_channels, ResBottleneckBlock, [3,8,36,3], useBottleneck=True, outputs=num_classes, k=k)
        if filename == 'wrn':
            model = WideResNet(3, n, k, num_classes, drop_rate=0)
            filename = f'wrn_{n*6+4}_{k}'
            return model, filename
        if filename == 'vgg':
            model = VGG(num_classes)
        
    except:
        raise Exception('model not found')

    return model
