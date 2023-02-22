from models.resnet import ResNet, ResBlock, ResBottleneckBlock 

def get_model(name:str, in_channels, num_classes):
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
    except:
        raise Exception('model not found')

    return model
