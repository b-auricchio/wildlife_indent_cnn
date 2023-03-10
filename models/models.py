from models.resnet import ResNet, ResBlock, ResBottleneckBlock
from models.wideresnet import WideResNet

def get_model(args, in_channels, num_classes):
    name = args.model
    try:
        #ResNets 
        if name == 'resnet18':
            model = ResNet(in_channels, ResBlock, [2,2,2,2], useBottleneck=False, outputs=num_classes, k=args.k)
        if name == 'resnet34':
            model = ResNet(in_channels, ResBlock, [3,4,6,3], useBottleneck=False, outputs=num_classes, k=args.k)
        if name == 'resnet50':
            model = ResNet(in_channels, ResBottleneckBlock, [3,4,6,3], useBottleneck=True, outputs=num_classes, k=args.k)
        if name == 'resnet101':
            model = ResNet(in_channels, ResBottleneckBlock, [3,4,23,3], useBottleneck=True, outputs=num_classes, k=args.k)
        if name == 'resnet152':
            model = ResNet(in_channels, ResBottleneckBlock, [3,8,36,3], useBottleneck=True, outputs=num_classes, k=args.k)
        if name == 'wrn':
            model = WideResNet(3, args.n, args.k, num_classes, drop_rate=0)
            args.model = f'wrn_{args.n*6+4}_{args.k}'
    except:
        raise Exception('model not found')

    return model
