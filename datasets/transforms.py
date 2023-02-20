from torchvision import transforms

stats = ((0.4887, 0.4998, 0.4423),(0.2309, 0.2264, 0.2617)) #for cub
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(468),
    transforms.RandomCrop(448),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

test_transform = transforms.Compose([
    transforms.Resize(468),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])