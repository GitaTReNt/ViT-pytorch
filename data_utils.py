import logging
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

logger = logging.getLogger(__name__)

def get_loader(args):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    trainset = datasets.CIFAR100(root="./data",
                                train=True,
                                download=True,
                                transform=transform_train)
    testset = datasets.CIFAR100(root="./data",
                                train=False,
                                download=True,
                                transform=transform_test)

    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True)

    return train_loader, test_loader
