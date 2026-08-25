import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transform

def load_data():
    transforms = transform.Compose([
        transform.ToTensor(),
        transform.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    train_data = torchvision.datasets.CIFAR10(train=True,transform=transforms,root ='./data/raw',download=True)
    test_data = torchvision.datasets.CIFAR10(train=False, transform=transforms, root='./data/raw', download=True)

    full_data = torch.utils.data.ConcatDataset([train_data,test_data])

    dataset = torch.utils.data.DataLoader(dataset=full_data,shuffle=True,batch_size=32,num_workers=2)

    return dataset

class_names = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']


