#from data_loader import class_names

from src.CNN_Architecture import Hnet
import torch
import torchvision
import torchvision.transforms as transforms

import PIL
from PIL import Image

def get_modal():
    net = Hnet(3)

    net.load_state_dict(torch.load('model/net.pth'))
    net.eval()
    return net


def load_image(image_path):
    new_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path)
    image = new_transform(image)
    image = image.unsqueeze(0)
    return  image


