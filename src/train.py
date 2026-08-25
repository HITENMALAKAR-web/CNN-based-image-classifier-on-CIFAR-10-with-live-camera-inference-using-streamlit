import torch
import torch.nn as nn

from data_loader import load_data,class_names
from CNN_Architecture import Hnet

import torch.optim as optim


def train():
    # load_data
    dataset = load_data()
    # gpu training
    device = torch.device('cuda')

    # lr and epoch
    learning_rate = 0.01
    epochs = 60

    # model
    model = Hnet(3)
    model.to(device)
    optimizer = optim.SGD(model.parameters(),lr = learning_rate,weight_decay=5e-4,momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # training_loop

    for epoch in range(epochs):
        print(f"epoch : {epoch}")
        running_loss = 0

        for data in dataset:
            image,label = data

            image = image.to(device)
            label = label.to(device)
            # gradient reset
            optimizer.zero_grad()
            # output
            output = model(image)
            # loss
            loss = criterion(output,label)
            loss.backward()
            # optim
            optimizer.step()

            running_loss +=loss.item()
        print(f"loss : {running_loss/len(dataset):.4f}")

    torch.save(model.state_dict(),'model/net.pth')

    return None

if __name__ == '__main__':
    train()



