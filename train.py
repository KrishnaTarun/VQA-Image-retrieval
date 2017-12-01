#Training file
import torch
from torch.autograd import Variable
import  torch.nn as nn
import torch.optim as optim
import time
import random

def get_tensor(x):
    """Get a Variable given indices x"""
    return Variable(torch.LongTensor(x))


def training(model, data):
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    random.shuffle(data)
    train_loss = 0.0

    for words, img_list, target_ind, img_id in data:
        scores = model(get_tensor([words]), img_list)

        loss = nn.CrossEntropyLoss()
        targets = get_tensor([target_ind])
        output = loss(scores, targets)
        train_loss += output.data[0]

        # backward pass
        model.zero_grad()
        output.backward()

        # update weights
        optimizer.step()

    return (train_loss/len(data))
