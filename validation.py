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


def validation(model, data):
    random.shuffle(data)
    eval_loss = 0.0

    for words, img_list, target_ind, img_id in data:
        scores = model(get_tensor([words]), img_list)
        loss = nn.CrossEntropyLoss()
        targets = get_tensor([target_ind])
        output = loss(scores, targets)
        eval_loss += output.data[0]

    return (eval_loss/len(data))
