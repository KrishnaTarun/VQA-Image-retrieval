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
    loss_train = []

    print("training")
    for ITER in range(20):
        random.shuffle(data)
        train_loss = 0.0
        start = time.time()
        count = 0

        for words, img_list, target_ind, img_id in data[:10]:
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
            count+=1

        print("iter %r: train loss/sent=%.4f, time=%.2fs" %(ITER+1, train_loss/len(data), time.time()-start))
        loss_train.append(train_loss/len(data))

    return loss_train
