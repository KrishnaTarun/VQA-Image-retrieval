#main python file
import torch
import random
from collections import defaultdict

from dataset import *
from options import get_args
from train import training
from validation import validation
from model import get_model
from util import plot_graph

torch.manual_seed(1)
random.seed(1)

#get arguments provided by the user or default arguments
args = get_args()

#load train and val dataset
w2i = defaultdict(lambda: len(w2i))
train = list(read_dataset('train', args, w2i))
UNK = w2i["<unk>"]
w2i = defaultdict(lambda: UNK, w2i)
val = list(read_dataset('val', args, w2i))
nwords = len(w2i)

#load the NN model
model = get_model(args.model_type, nwords)

#start the training
train_loss = []
eval_loss = []

print("Starting the training for %d epochs" %(args.epochs))
for ITER in range(args.epochs):
    tLoss = training(model, train)
    vLoss = validation(model, val)
    print("Epoch:%d, train loss: %.4f, validation loss:%.4f"%(ITER+1, tLoss, vLoss))
    train_loss.append(tLoss)
    eval_loss.append(vLoss)

plot_graph(train_loss, eval_loss)
