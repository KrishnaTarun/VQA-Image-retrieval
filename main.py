#main python file
from collections import defaultdict

from dataset import *
from options import get_args
from train import training
from model import get_model
from util import plot_graph

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
train_loss = training(model, train)
plot_graph(train_loss)
