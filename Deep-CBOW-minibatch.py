import json
import os
from collections import defaultdict
import argparse
import h5py
import glob
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time
from collections import namedtuple

# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
UNK = w2i["<unk>"]
PAD = w2i["<pad>"]

# One data point
Example = namedtuple("Example", ["word", "img_list","img_ind", "img_id"])


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dialog', type=int, help='consider only Q&A', default=0)
    parser.add_argument(
        '--caption', type=int, help='consider caption only', default=0)
    parser.add_argument(
        '--combine', type=int, help='combine both dialog and caption', default=0)
    parser.add_argument(
        '--path_folder', type=str, help='path of folder containing data', default="data/VQA_IR_data")
    parser.add_argument(
        '--type', type=str, help='Easy or Hard', default="Easy")
    parser.add_argument(
        '--img_feat', help='folder to image features', default="data/img_feat")

    # Array for all arguments passed to script
    args = parser.parse_args()
    # print(args)
    return args


def read_dataset(process):

   
   #data file path
   fld_path = os.path.join(args.path_folder,args.type)
   filename = 'IR_'+process+'_'+args.type.lower()

    #data 
   with open(os.path.join(fld_path,filename+'.json')) as json_data:
        data = json.load(json_data)
   
   for key, val in data.items():
        word_d, word_c, img_list, target_ind, img_id   = val['dialog'], val['caption'], val['img_list'], val['target'], val['target_img_id']
        stack_d=[]
        for i, sen in enumerate(word_d):
            sen = sen[0].lower().strip().split(" ")
            stack_d += sen
        word_c = word_c.lower().strip().split(" ")
        if args.dialog:
            
            yield Example(word=[w2i[x] for x in stack_d],img_list=img_list, img_ind = target_ind, img_id =img_id)

        if args.caption:
            
            yield Example(word=[w2i[x] for x in word_c],img_list=img_list, img_ind = target_ind, img_id =img_id)
        
        if args.combine:
           
           word = stack_d+word_c
           yield Example(word=[w2i[x] for x in word],img_list=img_list, img_ind = target_ind, img_id =img_id)



args = get_args()
   
#Loading img_features
path_to_h5_file = glob.glob(args.img_feat+"/*.h5")[0]
img_features = np.asarray(h5py.File(path_to_h5_file, 'r')['img_features'])

path_to_json_file = glob.glob(args.img_feat+"/*.json")[0]
with open(path_to_json_file, 'r') as f:
  visual_feat_mapping = json.load(f)['IR_imgid2id']
#-------------------------------------------------------


train = list(read_dataset('train'))
w2i = defaultdict(lambda: UNK, w2i)
val = list(read_dataset('val'))
nwords = len(w2i)
print(nwords)



class DeepCBOW(nn.Module):
  def __init__(self, vocab_size, embedding_dim, img_feat_dim, hidden_dim,  output_dim):
    super(DeepCBOW, self).__init__()
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.linear1 = nn.Linear((embedding_dim+img_feat_dim),hidden_dim)
    self.linear2 = nn.Linear(hidden_dim,output_dim)


  def forward(self, inputs, img_feat):
      embeds = self.embeddings(inputs)
      embeds = torch.sum(embeds, 1)
      embeds = embeds.unsqueeze(-1)
      embeds  = embeds.transpose(1,2)
      embeds  = embeds.repeat(1,10,1)

      emb_feat = torch.cat((embeds,img_feat),2)
      #---------------------------------                  
      h = F.tanh(self.linear1(emb_feat))
      h = self.linear2(h)
      # ---------------------------------
         
      return h


model = DeepCBOW(nwords, 300, 2048, 64, 1)


def minibatch(data, batch_size=32):
   for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

def get_img_feat(image_list):
  
   img_m = [img_features[visual_feat_mapping[str(i)]] for i in image_list]

   # img_m = torch.from_numpy(np.array(img_m))
   return img_m

def preprocess(batch):
    """ Add zero-padding to a batch. """


    # add zero-padding to make all sequences equally long
    seqs = [example.word for example in batch]
    max_length = max(map(len, seqs))
    seqs = [seq + [PAD] * (max_length - len(seq)) for seq in seqs]


    img_feat = [get_img_feat(example.img_list) for example in batch]
    img_feat = np.array(img_feat)

    tags = [example.img_ind for example in batch]

    return seqs, img_feat, tags

def evaluate(model, data):
    """Evaluate a model on a data set."""
    correct = 0.0

    for batch in minibatch(data):
          
          # pad data with zeros
          seqs, img_feat, target_ind = preprocess(batch)

          # forward pass
          scores = model(get_tensor([seqs])[0], Variable(torch.FloatTensor(img_feat)))
          scores = scores[:,:,0]
        
          _, predictions = torch.max(scores.data, 1)
          targets = get_tensor([target_ind])

          correct += torch.eq(predictions, targets[0]).sum().data[0]

    return correct, len(data), correct/len(data)

def get_tensor(x):
    """Get a Variable given indices x"""
    return Variable(torch.LongTensor(x))


optimizer = optim.Adam(model.parameters(), lr=0.001)

print("training")
for ITER in range(20):
    print(ITER)

    random.shuffle(train)
    train_loss = 0.0
    start = time.time()
    count = 0
    updates = 0
    for batch in minibatch(train[0:5000]):
     
        updates += 1

        # pad data with zeros
        seqs, img_feat, target_ind = preprocess(batch)

        scores = model(get_tensor([seqs])[0], Variable(torch.FloatTensor(img_feat)))

        loss = nn.CrossEntropyLoss()
        targets = get_tensor([target_ind])
        output = loss(scores[:,:,0], targets[0])
        train_loss += output.data[0]

        # backward pass
        model.zero_grad()
        output.backward()

        # update weights
        optimizer.step()
        count+=1

    print("iter %r: avg train loss=%.4f, time=%.2fs" %
          (ITER, train_loss/updates, time.time()-start))

# evaluate
    _, _, acc = evaluate(model, val)
    print("iter %r: test acc=%.4f" % (ITER, acc))