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

# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]

torch.manual_seed(1)
random.seed(1)


# CUDA = torch.cuda.is_available()
# print("CUDA: %s" % CUDA)

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

   print(fld_path,filename)
    #data 
   with open(os.path.join(fld_path,filename+'.json')) as json_data:
        data = json.load(json_data)
   
   for key, val in data.items():
        word_d, word_c, img_list, target_ind, img_id   = val['dialog'], val['caption'], val['img_list'], val['target'], val['target_img_id']
        stack_d=[]
        if len(img_list)==10:  
            for i, sen in enumerate(word_d):
                sen = sen[0].lower().strip().split(" ")
                stack_d += sen
            word_c = word_c.lower().strip().split(" ")
            if args.dialog:
                
                yield([w2i[x] for x in stack_d],img_list,target_ind,img_id)

            if args.caption:
                
                yield([w2i[x] for x in word_c],img_list,target_ind,img_id)
            
            if args.combine:
               
               word = stack_d+word_c
               yield([w2i[x] for x in word],img_list,target_ind,img_id)



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
  def __init__(self, vocab_size, img_list,embedding_dim, img_feat_dim, hidden_dim,  output_dim):
    super(DeepCBOW, self).__init__()
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.linear1 = nn.Linear((embedding_dim+img_feat_dim),hidden_dim)
    self.linear2 = nn.Linear(hidden_dim,output_dim)


  def forward(self,inputs,img_feat):
      embeds = self.embeddings(inputs)
      embeds = torch.sum(embeds, 1)
      embeds  = embeds.repeat(10,1)
      
      emb_feat = torch.cat((embeds,img_feat),1)
      #---------------------------------                  
      h = F.tanh(self.linear1(emb_feat))
      h = self.linear2(h)
      #---------------------------------
         
      
      return h.transpose(0,1)


img_list = [0]*10
model = DeepCBOW(nwords, img_list, 300, 2048, 64, 1)

# if CUDA:
#     model.cuda()

#get matrix of features given a image_list
def get_img_feat(image_list):
  
   img_m = [img_features[visual_feat_mapping[str(i)]] for i in image_list]

   img_m = torch.from_numpy(np.array(img_m))
   
   return Variable(torch.FloatTensor(img_m))


def evaluate(model, data):
    """Evaluate a model on a data set."""
    correct = 0.0
    
    for words, img_list, target_ind, img_id  in data:
        # forward pass

        scores = model(get_tensor([words]), get_img_feat(img_list))
        
        predict = scores.data.numpy().argmax(axis=1)[0]

        if predict == target_ind:
            correct += 1

    return correct, len(data), correct/len(data)

def get_tensor(x):
    """Get a Variable given indices x"""
    # tensor = torch.cuda.LongTensor(x) if CUDA else torch.LongTensor(x)
    return Variable(torch.LongTensor(x))


optimizer = optim.Adam(model.parameters(), lr=0.001)
print("training")
for ITER in range(10):
    print(ITER)

    random.shuffle(train)
    train_loss = 0.0
    start = time.time()
    count = 0
    for words, img_list, target_ind, img_id in train[0:100]:
        print(count)
        print(img_list)
        
        # forward pass
        scores = model(get_tensor([words]), get_img_feat(img_list))

        loss = nn.CrossEntropyLoss()
        target = get_tensor([target_ind])
        output = loss(scores, target)
        train_loss += output.data[0]

        # backward pass
        model.zero_grad()
        output.backward()

        # update weights
        optimizer.step()
        count+=1

    print("iter %r: train loss/sent=%.4f, time=%.2fs" %
        (ITER, train_loss/len(train), time.time()-start))
        # start = time.time()

# evaluate
    _, _, acc = evaluate(model, val)
    print("iter %r: test acc=%.4f" % (ITER, acc))