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
import torch.autograd as autograd
import preprocess as ps

# ----------------------
#Add CUDA 
# ---------------------
CUDA = torch.cuda.is_available()
# CUDA = False
print("CUDA: %s" % CUDA)
# ------------------------

# create a folder to store results for particular method
fld = "Combine"
if not os.path.isdir(fld):
  os.makedirs(fld)


# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
UNK = w2i["<unk>"]
PAD = w2i["<pad>"]
# count = Counter()



# One data point
Example = namedtuple("Example", ["w_d", "w_c", "img_list", "img_ind", "img_id"])



def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_folder', type=str, help='path of folder containing data', default="data/VQA_IR_data")
    parser.add_argument(
        '--type', type=str, help='Easy or Hard', default="Easy")
    parser.add_argument(
        '--img_feat', help='folder to image features', default="data/img_feat")
    parser.add_argument(
        '--lr_rate', type=float, help='initial learning rate', default=0.001)

    # Array for all arguments passed to script
    args = parser.parse_args()
    # print(args)
    return args


def read_dataset(process,w2i):

   #data file path
   fld_path = os.path.join(args.path_folder,args.type)
   filename = 'IR_'+process+'_'+args.type.lower()

   print(filename)
   print(os.path.join(fld_path,filename+'.json'))

    #data 
   with open(os.path.join(fld_path,filename+'.json')) as json_data:
        data = json.load(json_data)
   
   for key, val in data.items():
        word_d, word_c, img_list, target_ind, img_id   = val['dialog'], val['caption'], val['img_list'], val['target'], val['target_img_id']
        stack_d=[]
        if len(img_list) == 10:
            for i, sen in enumerate(word_d):
                sen = sen[0].lower().strip().split(" ")
                stack_d += sen
            word_c = word_c.lower().strip().split(" ")
            yield Example(w_d=[w2i[x] for x in stack_d], w_c =[w2i[x] for x in word_c] ,\
                         img_list=img_list, img_ind = target_ind, img_id =img_id)





com_types = "combine"
args = get_args()
# -----------------------------------------   
#Loading img_features
# -------------------------------------------

path_to_h5_file = glob.glob(args.img_feat+"/*.h5")[0]
img_features = np.asarray(h5py.File(path_to_h5_file, 'r')['img_features'])

path_to_json_file = glob.glob(args.img_feat+"/*.json")[0]
with open(path_to_json_file, 'r') as f:
  visual_feat_mapping = json.load(f)['IR_imgid2id']

#-------------------------------------------------------
# create train and val data
# ------------------------------------------------------
train = list(read_dataset('train',w2i))

w2i = defaultdict(lambda: UNK, w2i)

val = list(read_dataset('val',w2i))
#-------------------------------------------------------
nwords = len(w2i)


#LSTM class
class DeepSeq(nn.Module):
  def __init__(self, vocab_size, embed_size, img_feat_dim, hidden_dim_mlp, hidden_dim_lstm, num_layers_lstm,  output_dim, PAD):
    super(DeepSeq, self).__init__()
    self.hidden_dim_lstm = hidden_dim_lstm
    self.embeddings = nn.Embedding(vocab_size, embed_size, padding_idx = PAD)
    
    self.lstm = nn.LSTM(embed_size, hidden_dim_lstm, num_layers_lstm, dropout= 0.5, batch_first=True)
    self.linear_I = nn.Linear(img_feat_dim, hidden_dim_lstm)
    self.linear1 = nn.Linear(2048,embed_size)
    # self.linear2 = nn.Linear(hidden_dim_mlp, output_dim)


  
  def forward(self, inputs, max_d, img_feat, ori_seq_len):
      #-----------------------------------------------
      embeds = self.embeddings(inputs)
      
      #separate embeddings for dialog and caption 
      d_embeds = embeds[:,0:max_d,:]
      c_embeds = embeds[:,max_d:,: ]
      
      #------------------------------------------
      # get embedding for dialog using naive approach 
      d_embeds = torch.sum(d_embeds, 1)
      d_embeds = d_embeds.unsqueeze(1).repeat(1,10,1)
      
      # ------------------------------------------

      c0 = autograd.Variable(torch.zeros(1, c_embeds.size(0), self.hidden_dim_lstm)).cuda()  
      h0 = autograd.Variable(torch.zeros(1, c_embeds.size(0), self.hidden_dim_lstm)).cuda()
      out_, state_ = self.lstm(embeds, (h0, c0))
      
 
      hidden_state = torch.gather(out_, 1, ori_seq_len.view(-1,1,1).expand(ori_seq_len.size(0),1,self.hidden_dim_lstm)-1)
      hidden_state  = hidden_state.repeat(1,10,1)

      # transform 2048 into 1024 and take element wise multiplication
      I = self.linear_I(img_feat)
      I = F.tanh(I)
      h = torch.cat((hidden_state,I),2)
      # h = hidden_state * I
      # #---------------------------------
      h = self.linear1(h)
      h = F.tanh(h)
      h = torch.sum(d_embeds*h,2)
      # calculate dot product of d_embeds with h

         
      return h  


model = DeepSeq(nwords, 300, 2048, 128, 1024, 1, 1, PAD)
print(model)

if CUDA:
    model.cuda()

lr_ = args.lr_rate
optimizer = optim.SGD(model.parameters(), lr=lr_,  momentum=0.9, nesterov=True, weight_decay = 0.0001)


def minibatch(data, batch_size=32):
   for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

def get_img_feat(image_list):
  
   img_m = [img_features[visual_feat_mapping[str(i)]] for i in image_list]

   # img_m = torch.from_numpy(np.array(img_m))
   return img_m

def preprocess(batch):
    """ Add zero-padding to a batch. """

    # for example in batch:
    #     print("length VQA: {} length caption: {}".format(len(example.w_d), len(example.w_c)))

    # add zero-padding to make all sequences equally long
    # do this separately for caption and dialogs
    seqs_d = [example.w_d for example in batch]
    seqs_c = [example.w_c for example in batch]

    #---------------------------------------------------
    # Keep length of caption for gather in LSTM 
    len_seqs = [len(example.w_c) for example in batch]
    # --------------------------------------------------

    # Pad both the dialog and caption and concatenate them
    max_d = max(map(len, seqs_d))
    max_c = max(map(len, seqs_c))

    # Pad---------------------------------------------------- 
    seqs_d = [seq + [PAD] * (max_d - len(seq)) for seq in seqs_d]
    seqs_c = [seq + [PAD] * (max_c - len(seq)) for seq in seqs_c]
    # -------------------------------------------------------

    # combine then to get final sequence----------
    seqs = [seqs_d[i] + seqs_c[i] for i in range(len(batch))]

    # --------------------------------------------
    # Load Image Features
    # --------------------------------------------
    img_feat = [get_img_feat(example.img_list) for example in batch]
    img_feat = np.array(img_feat,dtype=np.float64)
    img_feat = torch.cuda.FloatTensor(img_feat) if CUDA else torch.FloatTensor(img_feat)
    # -----------------------------------------------------------------------
    
    tags = [example.img_ind for example in batch]

    return seqs, max_d, img_feat, tags, len_seqs

def evaluate(model, data):

    model.eval()
    """Evaluate a model on a data set."""
    correct = 0.0
    val_loss = 0.0
    updates = 0
    correct_k = 0
    # print("evalution")
    for batch in minibatch(data, batch_size = 64):


          updates+=1
          
          # pad data with zeros
          seqs, max_d, img_feat, target_ind, ori_seq_len = preprocess(batch)

          # forward pass
          scores = model(get_tensor([seqs])[0], max_d, Variable(img_feat), get_tensor(ori_seq_len))
          
          #calculating loss for validation set
          scores = scores
          loss = nn.CrossEntropyLoss()
          targets = get_tensor([target_ind])
          output = loss(scores, targets[0])
          val_loss += output.data[0]


          
          #Top_1 predicitons
          _, predictions = torch.max(scores.data, 1)
          correct += torch.eq(predictions, targets[0]).sum().data[0]

          #Top_5 predictions
          scores_5, predictions_5 = torch.topk(scores,5,1,largest=True)
          target = targets[0].view(-1,1).expand_as(predictions_5)
          correct_k+= predictions_5.eq(target).float().sum().data[0]

          # print(predictions_5.eq(targets[0].view(-1,1).expand_as(predictions_5)).sum(0,keepdim=True))

    return correct, correct_k, len(data), correct/len(data), correct_k/len(data), val_loss/updates

def get_tensor(x):
    """Get a Variable given indices x"""
    tensor =  torch.cuda.LongTensor(x) if CUDA else torch.LongTensor(x)
    return Variable(tensor)

def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        # if epoch<=7:
        #    lr = lr_ * (0.1 ** (epoch // 7))
        # else:
        lr = lr_ * (0.1 ** (epoch // 10))
        for param_group in optimizer.param_groups:
             param_group['lr'] = lr
        return lr
# ---------------------------------
#model_file_name
# ----------------------------------
name_ = str(args.type)+"_"+com_types
model_file = name_+".pt"
# ----------------------------------


print(len(train))


# At any point you can hit Ctrl + C to break out of training early.
try:
    
    best_val_loss = None
  # ---------------------------------------------
  #To store evaluation
  # ---------------------------------------------
    metric_ = defaultdict(list)
    metric_["folder"] = fld
    n_epochs = 20
    metric_['n_epochs'].append(n_epochs)
    for ITER in range(n_epochs ):

        lr=adjust_learning_rate(optimizer, ITER)

        random.shuffle(train)
        train_loss = 0.0
        start = time.time()
        count = 0
        updates = 0
        

        for batch in minibatch(train, batch_size=96):
            updates += 1
            

            # pad data with zeros
            seqs, max_d, img_feat, target_ind, ori_seq_len = preprocess(batch)
          
          # forward pass
            scores = model(get_tensor([seqs])[0], max_d, \
                      Variable(img_feat), get_tensor(ori_seq_len))

            loss = nn.CrossEntropyLoss()
            targets = get_tensor([target_ind])
            output = loss(scores, targets[0])
            # print(output)
            train_loss += output.data[0]

            # backward pass
            model.zero_grad()
            output.backward()

            # update weights
            optimizer.step()
            count+=1

        print("iter %r: avg train loss=%.4f, time=%.2fs" %
              (ITER, train_loss/updates, time.time()-start))
        # -------------------------------------------------------
        metric_['train_loss'].append(train_loss/updates)
        # -------------------------------------------------------
    # evaluate val

        _, _, _, acc_1, acc_k, val_loss = evaluate(model, val)
        print("iter %r: val top_1 acc=%.4f val top_5 acc=%.4f val_loss=%.4f" % (ITER, acc_1, acc_k, val_loss))

        # ------------------------------------------
        metric_['val_loss'].append(val_loss)
        metric_['val_top1_acc'].append(acc_1)
        metric_['val_top5_acc'].append(acc_k)
        # ----------------------------------

        if not best_val_loss or val_loss < best_val_loss:
              best_val_loss = val_loss
              with open(os.path.join(fld, model_file), 'wb') as f:
                    torch.save(model, f)
except KeyboardInterrupt:
      print('-' * 89)
      print('Exiting from training early')


# Load the best saved model.
with open(os.path.join(fld, model_file), 'rb') as f:
    model = torch.load(f)

#----------------------------------------------------
# Load test data for evaluation
#---------------------------------------------------- 
test = list(read_dataset('test',w2i))

# Run on test data.
test_loss = evaluate(model, test)
print('=' * 89)
_, _, _, test_acc_1, test_acc_k, test_loss = evaluate(model, test)

print("test top_1 acc=%.4f test top_5 acc=%.4f test_loss=%.4f" % (test_acc_1, test_acc_k, test_loss))

metric_['test_loss'].append(test_loss)
metric_['test_top1_acc'].append(test_acc_1)
metric_['test_top5_acc'].append(test_acc_k)

# ---------------------------------------
with open(os.path.join(fld,name_+'.json'), 'w') as outfile:  
    json.dump(metric_, outfile, indent=4)