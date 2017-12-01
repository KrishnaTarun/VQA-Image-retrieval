#python
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dataset import get_image_features, get_visual_feature_mapping
from options import get_args


#get matrix of features given a image_list
def get_img_feat(image_list):
    args = get_args()
    img_features = get_image_features(args)
    visual_feat_mapping = get_visual_feature_mapping(args)
    img_m = [img_features[visual_feat_mapping[str(i)]] for i in image_list]
    img_m = torch.from_numpy(np.array(img_m))

    return Variable(torch.FloatTensor(img_m))


class DeepCBOW(nn.Module):
  def __init__(self, vocab_size, img_list,embedding_dim, img_feat_dim, hidden_dim,  output_dim):
    super(DeepCBOW, self).__init__()
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.linear1 = nn.Linear((embedding_dim+img_feat_dim),hidden_dim)
    self.linear2 = nn.Linear(hidden_dim,output_dim)


  def forward(self,inputs,image_list):
      img_feat = get_img_feat(image_list)
      embeds = self.embeddings(inputs)
      embeds = torch.sum(embeds, 1)
      embeds  = embeds.repeat(10,1)

      emb_feat = torch.cat((embeds,img_feat),1)
      #---------------------------------                  
      h = F.tanh(self.linear1(emb_feat))
      score = self.linear2(h).transpose(0,1)
      #---------------------------------


      return score

