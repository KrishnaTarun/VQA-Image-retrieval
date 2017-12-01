#python
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from dataset import get_image_features, get_visual_feature_mapping
from options import get_args

class DeepCBOW(nn.Module):
  def __init__(self, vocab_size, img_list,embedding_dim, img_feat_dim, hidden_dim,  output_dim):
    super(DeepCBOW, self).__init__()
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.linear1 = nn.Linear((embedding_dim+img_feat_dim),hidden_dim)
    self.linear2 = nn.Linear(hidden_dim,output_dim)


  def forward(self,inputs,img_list):
      embeds = self.embeddings(inputs)
      embed_ = torch.sum(embeds, 1)

      store = torch.zeros(10,1)
      input_matrix = torch.FloatTensor(1,1)

      args = get_args()
      img_features = get_image_features(args)
      visual_feat_mapping = get_visual_feature_mapping(args)

      for ind, img_id in enumerate(img_list):
          h5_id = visual_feat_mapping[str(img_id)]
          img_feat = Variable(torch.FloatTensor(img_features[h5_id]))
          img_feat = img_feat.view(1,2048)
          h_ = torch.cat((img_feat,embed_),1)
          if input_matrix.size(1) == 1:
              input_matrix = h_.clone()
          else:
              input_matrix = torch.cat((input_matrix, h_), 0)

      #---------------------------------                  
      h = F.tanh(self.linear1(input_matrix))
      score = self.linear2(h).transpose(0, 1)
      #---------------------------------

      return score

