#python
import torch
import torch.nn as nn
import torch.nn.functional as F

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
