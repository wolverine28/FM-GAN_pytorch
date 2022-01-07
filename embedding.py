import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class Embedding(nn.Module):
    """Embedding layer """
    def __init__(self, vocab_size, emb_dim):
        super(Embedding, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.init_params()

    def forward(self, x):
        emb = self.emb(x)
        return emb

    def forward_from_vocab_size(self,onehot):
        return torch.matmul(onehot,self.emb.weight)
        
    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)
