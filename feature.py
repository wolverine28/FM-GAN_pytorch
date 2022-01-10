# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class Fextractor(nn.Module):
    """A CNN for text classification

    architecture: Embedding >> Convolution >> Max-pooling >> Softmax
    """

    def __init__(self,  emb_dim, dropout):
        super(Fextractor, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.init_parameters()
        self.RELU = nn.ReLU()
        self.conv1 = nn.Conv2d(1,300,[5,emb_dim],stride=[2,1])
        self.conv2 = nn.Conv2d(300,600,[5,1],stride=[2,1])
        self.conv3 = nn.Conv2d(600,256,[5,1],stride=[2,1])

    def forward(self, emb):
        """
        Args:
            x: (batch_size * seq_len)
        """
        emb = emb.unsqueeze(1)  # batch_size * 1 * seq_len * emb_dim
        h1 = self.conv1(F.pad(emb,(0,0,4,4),'constant',0))
        h1 = self.RELU(h1)
        
        h2 = self.conv2(h1)
        h2 = self.RELU(h2)

        h3 = self.conv3(h2)
        out = F.tanh(F.avg_pool2d(h3,(3,1)))

        return out.squeeze()

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

if __name__=="__main__":
    emb_dim = 100
    dropout = 0.5
    # Define the generator and initialize the weights
    f = Fextractor(emb_dim, dropout)

    emb = torch.randn((64,50,emb_dim))
    f(emb)

    print()
