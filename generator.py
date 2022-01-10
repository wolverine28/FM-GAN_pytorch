import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Generator(nn.Module):
    """Generator """
    def __init__(self, latant_dim, vocab_size, emb_dim, hidden_dim, use_cuda):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.lin = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.latant_lin = nn.Linear(latant_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.init_params()

    def forward(self, emb, h0):
        """
        Args:
            x: (batch_size, seq_len), sequence of tokens generated by generator
        """
        c0 = self.init_hidden(emb.size(0))
        h0 = self.latant_lin(h0)
        output, (h, c) = self.lstm(emb, (h0, c0))
        output = self.dropout(output.contiguous().view(-1, self.hidden_dim))
        pred = self.softmax(self.lin(output))
        return pred

    def step(self, emb, h, c, T):
        """
        Args:
            x: (batch_size,  1), sequence of tokens generated by generator
            h: (1, batch_size, hidden_dim), lstm hidden state
            c: (1, batch_size, hidden_dim), lstm cell state
        """
        output, (h, c) = self.lstm(emb, (h, c))
        output = self.dropout(output.view(-1, self.hidden_dim))
        pred = F.softmax(self.lin(output)*T, dim=1)
        return pred, h, c


    def init_hidden(self, batch_size):
        c = Variable(torch.zeros((1, batch_size, self.hidden_dim)))
        if self.use_cuda:
            c = c.cuda()
        return c

    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)
