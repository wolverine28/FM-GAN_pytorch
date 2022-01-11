import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def sample_from_generator(generator, embedding, batch_size, seq_len, h0, use_cuda=True, use_mvnrom=True):
    res = []

    x = Variable(torch.tensor(2).repeat((batch_size, 1)).type(torch.LongTensor))
    # x = Variable(torch.zeros((batch_size, 1)).long())
    if use_cuda:
        x = x.cuda()

    h = h0
    h = generator.latant_lin(h)
    c = generator.init_hidden(batch_size)
    samples = []
    for i in range(seq_len):
        emb = embedding(x)
        output, h, c = generator.step(emb, h, c,10)
        if use_mvnrom:
            x = output.multinomial(1)
        else:
            x = output.argmax(1).view((-1,1))
        samples.append(x)

    output = torch.cat(samples, dim=1)
    return output

def sample_from_generator_soft(generator, embedding, batch_size, seq_len, h0, use_cuda=True):
    x = Variable(torch.tensor(2).repeat((batch_size, 1)).type(torch.LongTensor))
    # x = Variable(torch.zeros((batch_size, 1)).long())
    if use_cuda:
        x = x.cuda()

    h = h0
    h = generator.latant_lin(h)
    c = generator.init_hidden(batch_size)
    emb = embedding(x)

    samples = []
    for i in range(seq_len):
        output, h, c = generator.step(emb, h, c, 100)
        samples.append(output)
        emb = embedding.forward_from_vocab_size(output).unsqueeze(1)

    output = torch.stack(samples, dim=1)
    return output

def cost_matrix(x,y):
    x, y = F.normalize(x), F.normalize(y)
    tmp1 = torch.matmul(x,y.t())
    cos_dis = 1-tmp1
    return cos_dis

def IPOT(x,y,n,beta=1,use_cuda=True):
    sigma = 1./n*torch.ones([n,1])
    T = torch.ones([n,n])
    if use_cuda:
        sigma = sigma.cuda()
        T = T.cuda()
    C = cost_matrix(x,y)
    A = torch.exp(-C/beta)

    for t in range(50):
        Q = A * T
        for k in range(1):
            delta = 1. / (n * torch.matmul(Q, sigma))
            sigma = 1. / (n * torch.matmul(Q.t(), delta))
        tmp = torch.matmul(torch.diag(delta.squeeze()), Q)
        T = torch.matmul(tmp, torch.diag(sigma.squeeze()))
    return T, C

def IPOT_distance(x, y,use_cuda=True):
    T, C = IPOT(x, y, x.size(0), 1, use_cuda)
    distance = (C*T).sum()
    return distance

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    # if (minibatch_start != n):
    #     # Make a minibatch out of what is left
    #     minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches), len(minibatches)