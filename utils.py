import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def sample_from_generator(generator, embedding, batch_size, seq_len, h0, use_cuda=True):
    res = []

    x = Variable(torch.tensor(2).repeat((batch_size, 1)).type(torch.LongTensor))
    # x = Variable(torch.zeros((batch_size, 1)).long())
    if use_cuda:
        x = x.cuda()

    h = h0
    c = generator.init_hidden(batch_size)
    samples = []
    for i in range(seq_len):
        emb = embedding(x)
        output, h, c = generator.step(emb, h, c,10)
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
    c = generator.init_hidden(batch_size)
    samples = []
    for i in range(seq_len):
        emb = embedding(x)
        output, h, c = generator.step(emb, h, c,10)
        samples.append(output)

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
        # pdb.set_trace()
        tmp = torch.matmul(torch.diag(delta.squeeze()), Q)
        T = torch.matmul(tmp, torch.diag(sigma.squeeze()))
    return T, C

def IPOT_distance(x, y,use_cuda=True):
    T, C = IPOT(x, y, x.size(0), 1, use_cuda)
    distance = (C*T).sum()
    return distance

# def sample(generator, embedding, batch_size, seq_len, h0, x=None, use_cuda=True):
#     res = []
#     flag = False # whether sample from zero
#     if x is None:
#         flag = True
#     if flag:
#         x = Variable(torch.tensor(0).repeat((batch_size, 1)).type(torch.LongTensor))
#         # x = Variable(torch.zeros((batch_size, 1)).long())
#     if use_cuda:
#         x = x.cuda()
#     h = h0
#     c = generator.init_hidden(batch_size)
#     samples = []
#     if flag:
#         # samples.append(x)
#         for i in range(seq_len):
#             emb = embedding(x)
#             output, h, c = generator.step(emb, h, c,10)
#             x = output.argmax(1).view((-1,1))
#             samples.append(x)
#     else:
#         given_len = x.size(1)
#         lis = x.chunk(x.size(1), dim=1)
#         for i in range(given_len):
#             output, h, c = self.step(lis[i], h, c)
#             samples.append(lis[i])
#         x = output.multinomial(1)
#         for i in range(given_len, seq_len):
#             samples.append(x)
#             output, h, c = self.step(x, h, c)
#             x = output.multinomial(1)
#     output = torch.cat(samples, dim=1)
#     return output