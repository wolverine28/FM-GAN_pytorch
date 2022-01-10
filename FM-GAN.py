# -*- coding:utf-8 -*-
# %%
import os
import random
import math

import argparse
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from word_correction import tokenizer

from torchtext.legacy import data, datasets
import torchtext
import dill

from generator import Generator
from embedding import Embedding
from feature import Fextractor
from utils import sample_from_generator, IPOT, IPOT_distance, sample_from_generator_soft

# %%
# ================== Parameter Definition =================
BATCH_SIZE = 128
SEQ_LEN = 50
PRE_EPOCH_NUM = 50
SEED = 88

emb_dim = 100
latant_dim = 128
hidden_dim = 100
use_cuda = True

# d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
# d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dropout = 0.5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# %%
# ================== Dataset Definition =================
glove = torchtext.vocab.GloVe(name='6B',dim=100)
print(len(glove.itos)) #400000
print(glove.vectors.shape)


# TEXT = data.Field(sequential=True, batch_first=True, lower=True,init_token='<sos>', eos_token='<eos>',
#                 fix_length=SEQ_LEN+2, tokenize=tokenizer)
# LABEL = data.Field(sequential=False, batch_first=True)

# trainset, testset = datasets.IMDB.splits(TEXT, LABEL)

# TEXT.build_vocab(trainset,max_size=30000,vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)
# LABEL.build_vocab(trainset)

## save set
# with open("log/TEXT.Field","wb")as f:
#      dill.dump(TEXT,f)

# with open("log/LABEL.Field","wb")as f:
#      dill.dump(LABEL,f)

# torch.save(trainset.examples,'./log/trset_orig')
# torch.save(testset.examples,'./log/tsset_orig')

## load set
trainset_examples = torch.load('./log/trset_orig')
testset_examples = torch.load('./log/tsset_orig')

with open("log/TEXT.Field","rb")as f:
    TEXT=dill.load(f)
with open("log/LABEL.Field","rb")as f:
    LABEL=dill.load(f)

trainset = data.Dataset(trainset_examples,{'text':TEXT,'label':LABEL})
testset = data.Dataset(testset_examples,{'text':TEXT,'label':LABEL})


VOCAB_SIZE = len(TEXT.vocab)
print('VOCAB_SIZE : {}'.format(VOCAB_SIZE))

train_loader = data.Iterator(dataset=trainset, batch_size = BATCH_SIZE)

###############################################################################
# %%
embedding = Embedding(VOCAB_SIZE, emb_dim)
generator = Generator(latant_dim, VOCAB_SIZE, emb_dim, hidden_dim, use_cuda)
extractor = Fextractor(emb_dim, dropout)

if use_cuda:
    embedding = embedding.cuda()
    generator = generator.cuda()
    extractor = extractor.cuda()
# %%
# pretrained_embeddings = TEXT.vocab.vectors
# print(pretrained_embeddings.shape)
# embedding.emb.weight.data.copy_(pretrained_embeddings)

# unk_idx = TEXT.vocab.stoi[TEXT.unk_token]
# pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
# init_idx = TEXT.vocab.stoi[TEXT.init_token]
# eos_idx = TEXT.vocab.stoi[TEXT.eos_token]

# embedding.emb.weight.data[unk_idx] = torch.zeros(emb_dim)
# embedding.emb.weight.data[pad_idx] = torch.zeros(emb_dim)
# embedding.emb.weight.data[init_idx] = torch.zeros(emb_dim)
# embedding.emb.weight.data[eos_idx] = torch.zeros(emb_dim)

# print(embedding.emb.weight.data)
# %%
# embedding_optim = optim.Adam(embedding.parameters(),lr=0.01)
# generator_optim = optim.Adam(generator.parameters(),lr=0.01)
# %%
bs = 1
z = torch.randn((1, bs, latant_dim)).cuda()
samples = sample_from_generator(generator, embedding,bs,50,z)
# samples = sample_from_generator_soft(generator, embedding,bs,50,z)
print(' '.join([TEXT.vocab.itos[i] for i in samples[0]]))

# %%
# ================== Pretrain with MLE =================
# def train_epoch(model, data_iter, criterion, optimizerG,optimizerE):
#     total_loss = 0.
#     total_words = 0.
#     for data in tqdm(data_iter, mininterval=2, desc=' - Training', leave=False):
#         data = Variable(data.text[1:])
#         z = torch.randn((1, data.size(0), latant_dim)).cuda()
#         # z = torch.zeros((1, data.size(0), hidden_dim)).cuda()
#         if use_cuda:
#             data = data.cuda()
#         emb = embedding(data)
#         pred = model.forward(emb, z)
#         loss = criterion(pred[:-1], data.view(-1)[1:])
#         total_loss += loss.item()
#         total_words += data.size(0) * data.size(1)

#         optimizerG.zero_grad()
#         optimizerE.zero_grad()
#         loss.backward()
#         optimizerG.step()
#         optimizerE.step()
#     return math.exp(total_loss / total_words)

# gen_criterion = nn.NLLLoss(reduction='sum')
# ##generator_optim = optim.Adam(generator.parameters(),lr=0.01)
# #embedding_optim = optim.Adam(embedding.parameters(),lr=0.01)
# if use_cuda:
#     gen_criterion = gen_criterion.cuda()
# print('Pretrain with MLE ...')
# for epoch in range(PRE_EPOCH_NUM):
#     loss = train_epoch(generator, train_loader, gen_criterion, generator_optim, embedding_optim)
#     print('Epoch [%d] Model Loss: %f'% (epoch, loss))

#     z = torch.randn((1, bs, latant_dim)).cuda()
#     # z = torch.zeros((1, bs, hidden_dim)).cuda()
#     samples = sample_from_generator(generator, embedding,bs,50,z,use_mvnrom=True)
#     print(' '.join([TEXT.vocab.itos[i] for i in samples[0]]))
# %%
embedding_optim = optim.Adam(embedding.parameters(),lr=1e-5)
generator_optim = optim.Adam(generator.parameters(),lr=1e-5)
extractor_optim = optim.Adam(extractor.parameters(),lr=1e-5)

embedding_scheduler = optim.lr_scheduler.ExponentialLR(embedding_optim, 0.99)
generator_scheduler = optim.lr_scheduler.ExponentialLR(generator_optim, 0.99)
extractor_scheduler = optim.lr_scheduler.ExponentialLR(extractor_optim, 0.99)
EPOCH = 200
for epoch in range(EPOCH):
    # train extractor
    losses_disc = []
    losses_gen = []
    for _ in range(1):
        for i, batch in enumerate(tqdm(train_loader)):
            text = Variable(batch.text)
            z = torch.randn((1, text.size(0), latant_dim))
            if use_cuda:
                text = text.cuda()
                z = z.cuda()
            samples = sample_from_generator_soft(generator, embedding,text.size(0),SEQ_LEN,z)

            fake_feat = extractor(embedding.forward_from_vocab_size(samples))
            real_feat = extractor(embedding(text[:,1:-1]))
            loss = IPOT_distance(fake_feat, real_feat)

            if i % 2==0:
                loss_disc = -loss
                extractor_optim.zero_grad()
                loss.backward()
                extractor_optim.step()
                losses_disc.append(loss_disc.item())
            else:
                loss_gen = loss
                embedding_optim.zero_grad()
                generator_optim.zero_grad()
                loss.backward()
                embedding_optim.step()
                generator_optim.step()
                losses_gen.append(loss_gen.item())

        extractor_scheduler.step()
        embedding_scheduler.step()
        generator_scheduler.step()

        print('Epoch [%d], IPOT_distance Discriminator: %f' % (epoch, -np.mean(losses_disc)))
        print('Epoch [%d], IPOT_distance Generator: %f' % (epoch, np.mean(losses_gen)))

    z = torch.randn((1, 1, latant_dim)).cuda()
    samples = sample_from_generator(generator, embedding,1,50,z,use_mvnrom=False)
    print(' '.join([TEXT.vocab.itos[i] for i in samples[0]]))
