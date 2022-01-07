# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from generator import Generator
from embedding import Embedding
from feature import Fextractor
from utils import sample_from_generator

vocab_size = 5000
emb_dim = 100
hidden_dim = 64
use_cuda = True

d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dropout = 0.5

# %%
embedding = Embedding(vocab_size, emb_dim)
generator = Generator(vocab_size, emb_dim, hidden_dim, use_cuda)
extractor = Fextractor(emb_dim, d_filter_sizes, d_num_filters, dropout)

embedding, generator, extractor = embedding.cuda(), generator.cuda(), extractor.cuda()
# %%
x = torch.randint(low=0,high=5000,size=(64,50)).cuda()
z = torch.randn((1, 64,hidden_dim)).cuda()
emb = embedding(x)
gen = generator(emb,z)
# %%
samples = sample_from_generator(generator, embedding,64,50,z)
emb = embedding(samples)
feature1 = extractor(emb)

samples = sample_from_generator(generator, embedding,64,50,z)
emb = embedding(samples)
feature2 = extractor(emb)
# %%
