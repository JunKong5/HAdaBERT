import torch
import sys
import importlib
importlib.reload(sys)
from functools import partial
import torch.nn as nn
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
device= 'cuda:3'

os.environ["CUDA_VISIBLE_DEDVICES"] ='0,1,2,3'

def sample_gumbel(shape, eps=1e-20):

    U = torch.FloatTensor(shape).uniform_().to(device)

    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size()).to(device)
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature = 5):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature).to(device)

    shape = y.size()
    _, ind = y.max(dim=-1)

    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)

    return (y_hard - y).detach() + y
