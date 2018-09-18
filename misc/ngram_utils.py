from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
bad_endding = ['with','in','on','of','a','at','to','for','an','this','his','her','that']
import collections
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def if_use_att(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'all_img', 'fc']:
        return False
    return True

def delete_badding(txt):
    if txt.endswith('with'):
        txt = txt[0:len(txt)-5]


# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out
def delete_decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        flag = 0
        for j in range(D):
            ix = seq[i,D-1-j]
            if ix > 0 and ix_to_word[str(ix)] not in bad_endding:
                flag = D-j
                break
        txt = ' '.join([ix_to_word[str(ix)] for ix in seq[i,0:flag]])
        out.append(txt)
    return out
def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * Variable(mask)
        output = torch.sum(output) / torch.sum(mask)

        return output
class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)