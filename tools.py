import time
import json
import logging
import numpy as np
import os.path as osp
from pycoco.bleu.bleu import Bleu
from pycoco.meteor.meteor import Meteor
from pycoco.rouge.rouge import Rouge
from pycoco.cider.cider import Cider
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Timer:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.total_time = 0
        self.avg_time = 0
        self.n_toc = 0

    def tic(self):
        self.n_toc = 0
        self.start_time = time.time()

    def toc(self):
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time
        self.n_toc += 1.
        self.avg_time = self.total_time / self.n_toc
        return self.total_time


class Logger:
    """
    When receiving a message, first print it on screen, then write it into log file.
    If save_dir is None, it writes no log and only prints on screen.
    """
    def __init__(self, save_dir):
        if save_dir is not None:
            self.logger = logging.getLogger()
            logging.basicConfig(filename=osp.join(save_dir, 'experiment.log'), format='%(asctime)s |  %(message)s')
            logging.root.setLevel(level=logging.INFO)
        else:
            self.logger = None

    def info(self, msg, to_file=True):
        print msg
        if self.logger is not None and to_file:
            self.logger.info(msg)
def evaluate(gt_file, re_file, logger=None):
    """
    This function is reformed from MSCOCO evaluating code.
    The reference sentences are read from gt_file,
    the generated sentences to be evaluated are read from res_file

    """
    gts = json.load(open(gt_file, 'r'))
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
         #(Meteor(), "METEOR"),
        # (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    metrics = []
    res = json.load(open(re_file, 'r'))
    res = {c['image_id']: [c['caption']] for c in res}
    gts = {k: v for k, v in zip(gts['image_ids'], gts['captions']) if k in res}
    for scorer, method in scorers:
        if logger is not None:
            logger.info('computing %s score...' % (scorer.method()))
        score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                if logger is not None:
                    logger.info("%s: %0.3f" % (m, sc))
            metrics.extend(score)
        else:
            if logger is not None:
                logger.info("%s: %0.3f" % (method, score))
            metrics.append(score)
    return metrics


def lm_caption_step(w_t,lm_state_t,caption_state_t,lm,caption_model,eta,manner):
    word = Variable(torch.LongTensor(w_t.tolist()))
    if lm.on_gpu:
        word = word.cuda()
    word_emb = lm.word_embedding_layer(word)
    logit, lm_state_t_1 = lm.forward(word_emb, lm_state_t)  # logit : (batch_size, vocab_size)
    prob = F.softmax(logit)  # (batch_size, vocab_size)
    P = prob - eta
    P *= 10000000
    mask = F.sigmoid(P).data.cpu().numpy()
    caption_state_t_1,w_t_1 = caption_model.ngram_single_step(caption_state_t, w_t,mask,manner)
    return w_t_1,lm_state_t_1,caption_state_t_1


def lm_caption(lm,model,image_ids,vocab,loader,feature,max_step,manner):
    w_0 = np.ones((len(image_ids),), dtype=np.int32) * 9488  # set start token for rnn language model
    lm_state_0 = lm.init_state()
    cap_state_0 = model.initial_state(feature)
    eta_0 = 0.00005
    cap = np.zeros((max_step, len(image_ids)), dtype=np.int32)
    if manner == 'sample':
        res = []
        for step in range(max_step-1):
            w_1, lm_state_1, cap_state_1 = lm_caption_step(w_0, lm_state_0, cap_state_0, lm, model, eta_0*(2**step), manner)
            cap[step + 1, :] = w_1[:]
            w_0 = w_1
            lm_state_0 = lm_state_1
            cap_state_0 = cap_state_1
        for i in range(loader.batch_size):
            index = np.where(cap[1:,i] == 0)[0]
            if len(index) > 0:
               s = ' '.join(vocab[w] for w in cap[1:index[0]+1, i])
            else:
                s = ' '.join(vocab[w] for w in cap[1:, i])
            res.append({'image_id': image_ids[i], 'caption': s})
    else:
         cap, res = model.inference(vocab, image_ids, feature, manner='greedy', max_length=max_step)
    return cap,res

def att_lm_caption_step(w_t,lm_state_t,patches,caption_state_t,lm,caption_model,eta,manner):
    word = Variable(torch.LongTensor(w_t.tolist()))
    if lm.on_gpu:
        word = word.cuda()
    word_emb = lm.word_embedding_layer(word)
    logit, lm_state_t_1 = lm.forward(word_emb, lm_state_t)  # logit : (batch_size, vocab_size)
    prob = F.softmax(logit)  # (batch_size, vocab_size)
    P = prob - eta
    P *= 10000000
    mask = F.sigmoid(P).data.cpu().numpy()
    caption_state_t_1,w_t_1 = caption_model.ngram_single_step(caption_state_t, w_t,patches,mask,manner)
    return w_t_1,lm_state_t_1,caption_state_t_1

def att_lm_caption(lm,model,image_ids,vocab,loader,features,max_step,manner):
    w_0 = np.ones((len(image_ids),), dtype=np.int32) * 9488  # set start token for rnn language model
    lm_state_0 = lm.init_state()
    eta_0 = 0.00005
    cap = np.zeros((max_step, len(image_ids)), dtype=np.int32)
    if manner == 'sample':
        pathes, cap_state_0 = model.initial_state(features)
        res = []
        for step in range(max_step-1):
            w_1, lm_state_1, cap_state_1 = att_lm_caption_step(w_0, lm_state_0,pathes, cap_state_0, lm, model, eta_0*(2**step), manner)
            cap[step + 1, :] = w_1[:]
            w_0 = w_1
            lm_state_0 = lm_state_1
            cap_state_0 = cap_state_1
        for i in range(loader.batch_size):
            index = np.where(cap[1:,i] == 0)[0]
            if len(index) > 0:
               s = ' '.join(vocab[w] for w in cap[1:index[0]+1, i])
            else:
                s = ' '.join(vocab[w] for w in cap[1:, i])
            res.append({'image_id': image_ids[i], 'caption': s})
    else:
         cap, res = model.inference(vocab, image_ids, features, manner='greedy', max_length=max_step)
    return cap,res


def lm2_caption_step(w_t,first_word,lm_state_t,caption_state_t,lm,caption_model,eta,manner,step):
    word = Variable(torch.LongTensor(w_t.tolist()))
    if lm.on_gpu:
        word = word.cuda()
    word_emb = lm.word_embedding_layer(word)
    logit, lm_state_t_1 = lm.forward(word_emb, lm_state_t)  # logit : (batch_size, vocab_size)
    prob = F.softmax(logit)  # (batch_size, vocab_size)
    P = prob - eta
    P *= 10000000
    mask = F.sigmoid(P).data.cpu().numpy()[:,:-1] # drop the start token
    if step == 0:
        caption_state_t_1,w_t_1 = caption_model.ngram_single_step(0,caption_state_t, first_word,mask,manner)
    else:
        caption_state_t_1, w_t_1 = caption_model.ngram_single_step(step, caption_state_t, w_t, mask, manner)
    return w_t_1,lm_state_t_1,caption_state_t_1


def lm2_caption(lm,model,image_ids,vocab,loader,feature,max_step,manner):
    w_0 = np.ones((len(image_ids),), dtype=np.int32) * 9488  # set start token for rnn language model
    lm_state_0 = lm.init_state()
    first_word, cap_state_0 = model.initial_state(feature)
    eta_0 = 0.00005
    cap = np.zeros((max_step, len(image_ids)), dtype=np.int32)
    if manner == 'sample':
        res = []
        for step in range(max_step-1):
            w_1, lm_state_1, cap_state_1 = lm2_caption_step(w_0,first_word, lm_state_0, cap_state_0, lm, model, eta_0*(2**step), manner,step)
            cap[step + 1, :] = w_1[:]
            w_0 = w_1
            lm_state_0 = lm_state_1
            cap_state_0 = cap_state_1
        for i in range(loader.batch_size):
            index = np.where(cap[1:,i] == 0)[0]
            if len(index) > 0:
               s = ' '.join(vocab[w] for w in cap[1:index[0]+1, i])
            else:
                s = ' '.join(vocab[w] for w in cap[1:, i])
            res.append({'image_id': image_ids[i], 'caption': s})
    else:
         cap, res = model.inference(vocab, image_ids, feature, manner='greedy', max_length=max_step)
    return cap[1:,:],res

def att2_lm_caption_step(w_t,first_word,lm_state_t,patches,caption_state_t,lm,caption_model,eta,manner,step):
    word = Variable(torch.LongTensor(w_t.tolist()))
    if lm.on_gpu:
        word = word.cuda()
    word_emb = lm.word_embedding_layer(word)
    logit, lm_state_t_1 = lm.forward(word_emb, lm_state_t)  # logit : (batch_size, vocab_size)
    prob = F.softmax(logit)  # (batch_size, vocab_size)
    P = prob - eta
    P *= 10000000
    mask = F.sigmoid(P).data.cpu().numpy()[:,:-1]  # drop the start token
    if step == 0:
        caption_state_t_1,w_t_1 = caption_model.ngram_single_step(0,caption_state_t,first_word,patches,mask,manner)
    else:
        caption_state_t_1, w_t_1 = caption_model.ngram_single_step(step,caption_state_t, w_t, patches, mask, manner)
    return w_t_1,lm_state_t_1,caption_state_t_1

def att2_lm_caption(lm,model,image_ids,vocab,loader,features,max_step,manner):
    w_0 = np.ones((len(image_ids),), dtype=np.int32) * 9488  # set start token for rnn language model
    lm_state_0 = lm.init_state()
    eta_0 = 0.00005
    cap = np.zeros((max_step, len(image_ids)), dtype=np.int32)
    if manner == 'sample':
        patches,first_word, cap_state_0 = model.initial_state(features)
        res = []
        for step in range(max_step-1):
            w_1, lm_state_1, cap_state_1 = att2_lm_caption_step(w_0,first_word, lm_state_0,patches, cap_state_0, lm, model, eta_0*(2**step), manner,step)
            cap[step + 1, :] = w_1[:]
            w_0 = w_1
            lm_state_0 = lm_state_1
            cap_state_0 = cap_state_1
        for i in range(loader.batch_size):
            index = np.where(cap[1:,i] == 0)[0]
            if len(index) > 0:
               s = ' '.join(vocab[w] for w in cap[1:index[0]+1, i])
            else:
                s = ' '.join(vocab[w] for w in cap[1:, i])
            res.append({'image_id': image_ids[i], 'caption': s})
    else:
         cap, res = model.inference(vocab, image_ids, features, manner='greedy', max_length=max_step)
    return cap[1:,:],res

