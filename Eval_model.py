import time
import json
import logging
import numpy as np
import os.path as osp
from pycoco.bleu.bleu import Bleu
from pycoco.meteor.meteor import Meteor
from pycoco.rouge.rouge import Rouge
from pycoco.cider.cider import Cider
bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']

def count_bad(sen,max_step):
    sen = sen.split(' ')
    if len(sen) < max_step and sen[-1] in bad_endings:
        return 1
    else:
        return 0


def evaluate(gt_file, re_file, logger=None):
    """
    This function is reformed from MSCOCO evaluating code.
    The reference sentences are read from gt_file,
    the generated sentences to be evaluated are read from res_file

    """
    gts = json.load(open(gt_file, 'r'))
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
         (Meteor(), "METEOR"),
         (Rouge(), "ROUGE_L"),
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



import sys
import ngram_opts
from dataloader import *
opts = ngram_opts.parse_opt()
lr = 0.0005
opts.batch_size = 50
loader = KKDataLoader(opts)
vocabs = loader.get_vocab()
vocab = ['#END#']
for i in range(len(vocabs)):
    ids = str(i+1)
    vocab.append(vocabs[ids])
save_dir = 'eval'
model_type = opts.caption_model  # fc or attention
rl_type = opts.rl_type # 'fourgram', 'trigram', 'rnnlm'
batch_size = opts.batch_size
image_dim = 2048
cell_size = 512
if rl_type == 'fourgram':
    if model_type == 'att':
        from caption_model.att import *
        vocab_size = 9489
        model = AttModel(batch_size=batch_size, image_dim=image_dim, vocab_size=vocab_size, cell_size=cell_size, lr=lr,
                         ngram=4, on_gpu=True)
        model.load('fourgram_cider_model/att_model/model.best')
        results = []
        for kkk in range(5000 / opts.batch_size):
            data = loader.get_batch('test')
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]
            fc_feats, att_feats = tmp
            image_id = [data['infos'][i]['id'] for i in range(batch_size)]
            att_feats = att_feats.reshape(att_feats.shape[0], att_feats.shape[1] * att_feats.shape[2],att_feats.shape[3])
            greedy_cap, greedy_res = model.inference(vocab, image_id, att_feats, manner='greedy', max_length=20)
            results += greedy_res
    else:
        from caption_model.fc import *
        vocab_size = 9489
        model = FCModel(batch_size=batch_size, image_dim=image_dim, vocab_size=vocab_size, cell_size=cell_size, lr=lr,
                        ngram=4, on_gpu=True)
        model.load('fourgram_cider_model/fc_model/model.best')
        results = []
        for kkk in range(5000 / opts.batch_size):
            data = loader.get_batch('test')
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]
            fc_feats, att_feats = tmp
            image_id = [data['infos'][i]['id'] for i in range(batch_size)]
            greedy_cap, greedy_res = model.inference(vocab, image_id, fc_feats, manner='greedy', max_length=20)
            results += greedy_res
elif rl_type =='trigram':
    if model_type == 'att':
        from caption_model.att import *
        vocab_size = 9489
        model = AttModel(batch_size=batch_size, image_dim=image_dim, vocab_size=vocab_size, cell_size=cell_size, lr=lr,
                         ngram=3, on_gpu=True)
        model.load('trigram_cider_model/att_model/model.best')
        results = []
        for kkk in range(5000 / opts.batch_size):
            data = loader.get_batch('test')
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]
            fc_feats, att_feats = tmp
            image_id = [data['infos'][i]['id'] for i in range(batch_size)]
            att_feats = att_feats.reshape(att_feats.shape[0], att_feats.shape[1] * att_feats.shape[2],att_feats.shape[3])
            greedy_cap, greedy_res = model.inference(vocab, image_id, att_feats, manner='greedy', max_length=20)
            results += greedy_res
    else:
        from caption_model.fc import *
        vocab_size = 9489
        model = FCModel(batch_size=batch_size, image_dim=image_dim, vocab_size=vocab_size, cell_size=cell_size, lr=lr,
                        ngram=3, on_gpu=True)
        model.load('trigram_cider_model/fc_model/model.best')
        results = []
        for kkk in range(5000 / opts.batch_size):
            data = loader.get_batch('test')
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]
            fc_feats, att_feats = tmp
            image_id = [data['infos'][i]['id'] for i in range(batch_size)]
            greedy_cap, greedy_res = model.inference(vocab, image_id, fc_feats, manner='greedy', max_length=20)
            results += greedy_res
elif rl_type =='rnnlm':
    if model_type == 'att':
        from caption_model.att import *
        vocab_size = 9489
        model = AttModel(batch_size=batch_size, image_dim=image_dim, vocab_size=vocab_size, cell_size=cell_size, lr=lr, on_gpu=True)
        model.load('rnnlm_cider_model/att_model/model.best')
        results = []
        for kkk in range(5000 / opts.batch_size):
            data = loader.get_batch('test')
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]
            fc_feats, att_feats = tmp
            image_id = [data['infos'][i]['id'] for i in range(batch_size)]
            att_feats = att_feats.reshape(att_feats.shape[0], att_feats.shape[1] * att_feats.shape[2],att_feats.shape[3])
            greedy_cap, greedy_res = model.inference(vocab, image_id, att_feats, manner='greedy', max_length=20)
            results += greedy_res
    else:
        from caption_model.fc import *
        vocab_size = 9489
        model = FCModel(batch_size=batch_size, image_dim=image_dim, vocab_size=vocab_size, cell_size=cell_size, lr=lr,on_gpu=True)
        model.load('rnnlm_cider_model/fc_model/model.best')
        results = []
        for kkk in range(5000 / opts.batch_size):
            data = loader.get_batch('test')
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]
            fc_feats, att_feats = tmp
            image_id = [data['infos'][i]['id'] for i in range(batch_size)]
            greedy_cap, greedy_res = model.inference(vocab, image_id, fc_feats, manner='greedy', max_length=20)
            results += greedy_res
elif rl_type =='sc':
    if model_type == 'att':
        from caption_model.att import *
        vocab_size = 9489
        model = AttModel(batch_size=batch_size, image_dim=image_dim, vocab_size=vocab_size, cell_size=cell_size, lr=lr, on_gpu=True)
        model.load('sc_cider_model/att_model/model.best')
        results = []
        for kkk in range(5000 / opts.batch_size):
            data = loader.get_batch('test')
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]
            fc_feats, att_feats = tmp
            image_id = [data['infos'][i]['id'] for i in range(batch_size)]
            att_feats = att_feats.reshape(att_feats.shape[0], att_feats.shape[1] * att_feats.shape[2],att_feats.shape[3])
            greedy_cap, greedy_res = model.inference(vocab, image_id, att_feats, manner='greedy', max_length=20)
            results += greedy_res
    else:
        from caption_model.fc import *
        vocab_size = 9489
        model = FCModel(batch_size=batch_size, image_dim=image_dim, vocab_size=vocab_size, cell_size=cell_size, lr=lr,on_gpu=True)
        model.load('sc_cider_model/fc_model/model.best')
        results = []
        for kkk in range(5000 / opts.batch_size):
            data = loader.get_batch('test')
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]
            fc_feats, att_feats = tmp
            image_id = [data['infos'][i]['id'] for i in range(batch_size)]
            greedy_cap, greedy_res = model.inference(vocab, image_id, fc_feats, manner='greedy', max_length=20)
            results += greedy_res

json.dump(results, open(osp.join(save_dir, 'result.json'), 'w'))
gt_file = osp.join('data/features', 'captions_test.json')
score = evaluate(gt_file=gt_file, re_file=osp.join(save_dir, 'result.json'))[-1]
bad_count = [count_bad(results[i]['caption']) for i in range(5000)]
total_bad_count = sum(bad_count)
print score , total_bad_count
