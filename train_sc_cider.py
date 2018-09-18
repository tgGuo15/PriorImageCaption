from caption_model.att import *
from caption_model.fc import *
from mycider import *
from multiprocessing import Pool
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import ngram_opts
from tools import *
from dataloader import *
opts = ngram_opts.parse_opt()
if opts.caption_model == 'fc':
    opts.use_att = False
else:
    opts.use_att = True

batch_size = opts.batch_size

loader = KKDataLoader(opts)
vocabs = loader.get_vocab()
vocab = ['#END#']
for i in range(len(vocabs)):
    ids = str(i+1)
    vocab.append(vocabs[ids])

if not os.path.exists('sc_cider_model'):
    os.mkdir('sc_cider_model')

if opts.use_att:
    save_dir = 'sc_cider_model/' + 'att_model'
else:
    save_dir = 'sc_cider_model/' + 'fc_model'
if not os.path.exists(save_dir):
   os.mkdir(save_dir)
print(save_dir + ' has been built')


image_dim = 2048
vocab_size = loader.vocab_size + 1
cell_size = 512
lr = 0.00005
if opts.use_att:
    model = AttModel(batch_size=batch_size, image_dim=image_dim, vocab_size=vocab_size, cell_size=cell_size, lr=lr, ngram=0,on_gpu=True)
    model.load('warm_model/att_warm/model.init')
else:
    model = FCModel(batch_size=batch_size, image_dim=image_dim, vocab_size=vocab_size, cell_size=cell_size, lr=lr, ngram=0,on_gpu=True)
    model.load('warm_model/fc_warm/model.init')


# Initialize cider-scorer
gts = transfer_json_to_cider_gts(osp.join('data/features', 'captions_train.json'))
cider_scorer = CiderScorer(refs=gts, n=4, sigma=6.0)

def cider_temp(res):
    cider_scorer.cook_append_test(test={res['image_id']: [res['caption']]})
    score, _ = cider_scorer.compute_score()
    return score

pool = Pool(processes=5)

best_score = -1
logger = Logger(save_dir)
iter = 0
finish_iter = 1000000
timer = Timer()
timer.tic()
while iter < finish_iter:
    iter += 1
    data = loader.get_batch('train')
    tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
           data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]
    fc_feats, att_feats = tmp
    image_id = [data['infos'][i]['id'] for i in range(opts.batch_size)]
    if opts.use_att:
        att_feats = att_feats.reshape(att_feats.shape[0], att_feats.shape[1] * att_feats.shape[2], att_feats.shape[3])
        feature = att_feats
    else:
        feature = fc_feats
    greedy_cap, greedy_res = model.inference(vocab, image_id, feature, manner='greedy', max_length=16)
    greedy_scores = np.array(pool.map(cider_temp, greedy_res))

    all_caps, all_results, all_scores = [], [], []
    for _ in xrange(20):
        # Generate captions by sampling
        sample_caps, sample_results = model.inference(vocab, image_id, feature,
                                                      manner='sample',
                                                      max_length=16)

        # Compute cider scores for sampled captions
        sample_scores = np.array(pool.map(cider_temp, sample_results))
        all_caps.append(sample_caps)
        all_results.append(sample_results)
        all_scores.append(sample_scores)

    all_scores = np.array(all_scores)
    sample_caps, sample_results, sample_scores = [], [], []
    for n in xrange(opts.batch_size):
        best_i = all_scores[:, n].argmax()
        sample_caps.append(all_caps[best_i][n])
        sample_results.append(all_results[best_i][n])
        sample_scores.append(all_scores[best_i, n])
    sample_scores = np.array(sample_scores)

    max_length = max([cap.shape[0] for cap in sample_caps])
    caption = np.zeros([max_length + 2, opts.batch_size], dtype=np.int32)
    for n in xrange(opts.batch_size):
        L = sample_caps[n].shape[0]
        caption[1:L + 1, n] = sample_caps[n]
        caption[L + 1:, n] = 0
    mask = np.zeros([max_length + 1, opts.batch_size], dtype=np.float32)
    for n in xrange(opts.batch_size):
        L = sample_caps[n].shape[0]
        mask[:L + 1, n] = 1
    reward = (sample_scores - greedy_scores).astype(np.float32)
    print image_id[0]
    print 'greedy: ', greedy_scores[0], greedy_res[0]['caption']
    print 'sample: ', sample_scores[0], sample_results[0]['caption']
    loss_train = model.train_on_batch(feature, caption[1:,:], mask, reward)
    if iter % 300 == 0:
        results = []
        for nn in range(5000/opts.batch_size):
            data = loader.get_batch('val')
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]
            fc_feats, att_feats = tmp
            if opts.use_att:
                att_feats = att_feats.reshape(att_feats.shape[0], att_feats.shape[1] * att_feats.shape[2],
                                         att_feats.shape[3])
                feature_val = att_feats
            else:
                feature_val = fc_feats
            image_id = [data['infos'][i]['id'] for i in range(opts.batch_size)]

            greedy_cap, greedy_res = model.inference(vocab,image_id,feature_val,manner='greedy',max_length=16)
            # Generate sentences for validation set
            results += greedy_res
        # Evaluate generated captions
        json.dump(results, open(osp.join(save_dir, 'result.json'), 'w'))
        gt_file = osp.join('data/features', 'captions_val.json')
        score = evaluate(gt_file=gt_file, re_file=osp.join(save_dir, 'result.json'))[-1]
       # json.dump(results, open(osp.join(save_dir, 'kk_rl_result_'+ str(iter) + '.json'), 'w'))
        # Update if finding new best model
        if score > best_score:
            best_score = score
            model.save(osp.join(save_dir, 'model.best'))
        model.save(osp.join(save_dir,'model.ckpt'))
        # Output training information
        logger.info('[{}], tr_loss={:.5f}, score/best={:.3f}/{:.3f}, finish->{}, time={:.1f}sec'
                    .format(iter, -1, score, best_score, finish_iter, timer.toc()))
        # Reset loss and timer
        train_losses = []
        timer.tic()

    # If early-stop condition triggers
    if iter > finish_iter:
        break

