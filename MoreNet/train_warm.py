from caption_model.att import *
from caption_model.fc import *
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

if opts.use_att:
   save_dir = 'warm_model/' + 'att_warm'
else:
    save_dir = 'warm_model/' + 'fc_warm'
if not os.path.exists(save_dir):
   os.mkdir(save_dir)
print(save_dir + ' has been built')


image_dim = 2048
vocab_size = loader.vocab_size + 2 # set start token to 9488
cell_size = 512
lr = 0.00005
if opts.use_att:
    model = AttModel(batch_size=batch_size, image_dim=image_dim, vocab_size=vocab_size, cell_size=cell_size, lr=lr, on_gpu=True)
else:
    model = FCModel(batch_size=batch_size, image_dim=image_dim, vocab_size=vocab_size, cell_size=cell_size, lr=lr, on_gpu=True)


iters = 0
best_score = -1
train_losses = []
timer = Timer()
timer.tic()
logger = Logger(save_dir)
finish_iter = 1000000
while iters < 1000000:
    iters += 1
    data = loader.get_batch('train')
    tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks']]

    fc_feats, att_feats, labels, masks = tmp
    if opts.use_att:
        feature = att_feats.reshape(att_feats.shape[0],att_feats.shape[1]* att_feats.shape[2],att_feats.shape[3])
    else:
        feature = fc_feats
    Label = labels.transpose()
    Label[0,:] = 9488
    Mask = masks.transpose()[0:-1,:]
    reward = np.ones((opts.batch_size*5,))

    train_losses.append(model.train_on_batch(feature,Label, Mask, reward))
    # Validation
    if iters % 500 == 0 and iters >= 20000:
        results = []
        for nn in range(5000/opts.batch_size):
            datas = loader.get_batch('val')
            tmp = [datas['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   datas['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]
            fc_feats, att_feats = tmp
            if opts.use_att:
                feature = att_feats.reshape(att_feats.shape[0], att_feats.shape[1] * att_feats.shape[2],
                                            att_feats.shape[3])
            else:
                feature = fc_feats
            image_id = [datas['infos'][i]['id'] for i in range(opts.batch_size)]

            greedy_cap, greedy_res = model.inference(vocab,image_id,feature,manner='greedy',max_length=16)

            results += greedy_res

        json.dump(results, open(osp.join(save_dir, 'tmp_result.json'), 'w'))
        gt_file = osp.join('data/features', 'captions_val.json')
        score = evaluate(gt_file=gt_file, re_file=osp.join(save_dir, 'tmp_result.json'))[-1]

        if score > best_score:
            best_score = score
            model.save(osp.join(save_dir, 'model.init'))

        # Output training information
        logger.info('[{}], tr_loss={:.5f}, score/best={:.3f}/{:.3f}, finish->{}, time={:.1f}sec'
                    .format(iters, np.mean(train_losses), score, best_score, finish_iter, timer.toc()))
        # Reset loss and timer
        train_losses = []
        timer.tic()
