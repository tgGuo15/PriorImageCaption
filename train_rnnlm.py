import ngram_opts
from dataloader import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from caption_model.rnnlm import *

opts = ngram_opts.parse_opt()
if opts.caption_model == 'fc':
    opts.use_att = False
else:
    opts.use_att = True
loader = KKDataLoader(opts)


batch_size = 100
hidden_size = 256
word_embed_size = 256
vocab_size = loader.vocab_size + 2  # set start token
lr = 0.0001
lm = LM(batch_size, hidden_size,vocab_size, word_embed_size,lr)
Labels = loader.h5_label_file['labels']
new_labels = np.zeros((Labels.shape[1]+1,Labels.shape[0]),dtype=Labels.dtype)
new_labels[0,:] = 9488   # Set start token to 9488, the total vocab size is 9489
for i in range(Labels.shape[0]):
    new_labels[1:,i] = Labels[i,:]

Ind = range(len(Labels))
mask = np.ones((16,100))
reward = np.ones((100,))
import random
for i in range(1000):
    random.shuffle(Ind)
    Loss = []
    for j in range(100):
        index = Ind[j*batch_size:(j+1)*batch_size]
        batch_sen = new_labels[:,index]
        loss = lm.train_on_batch(batch_sen,mask,reward)
        Loss.append(loss)
    print i,np.mean(Loss)
    if i % 10 == 0:
       lm.save('warm_model/rnnlm/model.init')

