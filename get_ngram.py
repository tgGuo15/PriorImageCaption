import json
from dataloader import *
from tools import remove_badEnding
import pickle
coco = json.load(open('data/cocotalk.json'))
ix_to_word = coco['ix_to_word']

vocab = {}
for i in ix_to_word:
    vocab[ix_to_word[i]] = int(i)


fourgram = {}
trigram = {}
vocab_size = 9488  # 0:'#END'  9487:'UNK'    9488:'#BEGIN'

train_data = json.load(open('data/captions_train.json'))
for all_cap in train_data['captions']:
    for cap in all_cap:
        tokens = cap.split(' ')
        L = len(tokens)
        index = []
        for i in range(len(tokens)):
            if tokens[i] in vocab:
                index.append(vocab[tokens[i]])
            else:
                index.append(vocab_size-1)
        index += [0] * (16-L)
        fourgram_seq = [vocab_size,vocab_size,vocab_size] + index
        trigram_seq = [vocab_size,vocab_size] + index
        for j in range(16):
            fourgram_tuple = tuple(fourgram_seq[j:j+3])
            trigram_tuple = tuple(trigram_seq[j:j+2])
            if fourgram_tuple not in fourgram:
                fourgram[fourgram_tuple] = {}
                fourgram[fourgram_tuple][index[j]] = 1
            else:
                if index[j] not in fourgram[fourgram_tuple]:
                    fourgram[fourgram_tuple][index[j]] = 1
                else:
                    fourgram[fourgram_tuple][index[j]] += 1

            if trigram_tuple not in trigram:
                trigram[trigram_tuple] = {}
                trigram[trigram_tuple][index[j]] = 1
            else:
                if index[j] not in trigram[trigram_tuple]:
                    trigram[trigram_tuple][index[j]] = 1
                else:
                    trigram[trigram_tuple][index[j]] += 1

f = open('data/fourgram.pkl','w')
pickle.dump(fourgram,f)
f.close()

f = open('data/trigram.pkl','w')
pickle.dump(trigram,f)
f.close()
