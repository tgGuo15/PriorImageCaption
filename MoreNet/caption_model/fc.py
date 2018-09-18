import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import pickle
class FCModel(nn.Module):
    def __init__(self, batch_size, cell_size, image_dim,
                 vocab_size, lr,ngram=0, on_gpu=False):
        super(FCModel, self).__init__()
        # Settings
        self.batch_size = batch_size
        self.cell_size = cell_size
        self.image_dim = image_dim
        self.vocab_size = vocab_size
        self.lr = lr
        self.on_gpu = on_gpu

        # Word embedding lookup table
        self.word_embedding = nn.Embedding(vocab_size, cell_size)

        # Image embedding mlp
        self.image_embedding = nn.Linear(image_dim, cell_size, bias=False)

        # State initializer
        self.c_initializer = nn.Linear(cell_size, cell_size, bias=False)
        self.h_initializer = nn.Linear(cell_size, cell_size, bias=False)

        # Recurrent layer
        self.rnn = nn.LSTMCell(cell_size, cell_size)

        # Word predicting mlp
        self.predictor = nn.Linear(cell_size, vocab_size)
        if ngram == 3:
            trigram = pickle.load(open('data/trigram.pkl'))
            self.trigram_mask = {}
            for tri in trigram:
                temp = np.zeros((vocab_size,))
                for word in trigram[tri]:
                    temp[word] = 1
                self.trigram_mask[tri] = temp
        elif ngram == 4:
            fourgram = pickle.load(open('data/fourgram.pkl'))
            self.fourgram_mask = {}
            for four in fourgram:
                temp = np.zeros((vocab_size,))
                for word in fourgram[four]:
                    temp[word] = 1
                self.fourgram_mask[four] = temp

        # Onehot encoder
        self.onehot = torch.torch.eye(vocab_size)
        if self.on_gpu:
            self.onehot = self.onehot.cuda()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # Move to gpu if necessary
        if self.on_gpu:
            self.cuda()

    def forward(self, word_emb, state):
        # Get states
        h_tm1, c_tm1 = state

        # RNN input
        h_t, c_t = self.rnn(word_emb, (h_tm1, c_tm1))

        # Next word's logtis
        logits = self.predictor(h_t)

        return logits, (h_t, c_t)

    def initial_state(self, image):
        image = Variable(torch.Tensor(image))
        if self.on_gpu:
            image = image.cuda()

        # Image embedding
        feat = self.image_embedding(image)

        # Initial state (batch_size, rnn_size)
        h0 = nn.Tanh()(self.h_initializer(feat))
        c0 = nn.Tanh()(self.c_initializer(feat))

        return h0, c0

    def train_on_batch(self, image, sentence, mask, reward):
        # Convert numpy to torch
        self.batch_size = image.shape[0]
        sentence = Variable(torch.LongTensor(sentence.tolist()))
        mask = torch.Tensor(mask)
        reward = torch.Tensor(reward)
        T = sentence.size()[0] - 1

        # If using gpu
        if self.on_gpu:
            sentence = sentence.cuda()
            mask = mask.cuda()
            reward = reward.cuda()

        # Initial state of RNN
        state = self.initial_state(image)

        # Word embedding for input sequence
        inputs = self.word_embedding(sentence[:-1, :])

        # Recurrent computation
        logits = []
        for i in xrange(T):
            word = inputs[i, :, :]
            logit, state = self.forward(word, state)
            logits.append(logit.unsqueeze(0))
        logits = torch.cat(logits, 0)  # (T, batch_size, vocab_size)
        logits = logits.resize(T*self.batch_size, self.vocab_size)

        # Next word's distribution
        prob = nn.Softmax()(logits).data

        # Ground-truth
        targets = sentence.data[1:, :].view(T*self.batch_size)
        gt_prob = self.onehot.index_select(0, targets)

        # Gradients
        logit_grad = (prob - gt_prob).view(T, self.batch_size, self.vocab_size)
        logit_grad = logit_grad * mask.view(T, self.batch_size, 1).expand_as(logit_grad)
        logit_grad = logit_grad * reward.view(1, self.batch_size, 1).expand_as(logit_grad)
        logit_grad = logit_grad / mask.sum(0).view(1, self.batch_size, 1).expand_as(logit_grad)
        logit_grad = logit_grad / self.batch_size
        logit_grad = logit_grad.view(T*self.batch_size, self.vocab_size)

        # Gradient descent
        self.optimizer.zero_grad()
        logits.backward(gradient=logit_grad)
        self.optimizer.step()
        loss = -1

        return loss

    def single_step(self, state, words, manner='greedy'):
        words = Variable(torch.LongTensor(words), volatile=True)
        if self.on_gpu:
            words = words.cuda()

        # Word embedding
        words = self.word_embedding(words.unsqueeze(0)).squeeze(0)

        # Take a rnn step
        logits, new_state = self.forward(words, state)

        # Next words
        if manner == 'greedy':
            new_words = logits.data.cpu().numpy().argmax(1)
        elif manner == 'sample':
            # Gumbel argmax trick
            if self.on_gpu:
                U = torch.cuda.FloatTensor(self.batch_size, self.vocab_size).uniform_(0, 1)
            else:
                U = torch.FloatTensor(self.batch_size, self.vocab_size).uniform_(0, 1)
            V = logits.data - torch.log(-U.log())
            new_words = V.cpu().numpy().argmax(1)
        else:
            raise ValueError('Unknown manner: [{}]'.format(manner))

        return new_state, new_words

    def inference(self, vocab, image_ids, image, manner='greedy',
                  max_length=16, verbose=0, batch_size=None):
        # Choose batch-size
        self.batch_size = image.shape[0]

        # Beginning tokens
        init_word = torch.LongTensor([0] * self.batch_size)

        # Initiazization
        results = []
        captions = []

        # Iteratively generate words
        state = self.initial_state(image)
        sentences = []
        word = init_word
        for _ in xrange(max_length):
            state, word = self.single_step(state, word, manner=manner)
            sentences.append(word)
        sentences = np.array(sentences).transpose()

        # Translate indexes to sentences
        for j in xrange(sentences.shape[0]):
            idxs = np.where(sentences[j, :] == 0)[0]
            end_index = idxs[0] if len(idxs) > 0 else max_length
            cap = ' '.join([vocab[w] for w in sentences[j, :end_index]])
            if verbose > 0:
                print 'id={}, {}'.format(image_ids[j], cap)
            captions.append(sentences[j, :end_index])
            results.append({'image_id': image_ids[j], 'caption': cap})
        # Type: captions (np.array), results (natural language)
        return captions, results

    def ngram_single_step(self, state, words,temp_mask, manner='greedy'):
        words = Variable(torch.LongTensor(words), volatile=True)
        Temp_mask = Variable(torch.Tensor(temp_mask))
        if self.on_gpu:
            words = words.cuda()
            Temp_mask = Temp_mask.cuda()
        Temp_mask = Temp_mask * 100000 - 100000
        # Word embedding
        words = self.word_embedding(words.unsqueeze(0)).squeeze(0)

        # Take a rnn step
        logits, new_state = self.forward(words, state)
        logits = logits + Temp_mask
        # Next words
        if manner == 'greedy':
            new_words = logits.data.cpu().numpy().argmax(1)
        elif manner == 'sample':
            # Gumbel argmax trick
            if self.on_gpu:
                U = torch.cuda.FloatTensor(self.batch_size, self.vocab_size).uniform_(0, 1)
            else:
                U = torch.FloatTensor(self.batch_size, self.vocab_size).uniform_(0, 1)
            V = logits.data - torch.log(-U.log())
            new_words = V.cpu().numpy().argmax(1)
        else:
            raise ValueError('Unknown manner: [{}]'.format(manner))

        return new_state, new_words

    def fourgram_inference(self, vocab, image_ids, image, manner='greedy',
                  max_length=16):
        # Choose batch-size
        self.batch_size = image.shape[0]

        # Beginning tokens
        init_word = torch.LongTensor([9488] * self.batch_size)

        # Initiazization
        results = []
        captions = []

        # Iteratively generate words
        state = self.initial_state(image)
        sentences = []
        word = init_word
        sentencemask = np.zeros((max_length + 3, self.batch_size), dtype=np.int32)
        sentencemask[0:3,:] = 9488
        for jj in xrange(max_length):
            temp_mask = self.get_four_Mask(sentencemask,jj+3)
            state, word = self.ngram_single_step(state, word,temp_mask,manner=manner)
            sentencemask[jj+3,:] = word
            sentences.append(word)
        sentences = np.array(sentences).transpose()

        # Translate indexes to sentences
        for j in xrange(sentences.shape[0]):
            idxs = np.where(sentences[j, :] == 0)[0]
            end_index = idxs[0] if len(idxs) > 0 else max_length
            cap = ' '.join([vocab[w] for w in sentences[j, :end_index]])
            captions.append(sentences[j, :end_index])
            results.append({'image_id': image_ids[j], 'caption': cap})

        # Type: captions (np.array), results (natural language)
        return captions, results

    def get_four_Mask(self,sentencemask,index):
        tempmask = np.zeros((self.batch_size,self.vocab_size))
        for hh in range(self.batch_size):
            temp = tuple(list(sentencemask[index-3:index,hh]))
            if temp in self.fourgram_mask:
                tempmask[hh] = self.fourgram_mask[temp]
            else:
                tempmask[hh][0] = 1    # END token
        return tempmask

    def trigram_inference(self, vocab, image_ids, image, manner='greedy',max_length=16):
        # Choose batch-size
        self.batch_size = image.shape[0]
        results = []
        captions = []

        # Iteratively generate words
        state = self.initial_state(image)
        sentences = []
        init_word = torch.LongTensor([9488] * self.batch_size)
        word = init_word
        sentencemask = np.zeros((max_length + 2, self.batch_size), dtype=np.int32)
        sentencemask[0:2,:] = 9488
        for jj in xrange(max_length):
            temp_mask = self.get_tri_Mask(sentencemask,jj+2)
            state, word = self.ngram_single_step(state, word,temp_mask,manner=manner)
            sentencemask[jj+2,:] = word
            sentences.append(word)
        sentences = np.array(sentences).transpose()

        # Translate indexes to sentences
        for j in xrange(sentences.shape[0]):
            idxs = np.where(sentences[j, :] == 0)[0]
            end_index = idxs[0] if len(idxs) > 0 else max_length
            cap = ' '.join([vocab[w] for w in sentences[j, :end_index]])
            captions.append(sentences[j, :end_index])
            results.append({'image_id': image_ids[j], 'caption': cap})

        # Type: captions (np.array), results (natural language)
        return captions, results

    def get_tri_Mask(self,sentencemask,index):
        tempmask = np.zeros((self.batch_size,self.vocab_size))
        for hh in range(self.batch_size):
            temp = tuple(list(sentencemask[index-2:index,hh]))
            if temp in self.trigram_mask:
                tempmask[hh] = self.trigram_mask[temp]
            else:
                tempmask[hh][0] = 1    # END token
        return tempmask

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.load_state_dict(torch.load(f))

