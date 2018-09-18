import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class LM(nn.Module):
    def __init__(self, batch_size, hidden_size,
                 vocab_size, word_embed_size,
                 lr, on_gpu=True):
        super(LM, self).__init__()
        self.lr = lr
        self.vocab_size = vocab_size
        self.word_embed_size = word_embed_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.on_gpu = on_gpu

        # word embedding layer
        self.word_embedding_layer = nn.Embedding(vocab_size,word_embed_size)

        # language model LSTM
        self.rnn = nn.LSTMCell(word_embed_size,hidden_size)

        # predict layer
        self.predict_layer = nn.Linear(hidden_size,vocab_size)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.onehot = torch.torch.eye(vocab_size)
        if self.on_gpu:
            self.onehot = self.onehot.cuda()
        # Move to gpu if necessary
        if self.on_gpu:
            self.cuda()
    def init_state(self):
        h = Variable(torch.zeros(self.batch_size, self.hidden_size))
        c = Variable(torch.zeros(self.batch_size, self.hidden_size))
        if self.on_gpu:
            h = h.cuda()
            c = c.cuda()
        return h,c

    def forward(self, word_emb, state):
        # Get states
        h_tm1, c_tm1 = state
        # RNN input
        h_t, c_t = self.rnn(word_emb, (h_tm1, c_tm1))
        # Next word's logtis
        logits = self.predict_layer(h_t)
        return logits, (h_t, c_t)

    def train_on_batch(self,sentence, mask, reward):
        # Convert numpy to torch
        sentence = Variable(torch.LongTensor(sentence.tolist()))
        mask = torch.Tensor(mask)
        reward = torch.Tensor(reward)
        T = sentence.size()[0] - 1
        # If using gpu

        sentence = sentence.cuda()
        mask = mask.cuda()
        reward = reward.cuda()

        # Initial state of RNN
        state = self.init_state()
        # Word embedding for input sequence
        inputs = self.word_embedding_layer(sentence[:-1, :])
        # Recurrent computation
        logits = []
        for i in xrange(T):
            word = inputs[i, :, :]
            logit, state = self.forward(word, state)
            logits.append(logit.unsqueeze(0))
        logits = torch.cat(logits, 0)  # (T, batch_size, vocab_size)
        logits = logits.resize(T*self.batch_size, self.vocab_size)
        # Next word's distribution
        prob = F.softmax(logits).data
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
        targets = targets.cpu().numpy()
        loss = - np.log(prob[np.arange(T * self.batch_size), targets])
        return loss.mean().numpy()

    def test_on_batch(self,sentence, mask):
        sentence = Variable(torch.LongTensor(sentence.tolist()))
        mask = torch.Tensor(mask)
        T = sentence.size()[0] - 1
        # If using gpu
        if self.on_gpu:
            sentence = sentence.cuda()
            mask = mask.cuda()
        # Initial state of RNN
        state = self.init_state()
        # Word embedding for input sequence
        inputs = self.word_embedding_layer(sentence[:-1, :])
        # Recurrent computation
        logits = []
        for i in xrange(T):
            word = inputs[i, :, :]
            logit, state = self.forward(word, state)
            logits.append(logit.unsqueeze(0))
        logits = torch.cat(logits, 0)  # (T, batch_size, vocab_size)
        logits = logits.resize(T * self.batch_size, self.vocab_size)
        # Next word's distribution
        prob = F.softmax(logits).data
        prob = prob.view(T,self.batch_size,self.vocab_size)
        # Ground-truth
        return prob

    def single_step_prob(self,word,state):
        word = Variable(torch.LongTensor(word.tolist()))
        if self.on_gpu:
            word = word.cuda()
        word_emb = self.word_embedding_layer(word)
        logit, state2 = self.forward(word_emb, state)   # logit : (batch_size, vocab_size)
        prob = F.softmax(logit).data   # (batch_size, vocab_size)
        return prob,state2

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.load_state_dict(torch.load(f))
