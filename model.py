import torch
import torch.nn as nn
import torch.autograd as autograd

import torch.nn.functional as F
import numpy as np
np.set_printoptions(threshold=np.nan)


class AttentiveCNN(nn.Module):
    def __init__(self, config):
        super(AttentiveCNN, self).__init__()
        self.char_embed = nn.Embedding(config.chars_num, config.char_dim)
        self.word_embed = nn.Embedding(config.words_num, config.word_dim)
        self.char_conv = nn.Conv2d(1, config.char_dim, kernel_size=(3, config.char_dim), padding=(2,0))
        self.word_conv = nn.Conv2d(1, config.word_dim, kernel_size=(3, config.word_dim), padding=(2, 0))
        self.cosine = torch.nn.CosineSimilarity(dim=2)
        self.match = torch.nn.CosineSimilarity(dim=1)
        if config.cuda:
            self.zero = autograd.Variable(torch.FloatTensor([0]).cuda())
        else:
            self.zero = autograd.Variable(torch.FloatTensor([0]))

    def scnn(self, mention, candidate):
        q = self.char_embed(mention).unsqueeze(1)

        a = self.char_embed(candidate).unsqueeze(1)
        # q = (batch, inputchannel, sentence_len, word_dim)
        q = F.tanh(self.char_conv(q)).squeeze(3)
        # q = (batch, outputchannel, sentence_len, 1)
        a = F.tanh(self.char_conv(a)).squeeze(3)
        q = F.max_pool1d(q, q.size(2)).squeeze(2)
        a = F.max_pool1d(a, a.size(2)).squeeze(2)
        x = self.match(q, a)
        return x

    def ampcnn(self, pattern, predicate):
        q = self.word_embed(pattern).unsqueeze(1)
        a = self.word_embed(predicate).unsqueeze(1)
        q = F.tanh(self.word_conv(q)).squeeze(3).transpose(1, 2).contiguous()
        a = F.tanh(self.word_conv(a)).squeeze(3)
        a = F.max_pool1d(a, a.size(2)).squeeze(2)
        cos = self.cosine(q, a.unsqueeze(1))
        if len(cos.size()) == 1:
            cos = cos.unsqueeze(0)
        # set negative cosine value to zero and regularize
        # cos = (batch, ~=sentenceLen)
        # q = (batch, ~=sentenceLen, outputChannel(=word_dim))
        max0_cos = torch.max(cos, self.zero)
        ad_cos = max0_cos / torch.max(max0_cos, dim=1)[0].unsqueeze(1).clamp(min=1e-8)

        modify_q = q * ad_cos.unsqueeze(2)
        modify_q = modify_q.transpose(1, 2).contiguous()
        _, pool_index = F.max_pool1d(modify_q, modify_q.size(2), return_indices=True)
        if (pool_index.cpu().data.numpy() < 0).any():
            #print(modify_q.cpu().data.numpy())
            #print(pool_index.cpu().data.numpy())
            print(cos.cpu().data.numpy())
            print(ad_cos.cpu().data.numpy())
        q = q.transpose(1, 2).contiguous()
        q = torch.gather(q, 2, pool_index).squeeze(2)
        x = self.match(q, a)
        return x

    def forward(self, x):
        positive = self.scnn(x.mention1, x.candidate1) + self.ampcnn(x.pattern1, x.predicate1) + x.score1.squeeze(1)
        negative = self.scnn(x.mention2, x.candidate2) + self.ampcnn(x.pattern2, x.predicate2) + x.score2.squeeze(1)
        return positive, negative




