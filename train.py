import os
import sys
import numpy as np
import torch
import torch.autograd as autograd
from torchtext import data
from args import get_args
from attentive_cnn_data import SimpleQADataset
from model import AttentiveCNN
import random
import time
from eval import evaluation

args = get_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if torch.cuda.is_available() and args.cuda:
    print("Note: You are using GPU for training")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("Warning: You have Cuda but not use it. You are using CPU for training.")
    args.gpu = -1

# Support More QA dataset if possible
if args.dataset == 'SimpleQA':
    ID = data.Field(sequential=False, tensor_type=torch.LongTensor, batch_first=True, use_vocab=False,
                    postprocessing=data.Pipeline(lambda x, train: int(x)))
    WORD = data.Field(batch_first=True)
    CHAR = data.Field(batch_first=True)
    SCORE = data.Field(sequential=False, tensor_type=torch.FloatTensor, batch_first=True, use_vocab=False, \
                       preprocessing=data.Pipeline(lambda x: x.split()), \
                       postprocessing=data.Pipeline(lambda x, train: [float(y) for y in x]))
    LABEL = data.Field(batch_first=True, sequential=False)
    train, dev, test = SimpleQADataset.splits(ID, WORD, CHAR, SCORE, LABEL)
    WORD.build_vocab(train, dev, test)
    CHAR.build_vocab(train, dev, test)
    LABEL.build_vocab(train, dev, test)
else:
    print("You specify a wrong dataset")
    exit()

# Original paper uses randomly initialized embedding
'''
if os.path.isfile(args.vector_cache):
    stoi, vectors, dim = torch.load(args.vector_cache)
    WORD.vocab.vectors = torch.Tensor(len(WORD.vocab), dim)
    for i, token in enumerate(WORD.vocab.itos):
        wv_index = stoi.get(token, None)
        if wv_index is not None:
            WORD.vocab.vectors[i] = vectors[wv_index]
        else:
            WORD.vocab.vectors[i] = torch.Tensor.zero_(WORD.vocab.vectors[i])
else:
    print("Error: Need word embedding pt file")
    exit(1)
'''


train_iter = data.Iterator(train, batch_size=args.batch_size, device=args.gpu, train=True, repeat=False,
                                   sort=False, shuffle=True)
dev_iter = data.Iterator(dev, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                                   sort=False, shuffle=False)
test_iter = data.Iterator(test, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                                   sort=False, shuffle=False)

config = args
config.target_class = len(LABEL.vocab)
config.words_num = len(WORD.vocab)
config.chars_num = len(CHAR.vocab)
print(CHAR.vocab.itos)

model = AttentiveCNN(config)
#model.static_embed.weight.data.copy_(WORD.vocab.vectors)
if args.cuda:
    model.cuda()
    print("Shift model to GPU")

print(model)
parameter = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adagrad(parameter, lr=args.lr, lr_decay=args.lr_decay, weight_decay=args.weight_decay)
criterion = torch.nn.MarginRankingLoss(margin=0.5)
early_stop = False
best_dev_acc = 0
iterations = 0
iters_not_improved = 0
epoch = 0
start = time.time()
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Best Acc on Dev  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{:12.4f}'.split(','))
log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
os.makedirs(args.save_path, exist_ok=True)
os.makedirs(os.path.join(args.save_path, args.dataset), exist_ok=True)
print(header)
index2word = np.array(WORD.vocab.itos)
index2char = np.array(CHAR.vocab.itos)
index2label = np.array(LABEL.vocab.itos)

while True:
    if early_stop:
        print("Early Stopping. Epoch: {}, Best Dev Acc: {}".format(epoch, best_dev_acc))
        break
    epoch += 1
    train_iter.init_epoch()

    for batch_idx, batch in enumerate(train_iter):
        iterations += 1
        model.train(); optimizer.zero_grad()
        pos, neg = model(batch)
        if (args.cuda):
            y = autograd.Variable(torch.Tensor(batch.batch_size).fill_(1).cuda())
        else:
            y = autograd.Variable(torch.Tensor(batch.batch_size).fill_(1))
        #print("Pos: {} Neg: {}, y: {}".format(pos, neg, y))
        loss = criterion(pos, neg, y)
        loss.backward()
        optimizer.step()
        #print("Loss: {}".format(loss.data[0]))
        #exit()

        if iterations % args.dev_every == 1:
            model.eval(); dev_iter.init_epoch()
            data_to_eval = []
            for dev_batch_idx, dev_batch in enumerate(dev_iter):
                score, _ = model(dev_batch)
                temp_id = dev_batch.id1.cpu().data.numpy()
                temp_pattern = index2word[dev_batch.pattern1.cpu().data.numpy()]
                temp_predicate = index2word[dev_batch.predicate1.cpu().data.numpy()]
                temp_mention = index2char[dev_batch.mention1.cpu().data.numpy()]
                temp_candidate = index2char[dev_batch.candidate1.cpu().data.numpy()]
                temp_score = score.cpu().data.numpy()
                for i in range(dev_batch.batch_size):
                    data_to_eval.append((int(temp_id[i]), list(temp_pattern[i]), list(temp_predicate[i]), list(temp_mention[i]), \
                                     list(temp_candidate[i]), float(temp_score[i])))

            #print("size of eval data", len(data_to_eval))
            accuracy = evaluation(data_to_eval, "data/attentivecnn.valid.gold")
            print(dev_log_template.format(time.time() - start,
                                          epoch, iterations, 1 + batch_idx, len(train_iter),
                                          100. * (1 + batch_idx) / len(train_iter), loss.data[0],
                                          "N/A     ", best_dev_acc* 100.0, accuracy * 100.0))
            if accuracy > best_dev_acc:
                iters_not_improved = 0
                best_dev_acc = accuracy
                snapshot_path = os.path.join(args.save_path, args.dataset, args.specify_prefix +
                                             '_best_model.pt')
                torch.save(model, snapshot_path)
            else:
                iters_not_improved += 1
                if iters_not_improved >= args.patience:
                    early_stop = True
                    break


        if iterations % args.log_every == 1:
            print(log_template.format(time.time() - start,
                                      epoch, iterations, 1 + batch_idx, len(train_iter),
                                      100. * (1 + batch_idx) / len(train_iter), loss.data[0], ' ' * 8,
                                      best_dev_acc * 100, ' ' * 12))

