from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description="Kim CNN")
    parser.add_argument('--no_cuda', action='store_false', help='do not use cuda', dest='cuda')
    parser.add_argument('--gpu', type=int, default=0) # Use -1 for CPU
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--seed', type=int, default=3435)
    parser.add_argument('--dataset', type=str, default='SimpleQA')
    parser.add_argument('--resume_snapshot', type=str, default=None)
    parser.add_argument('--dev_every', type=int, default=400)
    parser.add_argument('--log_every', type=int, default=40)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--save_path', type=str, default='saves')
    parser.add_argument('--specify_prefix', type=str, default='')
    parser.add_argument('--word_dim', type=int, default=500)
    parser.add_argument('--char_dim', type=int, default=100)
    # parser.add_argument('--embed_dim', type=int, default=300)
    # parser.add_argument('--pos_dim', type=int, default=50)
    # parser.add_argument('--dep_dim', type=int, default=50)
    # parser.add_argument('--ind_dim', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--epoch_decay', type=int, default=15)
    parser.add_argument('--vector_cache', type=str, default="data/word2vec.sq.pt")
    parser.add_argument('--trained_model', type=str, default="")
    parser.add_argument('--weight_decay',type=float, default=0)


    args = parser.parse_args()
    return args
