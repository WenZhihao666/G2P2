
from torch.utils.data import DataLoader
from sklearn import preprocessing
import numpy as np
import argparse
import torch
from random import sample
import random
import math
import time
from model import CLIP, tokenize
from data import DataHelper
from sklearn import preprocessing


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    setup_seed(seed)


    model = CLIP(args).to(device)
    Data = DataHelper(arr_edge_index, args)
    model.train()

    for j in range(args.epoch_num):
        loader = DataLoader(Data, batch_size=args.batch_size, shuffle=True, num_workers=10)
        for i_batch, sample_batched in enumerate(loader):
            s_n, t_n = sample_batched['s_n'], sample_batched['t_n']
            s_n_arr, t_n_arr = s_n.numpy(), t_n.numpy().reshape(-1)  # .reshape((1, -1))
            s_n_text, t_n_text = np.array(tit_list)[s_n_arr].tolist(), np.array(tit_list)[t_n_arr].tolist()
            s_n_text, t_n_text = tokenize(s_n_text, context_length=args.context_length).to(device), tokenize(t_n_text, context_length=args.context_length).to(device)

            s_n, t_n = s_n.type(LType).to(device), t_n.type(LType).to(device)
            loss = model.forward(node_f, edge_index, s_n, t_n, s_n_text, t_n_text, device)
            # if i_batch >2 :
            #     break
            if j == 0 and i_batch % 100 == 0:
                print('{}th loss in the first epoch:{}'.format(i_batch, loss))

        # break
        print('{}th epoch loss:{}'.format(j, loss))

    torch.save(model.state_dict(), './res/{}/node_ttgt_8&12_0.1.pkl'.format(data_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--aggregation_times', type=int, default=2, help='Aggregation times')
    parser.add_argument('--epoch_num', type=int, default=2, help='epoch number')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--edge_coef', type=float, default=10)
    parser.add_argument('--neigh_num', type=int, default=3)

    parser.add_argument('--gnn_input', type=int, default=128)
    parser.add_argument('--gnn_hid', type=int, default=128)
    parser.add_argument('--gnn_output', type=int, default=128)

    parser.add_argument('--context_length', type=int, default=128)

    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--transformer_heads', type=int, default=8)
    parser.add_argument('--transformer_layers', type=int, default=12)
    parser.add_argument('--transformer_width', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=49408)  # 49408
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    data_name = 'cora'
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print('device:', device)


    num_nodes = 0
    tit_list = []
    with open('./data/train_text.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            tit_list.append(line[2])
            num_nodes += 1

    print('num_nodes', num_nodes)

    raw_edge_index = [[], []]
    with open('./data/mapped_edges.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            raw_edge_index[0].append(int(line[0]))
            raw_edge_index[1].append(int(line[1]))

    print('num of edges', len(raw_edge_index[0] + raw_edge_index[1]))

    edge_index = [raw_edge_index[0] + raw_edge_index[1], raw_edge_index[1] + raw_edge_index[0]]
    arr_edge_index = np.array(edge_index)
    edge_index = np.array(edge_index)
    edge_index = torch.from_numpy(edge_index).to(device)

    node_f = np.load('./data/node_f.npy')
    node_f = preprocessing.StandardScaler().fit_transform(node_f)
    node_f = torch.from_numpy(node_f).to(device)

    start = time.perf_counter()

    seed = 1
    main(args)

    end = time.perf_counter()
    print("time consuming {:.2f}".format(end - start))
