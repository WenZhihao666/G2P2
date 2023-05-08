from __future__ import division
from torch.utils.data import Dataset
import numpy as np

class DataHelper(Dataset):
    def __init__(self, edge_index, args, the_nodes, directed=False, transform=None):
        # self.num_nodes = len(node_list)
        self.transform = transform
        self.degrees = dict()
        self.node_set = set()
        self.neighs = dict()
        self.args = args

        idx, degree = np.unique(edge_index, return_counts=True)
        # for (a, b) in (idx, degree):
        for i in range(idx.shape[0]):
            self.degrees[idx[i]] = degree[i].item()

        self.node_dim = idx.shape[0]

        train_edge_index = edge_index  # .T#[:int(0.8 * edge_nums)]

        self.final_edge_index = train_edge_index.T

        for i in range(self.final_edge_index.shape[0]):
            s_node = self.final_edge_index[i][0].item()
            t_node = self.final_edge_index[i][1].item()

            if s_node not in self.neighs:
                self.neighs[s_node] = []
            if t_node not in self.neighs:
                self.neighs[t_node] = []

            self.neighs[s_node].append(t_node)
            if not directed:
                self.neighs[t_node].append(s_node)

        self.idx = idx
        self.the_nodes = the_nodes

    def __len__(self):
        return len(self.the_nodes)

    def __getitem__(self, idx):

        s_n = self.the_nodes[idx]#.item()
        if len(self.neighs[s_n]) > self.args.neigh_num:
            t_n = np.random.choice(self.neighs[s_n], self.args.neigh_num, replace=False)
        else:
            t_n = np.random.choice(self.neighs[s_n], self.args.neigh_num, replace=True)
        # t_n = np.array(t_n)

        sample = {
            's_n': s_n,  # e.g., 5424
            't_n': t_n,  # e.g., 5427
            # 'neg_n': neg_n
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
