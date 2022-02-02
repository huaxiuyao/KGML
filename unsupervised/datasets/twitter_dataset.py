import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from .utils.data_reader import read_dir
from .utils.language_utils import line_to_indices, get_word_emb_arr, val_to_vec
import numpy as np
from itertools import chain
VOCAB_DIR = 'data/twitter/embs.json'


class TwitterDataset(Dataset):

    def __init__(self, split, root_dir, tokenizer, max_words=10):
        self.root_dir = Path(root_dir)/split
        self.seq_len = 10
        self.num_classes = 2
        self.tokenizer = tokenizer
        users, _, data = read_dir(self.root_dir)
        assert len(users) == len(set(users)), 'duplicate users'

        self.users = users
        self.n_groups = len(users)
        self.groups = [i for i in range(self.n_groups)] # unique
        # _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
        # self.vocab_size = len(vocab)
        # N=20152, number of records
        self.raw_x = [data[u]['x'][i][4] for u in self.users for i in range(len(data[u]['x']))]
        self.raw_y = [data[u]['y'][i] for u in self.users for i in range(len(data[u]['y']))]

        inds = []
        masks = []
        for e in self.raw_x:
            ind, mask = line_to_indices(e, self.tokenizer, max_words)
            inds.append(ind); masks.append(mask)

        self.pro_x = torch.tensor(inds).long() # (N, 25)
        self.data_mask = torch.tensor(masks).long()
        self.pro_y = torch.tensor(self.raw_y).long() # (N)
        self.group_ids = [[u] * len(data[self.users[u]]['y']) for u in range(self.n_groups)]
        self.group_ids = np.array(list(chain(*self.group_ids)), dtype=np.int32) # (N), corresponding group ids

        self.group_counts, bin_edges = np.histogram(self.group_ids, bins=range(self.n_groups+1), density=False)
        self.group_counts = np.array(self.group_counts, dtype=np.float64)
        self.group_dist, bin_edges = np.histogram(self.group_ids, bins=range(self.n_groups+1), density=True)
        
        if np.sum(self.group_dist) - 1. > 1e-4:
            raise ValueError

        #######################
        #### Dataset stats ####
        #######################
        self.group_stats = np.zeros((self.n_groups, 3))
        self.group_stats[:, 0] = self.group_counts
        self.group_stats[:, 1] = self.group_dist
        for group_id in range(self.n_groups):
            indices = np.nonzero(np.asarray(self.group_ids == group_id))[0]
            self.group_stats[group_id, 2] = np.mean(self.pro_y[indices].cpu().numpy())

        self.df_stats = pd.DataFrame(self.group_stats, columns=['n', 'frac', 'class_balance'])
        self.df_stats['group_id'] = self.df_stats.index
        self.df_stats['binary'] = self.df_stats['group_id'].apply(lambda x: '{0:b}'.format(x).zfill(int(np.log(self.n_groups) + 1)))

        print("#Users in %s-set: %d" % (split, len(self.users)))
        print("#Records in %s-set: %d" % (split, self.pro_x.size(0)))

    def __len__(self):
        return len(self.group_ids)

    def __getitem__(self, index):
        return self.pro_x[index], self.pro_y[index], \
            self.group_ids[index], self.data_mask[index], \
                self.raw_x[index], self.raw_y[index]

