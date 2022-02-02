import pickle

import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AutoTokenizer

class AmazonReview(Dataset):
    def __init__(self, args, mode, KG, max_len=128, bert_model='albert-base-v1'):
        super(AmazonReview, self).__init__()
        self.args = args
        self.nb_classes = args.num_classes
        self.nb_samples_per_class = args.update_batch_size + args.update_batch_size_eval
        self.n_way = args.num_classes  # n-way
        self.KG = KG
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.k_shot = args.update_batch_size  # k-shot
        self.k_query = args.update_batch_size_eval  # for evaluation
        self.set_size = self.n_way * self.k_shot  # num of samples per set
        self.query_size = self.n_way * self.k_query  # number of samples per set for evaluation
        self.max_len = max_len
        self.mode = mode

        self.data = pickle.load(open('xxx/amazon.pkl', 'rb'))

        self.key_map = dict({eachkey:idx for idx, eachkey in enumerate(self.data.keys())})

        if self.mode == 'train':
            self.classes_idx = np.array([2, 3, 4, 7, 11, 12, 13, 18, 19, 20])
        elif self.mode =='val':
            self.classes_idx = np.array([1, 22, 23, 6, 9])
        elif self.mode == 'test':
            self.classes_idx = np.array([0, 5, 14, 15, 8, 10, 16, 17, 21])

    def to_graphs(self, seq):
        graph_lst = []
        for raw_x in seq:
            x, edge_index, edge_attr, batch, num_nodes, num_edges, entities_id, edge_type = \
                self.KG.reduce_connected([raw_x], raw=False)
            graph_lst.append((x, edge_index, edge_attr, num_nodes, num_edges, entities_id, edge_type))
        return graph_lst

    def get_part_data(self, sel_category, sel_sample):
        seq = list(self.data[sel_category][sel_sample])

        graph_lst = []

        if self.args.use_kg:
            graph_lst = self.to_graphs(seq)

        encoded_seq = self.tokenizer(seq,
                                     padding='max_length',
                                     truncation=True,
                                     max_length=self.max_len,
                                     return_tensors='pt')

        token_ids = encoded_seq['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_seq['attention_mask'].squeeze(
            0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_seq['token_type_ids'].squeeze(0)

        return token_ids, attn_masks, token_type_ids, graph_lst

    def __getitem__(self, index):
        inputa=[]
        inputb = []

        support_y = np.zeros([self.args.meta_batch_size, self.set_size])
        query_y = np.zeros([self.args.meta_batch_size, self.query_size])

        for meta_batch_id in range(self.args.meta_batch_size):
            self.choose_classes = np.random.choice(self.classes_idx, size=self.nb_classes, replace=False)
            token_ids_a, token_ids_b = [], []
            attn_masks_a, attn_masks_b = [], []
            token_type_ids_a, token_type_ids_b = [], []
            graph_lst_a, graph_lst_b = [], []

            for j in range(self.nb_classes):
                self.samples_idx = np.arange(len(self.data[self.choose_classes[j]]))
                np.random.shuffle(self.samples_idx)
                choose_samples = self.samples_idx[:self.nb_samples_per_class]
                process_data = self.get_part_data(self.choose_classes[j], choose_samples)
                token_ids_a.append(process_data[0][:self.k_shot])
                token_ids_b.append(process_data[0][self.k_shot:])

                attn_masks_a.append(process_data[1][:self.k_shot])
                attn_masks_b.append(process_data[1][self.k_shot:])

                token_type_ids_a.append(process_data[2][:self.k_shot])
                token_type_ids_b.append(process_data[2][self.k_shot:])

                graph_lst_a.extend(process_data[3][:self.k_shot])
                graph_lst_b.extend(process_data[3][self.k_shot:])

                support_y[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = j
                query_y[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = j

            token_ids_a = torch.cat(token_ids_a, dim=0)
            token_ids_b = torch.cat(token_ids_b, dim=0)

            attn_masks_a = torch.cat(attn_masks_a, dim=0)
            attn_masks_b = torch.cat(attn_masks_b, dim=0)

            token_type_ids_a = torch.cat(token_type_ids_a, dim=0)
            token_type_ids_b = torch.cat(token_type_ids_b, dim=0)

            inputa.append([token_ids_a, attn_masks_a, token_type_ids_a, graph_lst_a])
            inputb.append([token_ids_b, attn_masks_b, token_type_ids_b, graph_lst_b])

        return inputa, torch.LongTensor(support_y), inputb, torch.LongTensor(
            query_y)


class Huffpost(Dataset):
    def __init__(self, args, mode, KG, max_len=128, bert_model='albert-base-v1'):
        super(Huffpost, self).__init__()
        self.args = args
        self.nb_classes = args.num_classes
        self.nb_samples_per_class = args.update_batch_size + args.update_batch_size_eval
        self.n_way = args.num_classes  # n-way
        self.KG = KG
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.k_shot = args.update_batch_size  # k-shot
        self.k_query = args.update_batch_size_eval  # for evaluation
        self.set_size = self.n_way * self.k_shot  # num of samples per set
        self.query_size = self.n_way * self.k_query  # number of samples per set for evaluation
        self.max_len = max_len
        self.mode = mode

        if self.mode == 'train':
            self.classes_idx = np.array([30, 25, 36, 4, 14, 22, 28, 7, 20, 8, 9, 3, 15, 34, 24, 29, 1,
                                         6, 0, 16, 37, 5, 33, 35, 27])
        elif self.mode =='val':
            self.classes_idx = np.array([17, 13, 18,  2, 40, 39])
        elif self.mode == 'test':
            self.classes_idx = np.array([32, 11, 23, 19, 10, 26, 12, 31, 21, 38])

        self.data = pickle.load(open('xxx/huffpost.pkl', 'rb'))

    def to_graphs(self, seq):
        graph_lst = []
        for raw_x in seq:
            x, edge_index, edge_attr, batch, num_nodes, num_edges, entities_id, edge_type = \
                self.KG.reduce_connected([raw_x], raw=False)
            graph_lst.append((x, edge_index, edge_attr, num_nodes, num_edges, entities_id, edge_type))
        return graph_lst

    def get_part_data(self, sel_category, sel_sample):
        seq = list(self.data[sel_category][sel_sample])

        graph_lst = []

        if self.args.use_kg:
            graph_lst = self.to_graphs(seq)

        encoded_seq = self.tokenizer(seq,
                                     padding='max_length',
                                     truncation=True,
                                     max_length=self.max_len,
                                     return_tensors='pt')

        token_ids = encoded_seq['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_seq['attention_mask'].squeeze(
            0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_seq['token_type_ids'].squeeze(0)

        return token_ids, attn_masks, token_type_ids, graph_lst

    def __getitem__(self, index):
        inputa=[]
        inputb = []

        support_y = np.zeros([self.args.meta_batch_size, self.set_size])
        query_y = np.zeros([self.args.meta_batch_size, self.query_size])

        for meta_batch_id in range(self.args.meta_batch_size):
            self.choose_classes = np.random.choice(self.classes_idx, size=self.nb_classes, replace=False)
            token_ids_a, token_ids_b = [], []
            attn_masks_a, attn_masks_b = [], []
            token_type_ids_a, token_type_ids_b = [], []
            graph_lst_a, graph_lst_b = [], []

            for j in range(self.nb_classes):
                self.samples_idx = np.arange(len(self.data[self.choose_classes[j]]))
                np.random.shuffle(self.samples_idx)
                choose_samples = self.samples_idx[:self.nb_samples_per_class]
                process_data = self.get_part_data(self.choose_classes[j], choose_samples)
                token_ids_a.append(process_data[0][:self.k_shot])
                token_ids_b.append(process_data[0][self.k_shot:])

                attn_masks_a.append(process_data[1][:self.k_shot])
                attn_masks_b.append(process_data[1][self.k_shot:])

                token_type_ids_a.append(process_data[2][:self.k_shot])
                token_type_ids_b.append(process_data[2][self.k_shot:])

                graph_lst_a.extend(process_data[3][:self.k_shot])
                graph_lst_b.extend(process_data[3][self.k_shot:])

                support_y[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = j
                query_y[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = j

            token_ids_a = torch.cat(token_ids_a, dim=0)
            token_ids_b = torch.cat(token_ids_b, dim=0)

            attn_masks_a = torch.cat(attn_masks_a, dim=0)
            attn_masks_b = torch.cat(attn_masks_b, dim=0)

            token_type_ids_a = torch.cat(token_type_ids_a, dim=0)
            token_type_ids_b = torch.cat(token_type_ids_b, dim=0)

            inputa.append([token_ids_a, attn_masks_a, token_type_ids_a, graph_lst_a])
            inputb.append([token_ids_b, attn_masks_b, token_type_ids_b, graph_lst_b])

        return inputa, torch.LongTensor(support_y), inputb, torch.LongTensor(
            query_y)
