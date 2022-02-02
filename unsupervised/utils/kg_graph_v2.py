# This is the Accelerated Version of kg_graph.py
# If exits one pair of entities in the KG is not connected,
# then simply output the trivial graph (without edges)

import re
import os
import networkx as nx
import os.path as osp
import numpy as np
import torch
from torch_geometric.utils.num_nodes import maybe_num_nodes


class Sentence2Graph(object):
    
    LARGE_NUM = 1e10
    
    def __init__(self, kg_folder, num_nodes=None):
        """
        all_x          -> [torch.FloatTensor]  shape: (N, node_emb_dim)
        all_edge_index -> [torch.LongTensor]   shape: (2, E)
        all_edge_type  -> [torch.LongTensor]   shape: (E)
        all_edge_attr  -> [torch.FloatTensor]  shape: (n_edge_type, edge_emb_dim)
        vocab          -> [Dict] {word: word_id}
        """
        self.vocab = torch.load(osp.join(kg_folder, 'vocab.pt'))
        self.all_x = torch.load(osp.join(kg_folder, 'wn18rr_x.pt'))
        all_edge_index = torch.load(osp.join(kg_folder, 'wn18rr_edge_index.pt'))
        all_edge_type = torch.load(osp.join(kg_folder, 'wn18rr_edge_type.pt'))
        all_edge_attr = torch.load(osp.join(kg_folder, 'wn18rr_edge_attr.pt'))
        knn_edge_index = torch.load(osp.join(kg_folder, 'wn18rr_knn_edge_index.pt'))

        # simply cat all_edge_index and knn_edge_index, and use the mean edge_attr as knn_edge_attr 
        self.all_edge_index = torch.cat([all_edge_index, knn_edge_index], dim=1)
        knn_edge_type = max(all_edge_type) + 1
        self.all_edge_type = torch.cat([all_edge_type, torch.ones(knn_edge_index.size(1)).long() * knn_edge_type], dim=0)
        self.all_edge_attr = torch.cat([all_edge_attr, all_edge_attr.mean(dim=0).unsqueeze(dim=0)], dim=0)

        self.num_nodes = maybe_num_nodes(all_edge_index, num_nodes)
        self.G = nx.Graph() # undirected
        self.G.add_nodes_from(range(self.num_nodes))
        self.G.add_edges_from(list(all_edge_index.cpu().numpy().T))
    
    @staticmethod
    def split_line(line):
        '''split given line/phrase into list of words
        Input:   line   -> [str]
                string representing phrase to be split
        Output: strings -> [list]
                list of strings, with each string representing a word
        '''
        return  re.findall(r"[\w']+|[.,!?;]", line)

    @staticmethod
    def tokens_to_ids(vocab, batched_data):
        """
        Input:  vocab -> [kge.dataset.Dataset]
                Dataset for training embeddings(WordNet)
                batched_data --> [list, np.array]
                Batched and tokenized data with shape (Batch_size, Sequence_Length)
        Output: entities_id_list -> [list]
                shape (Batch_size, Sequence_Length)
        """
        def tokens_to_word_ids(tokens, vocab):
            return list(set([vocab[word] for word in tokens if word in vocab.keys()]))

        entities_id_list = [tokens_to_word_ids(seq, vocab) for seq in batched_data]
        return entities_id_list

    @staticmethod
    def subgraph(subset, edge_index, edge_type=None, edge_attr=None, num_nodes=None):
        """
        Returns the induced subgraph of :obj:`(edge_index, edge_attr)`
        containing the nodes in :obj:`subset`.
        Input:  subset         -> [torch.LongTensor]  
                edge_index     -> [torch.LongTensor]   shape: (2, E)
                edge_type      -> [torch.LongTensor]   shape: (E)
                edge_attr      -> [torch.FloatTensor]  shape: (n_edge_type, edge_emb_dim)
        Output: 
                masked_edge_index    -> [torch.LongTensor]
                masked_edge_attr     -> [torch.FloatTensor]
        """
        device = edge_index.device

        if isinstance(subset, list) or isinstance(subset, tuple):
            subset = torch.tensor(subset, dtype=torch.long)

        if subset.dtype == torch.bool or subset.dtype == torch.uint8:
            n_mask = subset
        else:
            num_nodes = maybe_num_nodes(edge_index, num_nodes)
            n_mask = torch.zeros(num_nodes, dtype=torch.bool)
            n_mask[subset] = 1

        mask = n_mask[edge_index[0]] & n_mask[edge_index[1]]
        masked_edge_index = edge_index[:, mask]

        if edge_type is not None:
            assert edge_attr is not None
            masked_edge_type = edge_type[mask]
            masked_edge_attr = edge_attr[masked_edge_type]
        else:
            masked_edge_attr = None

        return masked_edge_index, masked_edge_attr, masked_edge_type
    
    def __relabel__(self, sub_nodes, edge_index):

        sub_nodes = torch.tensor(sub_nodes).long()
        row, col = edge_index
        # remapping the nodes in the explanatory subgraph to new ids.
        node_idx = row.new_full((self.num_nodes,), -1)
        node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
        relabeled_edge_index = node_idx[edge_index]
        return relabeled_edge_index
    
    def reduce(self, batched_sentence, device, raw=False):
        """
        Input:  batched_sentence -> [list]
                e.g. [["good", "day", "i", "am", "your", "friend"],
                      ["hate", "terrible", "weather"]]
                      then set raw = False
                or  [["good day, i am your friend"],
                      ["hate terrible weather"]
                    set raw = False
        Output:  x, edge_index, edge_attr: information for subgraphs
                batch             ->   [torch.LongTensor]  shape: (N)
                num_nodes         ->   [torch.LongTensor]  shape: (N_graphs)
                num_edges         ->   [torch.LongTensor]  shape: (N_graphs)
                flat_entities_id  ->   [list]              
                edge_type         ->   [torch.LongTensor]  shape: (E)
        """
        if raw:
            batched_sentence = [self.split_line(sentence.lower()) for sentence in batched_sentence]
        entities_id = self.tokens_to_ids(self.vocab, batched_sentence)

        num_nodes = torch.tensor([len(entities) for entities in entities_id]).long()
        batch = torch.tensor([i for i in range(len(num_nodes)) for _ in range(num_nodes[i])]).long()
        cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]]).long()
        
        def extract_subgraph(entities):
            node_idx = torch.unique(torch.tensor(entities).long())
            masked_edge_index, masked_edge_attr, masked_edge_type = self.subgraph(node_idx, self.all_edge_index, self.all_edge_type, self.all_edge_attr)
            return masked_edge_index, masked_edge_attr, masked_edge_type
        
        num_edges = []
        x = torch.tensor([])
        edge_index = torch.tensor([[], []])
        edge_attr = torch.tensor([])
        edge_type = torch.tensor([])
        
        for i, entities in enumerate(entities_id):
            x = torch.cat([x, self.all_x[entities]], dim=0)
            masked_edge_index, masked_edge_attr, masked_edge_type = extract_subgraph(entities)
            masked_edge_index = self.__relabel__(entities, masked_edge_index) + cum_nodes[i]
            
            edge_index = torch.cat([edge_index, masked_edge_index], dim=1)
            edge_type = torch.cat([edge_type, masked_edge_type], dim=0)
            edge_attr = torch.cat([edge_attr, masked_edge_attr], dim=0)
            num_edges.append(masked_edge_index.size(1))
            
        num_edges = torch.tensor(num_edges).long()
        x = x.to(device)
        edge_index = edge_index.long().to(device)
        edge_type = edge_type.long().to(device)
        edge_attr = edge_attr.to(device)
        num_nodes = num_nodes.to(device)
        num_edges = num_edges.to(device)
        
        flat_entities_id = []
        for entities in entities_id:
            flat_entities_id.extend(entities)
        flat_entities_id = torch.tensor(flat_entities_id).long().to(device)
        
        assert x.size(0) == flat_entities_id.size(0)
        assert x.size(0) == batch.size(0)
        
        assert edge_index.size(1) == edge_attr.size(0)
        assert edge_index.size(1) == edge_type.size(0)
        assert len(num_nodes) == len(num_edges)
        
        return x, edge_index, edge_attr, batch, num_nodes, num_edges, flat_entities_id, edge_type
    
    def reduce_connected(self, batched_sentence, device, raw=False):
        """
        Input:  batched_sentence -> [list]
                e.g. [["good", "day", "i", "am", "your", "friend"],
                      ["hate", "terrible", "weather"]]
                      then set raw=True
                or  [["good day, i am your friend"],
                      ["hate terrible weather"]
                    set raw = False
        Output:  x, edge_index, edge_attr: information for subgraphs
                batch             ->   [torch.LongTensor]  shape: (N)
                num_nodes         ->   [torch.LongTensor]  shape: (N_graphs)
                num_edges         ->   [torch.LongTensor]  shape: (N_graphs)
                flat_entities_id  ->   [list]              
                edge_type         ->   [torch.LongTensor]  shape: (E)
        """
        # 1. prerpocess the input sentences into lists of entities
        if raw:
            batched_sentence = [self.split_line(sentence.lower()) for sentence in batched_sentence]
        entities_id = self.tokens_to_ids(self.vocab, batched_sentence)
        
        num_edges = []
        num_nodes = []
        cum_nodes = [0]
        flat_entities_id = []
        x = torch.tensor([]).to(device)
        edge_index = torch.tensor([[], []]).long().to(device)
        edge_attr = torch.tensor([]).to(device)
        edge_type = torch.tensor([]).long().to(device)
        for subset in entities_id:
            # 2. for each node subset, we calculate the pair-wise distance
            # ... and get the minimum spanning tree whose nodes contains subset 
            n =  len(subset)
            flag = 1
            if n < 2:
                num_nodes.append(n)
                num_edges.append(0)
                cum_nodes.append(cum_nodes[-1] + num_nodes[-1]) 
                x = torch.cat([x, self.all_x[subset].to(device)], dim=0)
                flat_entities_id.extend(subset)
                continue
                
            g = nx.DiGraph()
            adj = torch.ones([n, n]) * self.LARGE_NUM
            paths = {i:{} for i in range(n)}
            g.add_nodes_from(subset)
            for i in range(n):
                for j in range(i + 1, n):
                    try:
                        path_ij = nx.shortest_path(self.G, source=subset[i], target=subset[j])
                        g.add_weighted_edges_from([(subset[i], subset[j], len(path_ij))])
                        g.add_weighted_edges_from([(subset[j], subset[i], len(path_ij))])
                        paths[i][j] = path_ij
                    except:
                        flag = 0
                        break
                if flag ==0 :
                    break
            if flag ==0 :
                num_nodes.append(n)
                num_edges.append(0)
                cum_nodes.append(cum_nodes[-1] + num_nodes[-1]) 
                x = torch.cat([x, self.all_x[subset].to(device)], dim=0)
                flat_entities_id.extend(subset)
                continue
            sg = nx.minimum_spanning_arborescence(g)
            sg_edges = list(sg.edges)
            
            # 3. collect the paths(each path representes an ego edge in the spanning tree) 
            # ... into one graph
            sub_nodes = []
            return_edge_index = []
            return_edge_type = []
            row, col = self.all_edge_index
            for e in sg_edges:
                _i = subset.index(e[0]); _j = subset.index(e[1])
                i = min(_i, _j); j = max(_i, _j)
                p = paths[i][j]
                sub_nodes.extend(p)
                for k in range(len(p)-1):
                    return_edge_index.append([p[k], p[k+1]])
                    mask = (row == p[k]) * (col == p[k+1])
                    if mask.sum() > 0:
                        return_edge_type.append(self.all_edge_type[torch.nonzero(mask).view(-1)[0]])
                    else:
                        mask = (row == p[k+1]) * (col == p[k])
                        return_edge_type.append(self.all_edge_type[torch.nonzero(mask).view(-1)[0]])
                    
            # 4. aggregate the subgraph into batch
            sub_nodes = torch.tensor(sub_nodes).long().unique().to(device)
            num_nodes.append(sub_nodes.size(0))
            return_x = self.all_x[sub_nodes].to(device)
            
            return_edge_index = torch.tensor(return_edge_index).long().T.to(device)
            return_edge_index = self.__relabel__(sub_nodes, return_edge_index) + cum_nodes[-1]
            
            return_edge_type = torch.tensor(return_edge_type).to(device)
            return_edge_attr = self.all_edge_attr[return_edge_type].to(device)
            
            cum_nodes.append(cum_nodes[-1] + num_nodes[-1])
            num_edges.append(return_edge_index.size(1))
            
            x = torch.cat([x, return_x], dim=0)
            edge_index = torch.cat([edge_index, return_edge_index], dim=1)
            edge_attr = torch.cat([edge_attr, return_edge_attr], dim=0)
            edge_type = torch.cat([edge_type, return_edge_type], dim=0)
            flat_entities_id.extend(sub_nodes.tolist())
            
        num_nodes = torch.tensor(num_nodes).long()
        num_edges = torch.tensor(num_edges).long()
        num_nodes = num_nodes.to(device)
        num_edges = num_edges.to(device)
        flat_entities_id = torch.tensor(flat_entities_id).long().to(device)
        batch = torch.tensor([i for i in range(len(num_nodes)) for _ in range(num_nodes[i])]).long().to(device)
        
        
        assert x.size(0) == flat_entities_id.size(0)
        assert x.size(0) == batch.size(0)
        
        assert edge_index.size(1) == edge_attr.size(0)
        assert edge_index.size(1) == edge_type.size(0)
        assert len(num_nodes) == len(num_edges)
        
        return x, edge_index, edge_attr, batch, num_nodes, num_edges, flat_entities_id, edge_type