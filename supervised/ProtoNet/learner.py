import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup

from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import SAGEConv


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Seq_Classification(nn.Module):
    def __init__(self, args, KG=None):
        super(Seq_Classification, self).__init__()
        self.args = args
        self.KG=KG
        self.hid_dim = 768
        self.relu = nn.ReLU()
        self.net_adaptive = nn.Sequential(nn.Linear(self.hid_dim, 64))
        self.bert_layer = AutoModel.from_pretrained("albert-base-v1")

        if self.KG is not None:
            self.conv1 = SAGEConv(in_channels=128, out_channels=64)
            self.relu = nn.ReLU()
            self.conv2 = SAGEConv(in_channels=64, out_channels=16)
            self.map_layer = nn.Linear(16, 16)

    def forward(self, x_input_ids, x_attn_masks, x_token_type_ids, x_graph=None):
        last_state = self.bert_layer(x_input_ids, x_attn_masks, x_token_type_ids)['last_hidden_state'][:, 0]
        last_state = self.net_adaptive[0](last_state)

        bs = last_state.shape[0]

        if self.KG is not None:
            num_nodes = torch.tensor([]).cuda()
            x = torch.tensor([]).cuda()
            batch = torch.tensor([]).cuda()
            edge_index = torch.tensor([[], []]).cuda()

            for i, g in enumerate(x_graph):
                x_i, edge_index_i, _, num_nodes_i, _, _, _ = g
                num_nodes = torch.cat([num_nodes, num_nodes_i], axis=0)
                x = torch.cat([x, x_i], axis=0)
                edge_index = torch.cat([edge_index, edge_index_i], axis=1)
            batch = torch.tensor([i for i in range(len(num_nodes)) for _ in range(int(num_nodes[i].item()))]).long().cuda()
            edge_index = edge_index.long()
            x = self.relu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
            graph_emb = global_mean_pool(x, batch, size=num_nodes.size(0))
            graph_emb = self.map_layer(graph_emb)

            final_feat = torch.cat([last_state, graph_emb], dim=-1)
        else:
            final_feat = last_state
        return final_feat

    def functional_forward(self, x_input_ids, x_attn_masks, x_token_type_ids, x_graph, weights):
        last_state = self.bert_layer(x_input_ids, x_attn_masks, x_token_type_ids)['last_hidden_state'][:, 0]
        last_state = self.relu(F.linear(last_state, weights['0.weight'], weights['0.bias']))

        bs = last_state.shape[0]

        if self.KG is not None:
            num_nodes = torch.tensor([]).cuda()
            x = torch.tensor([]).cuda()
            batch = torch.tensor([]).cuda()
            edge_index = torch.tensor([[], []]).cuda()

            for i, g in enumerate(x_graph):
                x_i, edge_index_i, _, num_nodes_i, _, _, _ = g
                num_nodes = torch.cat([num_nodes, num_nodes_i], axis=0)
                x = torch.cat([x, x_i], axis=0)
                edge_index = torch.cat([edge_index, edge_index_i], axis=1)
            batch = torch.tensor(
                [i for i in range(len(num_nodes)) for _ in range(int(num_nodes[i].item()))]).long().cuda()
            edge_index = edge_index.long()
            x = self.relu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
            graph_emb = global_mean_pool(x, batch, size=num_nodes.size(0))
            graph_emb = self.relu(self.map_layer(graph_emb))

            final_feat = torch.cat([last_state, graph_emb], dim=-1)

        else:
            final_feat = last_state

        return final_feat