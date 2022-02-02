import torch
import numpy as np
import torch.nn as nn
from transformers import AlbertPreTrainedModel
from transformers.modeling_albert import AlbertTransformer

from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import SAGEConv


class ALBERTClassifier(AlbertPreTrainedModel):

    def __init__(self, config, embedding_dim, 
                pred_hidden_dim, pred_output_dim, KG=None):
        super(ALBERTClassifier, self).__init__(config)

        self.KG = KG
        self.config = config
        self.config.embedding_size = embedding_dim
        self.config.hidden_size = pred_hidden_dim
        self.encoder = AlbertTransformer(self.config) # AlbertModel(self.config).from_pretrained('albert-base-v2').encoder 
        self.pooler = nn.Linear(pred_hidden_dim, pred_output_dim)
        self.pooler_activation = nn.Tanh()

        if self.KG is not None:
            self.conv1 = SAGEConv(in_channels=128, out_channels=64)
            self.relu = nn.ReLU()
            self.conv2 = SAGEConv(in_channels=64, out_channels=64)
            self.pooler = nn.Linear(pred_hidden_dim + 64, pred_output_dim)
            
    def forward(self, config, embedding_output, raw_x, attention_mask):

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        output_attentions = self.config.output_attentions
        output_hidden_states = (self.config.output_hidden_states)
        return_dict =  self.config.use_return_dict
        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)
        
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        sequence_output = encoder_outputs[0]
        out = sequence_output[:, 0]
        if self.KG is not None:
            batched_sentence = raw_x
            x, edge_index, edge_attr, batch, num_nodes, num_edges, entities_id, edge_type = \
                self.KG.reduce_connected(batched_sentence, device=embedding_output.device, raw=True)
            x = self.relu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
            graph_emb = global_mean_pool(
                x.to(embedding_output.device), 
                batch.to(embedding_output.device), 
                size=num_nodes.size(0)
                )
            out = torch.cat([out, graph_emb], dim=-1)

        pooled_output = self.pooler_activation(self.pooler(out))

        return pooled_output
