import torch
import numpy as np
import torch.nn as nn
from transformers import AlbertPreTrainedModel
from transformers.modeling_albert import AlbertTransformer

class ALBERTContextModel(AlbertPreTrainedModel):

    def __init__(self, config, 
                embedding_dim, context_hidden_dim,
                ):
        super(ALBERTContextModel, self).__init__(config)

        self.config = config
        self.config.embedding_size = embedding_dim
        self.config.hidden_size = context_hidden_dim
        self.encoder = AlbertTransformer(self.config)

    def forward(self, config, embedding_output, attention_mask):
        
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
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        return sequence_output