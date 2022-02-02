import torch
import torch.nn as nn
from transformers import AutoConfig, AlbertModel

class ContextualNet(nn.Module):

    def __init__(self, ContextNet: nn.Module, PredictionNet: nn.Module,
                 embedding_dim, pred_hidden_dim, pred_output_dim,
                 context_hidden_dim=0, support_size=50, use_context=None, KG=None):
        super(ContextualNet, self).__init__()
        
        self.config = AutoConfig.from_pretrained('albert-base-v2')
        
        self.config.embedding_size = embedding_dim
        self.embeddings = AlbertModel(self.config).from_pretrained('albert-base-v2').embeddings # AlbertEmbeddings(self.config)
        self.support_size = support_size
        self.use_context = use_context

        if use_context:
            self.context_net = ContextNet(
                config=self.config, 
                embedding_dim=embedding_dim, 
                context_hidden_dim=context_hidden_dim
                )
            new_embed_dim = embedding_dim + context_hidden_dim
        else:
            new_embed_dim = embedding_dim

        self.prediction_net = PredictionNet(
            config=self.config,
            embedding_dim=new_embed_dim, 
            pred_hidden_dim=pred_hidden_dim,
            pred_output_dim=pred_output_dim, 
            KG=KG) # use pretrained embed by default

    def forward(self, sentence, attention_mask, raw_x):

        embeds = self.embeddings(sentence)
        if self.use_context:
            batch_size, W, E = embeds.size()
            if batch_size % self.support_size == 0:
                meta_batch_size, support_size = batch_size // self.support_size, self.support_size
            else:
                meta_batch_size, support_size = 1, batch_size

            context = self.context_net(self.config, embeds, attention_mask) # Shape: batch_size, W, _E
            _, W, _E = context.size()
            context = context.reshape((meta_batch_size, support_size, W, _E))
            context = context.mean(dim=1) # (meta_batch_size, W, _E)

            context = torch.repeat_interleave(context, repeats=support_size, dim=0)# (batch_size, W, _E)
            embeds = torch.cat([embeds, context], dim=-1)# (batch_size, W, E+_E)

        out = self.prediction_net(
            self.config, embeds, 
            attention_mask=attention_mask, raw_x=raw_x
            )
        return out