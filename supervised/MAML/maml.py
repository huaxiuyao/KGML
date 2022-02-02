import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import random
from learner import Seq_Classification
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict
from torch.distributions import Beta
from sklearn.metrics import f1_score, roc_auc_score


class MAML(nn.Module):
    def __init__(self, args, KG):
        super(MAML, self).__init__()
        self.args = args
        self.learner = Seq_Classification(args=args, KG=KG)
        self.loss_fn = nn.CrossEntropyLoss()
        self.dist = Beta(torch.FloatTensor([2]), torch.FloatTensor([2]))
        if self.args.train:
            self.num_updates = self.args.num_updates
        else:
            self.num_updates = self.args.num_updates_test

    def forward(self, xs, ys, xq, yq):

        xs_input_ids, xs_attn_masks, xs_token_type_ids, xs_graph = xs
        xq_input_ids, xq_attn_masks, xq_token_type_ids, xq_graph = xq

        xs_input_ids, xs_attn_masks, xs_token_type_ids = xs_input_ids.to("cuda"), xs_attn_masks.to(
            "cuda"), xs_token_type_ids.to("cuda")
        xq_input_ids, xq_attn_masks, xq_token_type_ids = xq_input_ids.to("cuda"), xq_attn_masks.to(
            "cuda"), xq_token_type_ids.to("cuda")

        ys = ys.to("cuda")
        yq = yq.to("cuda")

        create_graph = True

        fast_weights = OrderedDict(self.learner.net_adaptive.named_parameters())

        for inner_batch in range(self.num_updates):
            logits = self.learner.functional_forward(xs_input_ids, xs_attn_masks, xs_token_type_ids, xs_graph,
                                                     fast_weights, inner=True)
            loss = self.loss_fn(logits, ys)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph, only_inputs=True)

            fast_weights = OrderedDict(
                (name, param - self.args.update_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )

        if self.args.train:
            query_logits = self.learner.functional_forward(xq_input_ids, xq_attn_masks, xq_token_type_ids, xq_graph,
                                                           fast_weights)
        else:
            with torch.no_grad():
                query_logits = self.learner.functional_forward(xq_input_ids, xq_attn_masks, xq_token_type_ids, xq_graph,
                                                               fast_weights)
        query_loss = self.loss_fn(query_logits, yq)

        y_pred = query_logits.softmax(dim=1).max(dim=1)[1]

        query_acc = (y_pred == yq).sum().float() / yq.shape[0]
        return query_loss, query_acc