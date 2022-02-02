import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from utils import euclidean_dist
import ipdb
import numpy as np
from torch.distributions import Beta
from learner import Seq_Classification

from torchvision.utils import make_grid
from matplotlib import pyplot as plt

class Protonet(nn.Module):
    def __init__(self, args, KG):
        super(Protonet, self).__init__()
        self.args = args
        self.learner = Seq_Classification(args=args, KG=KG)

    def forward(self, xs, ys, xq, yq):
        xs_input_ids, xs_attn_masks, xs_token_type_ids, xs_graph = xs
        xq_input_ids, xq_attn_masks, xq_token_type_ids, xq_graph = xq

        xs_input_ids, xs_attn_masks, xs_token_type_ids = xs_input_ids.to("cuda"), xs_attn_masks.to(
            "cuda"), xs_token_type_ids.to("cuda")
        xq_input_ids, xq_attn_masks, xq_token_type_ids = xq_input_ids.to("cuda"), xq_attn_masks.to(
            "cuda"), xq_token_type_ids.to("cuda")

        ys = ys.to("cuda")
        yq = yq.to("cuda")

        x_input_idx = torch.cat([xs_input_ids, xq_input_ids], 0)
        x_attn_masks = torch.cat([xs_attn_masks, xq_attn_masks], 0)
        x_token_type_ids = torch.cat([xs_token_type_ids, xq_token_type_ids], 0)
        x_graph=xs_graph+xq_graph

        z = self.learner(x_input_idx, x_attn_masks, x_token_type_ids, x_graph)

        z_dim = z.size(-1)

        z_proto = z[:self.args.num_classes * self.args.update_batch_size].view(self.args.num_classes,
                                                                               self.args.update_batch_size, z_dim).mean(1)
        zq = z[self.args.num_classes * self.args.update_batch_size:]

        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1)

        loss_val = []
        for i in range(self.args.num_classes * self.args.update_batch_size_eval):
            loss_val.append(-log_p_y[i, yq[i]])

        loss_val = torch.stack(loss_val).squeeze().mean()

        _, y_hat = log_p_y.max(1)

        acc_val = torch.eq(y_hat, yq).float().mean()

        return loss_val, acc_val