import os
from datetime import datetime
from pathlib import Path
import argparse
import utils
import datasets
import numpy as np
import torch
import models
import torch.nn as nn
from tqdm import trange, tqdm
from dro_loss import LossComputer
from utils import *
from utils.args import parse_args
from utils import set_seed
from transformers import AlbertTokenizer, AutoTokenizer
from transformers import AutoConfig
# Set parameters
from default_param import default_dict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_one_hot(values, num_classes):
    return np.eye(num_classes)[values]

def test(args, eval_on):
    Dict = default_dict[args.dataset]
    args.data_dir = Dict['data_dir']
    # Cuda
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Dict = default_dict[args.dataset]
    # Make reproducible
    if args.seed is not None:
        print('setting seed', args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Get data
    tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
    train_loader, train_eval_loader, val_loader, test_loader = datasets.get_loaders(args, tokenizer)
    args.n_groups = train_loader.dataset.n_groups
    # Get KG
    if args.use_kg:
        from utils.kg_graph_v2 import Sentence2Graph
        kg_folder = 'data/wn18rr_kg'
        KG =  Sentence2Graph(kg_folder=kg_folder)
    else: 
        KG = None

        # Get model
    args.n_groups = train_loader.dataset.n_groups
    seq_len = train_loader.dataset[0][0].size(0)
    # Load lots of parameters and get model
    model = models.ContextualNet(
    ContextNet=models.ALBERTContextModel, PredictionNet=Dict['PredictionNet'],  # context model and predictor
    embedding_dim=Dict['embedding_dim'], pred_hidden_dim=Dict['pred_hidden_dim'], pred_output_dim=Dict['pred_output_dim'],  # dims for predictor
    context_hidden_dim=Dict['context_hidden_dim'], # dims for context model
    use_context=args.use_context,# specify whether to use context net, and the type for the context net(if YES)
    support_size=args.support_size, KG=KG # meta-learning
    ).to(device)

    state_dict = torch.load(args.ckpt_path)[0]
    model.load_state_dict(state_dict)

    model = model.to(args.device)
    model.eval()

    if eval_on == 'train':
        worst_case_acc, stats = utils.evaluate_groups(args, model, train_eval_loader, split='train')
    elif eval_on == 'val':
        worst_case_acc, stats = utils.evaluate_groups(args, model, val_loader, split='val')
    elif eval_on == 'test':
        worst_case_acc, stats = utils.evaluate_groups(args, model, test_loader, split='test')

    return worst_case_acc, stats


#################
### Arguments ###
#################

parser = argparse.ArgumentParser()

# Model args
parser.add_argument('--model', type=str, default='ContextualConvNet')

parser.add_argument('--drop_last', type=int, default=0)
parser.add_argument('--ckpt_path', type=str, default=None)
parser.add_argument('--use_kg', type=bool, default=False)

parser.add_argument('--pretrained', type=int, default=1,
                                   help='Pretrained resnet')
# If model is Convnet
parser.add_argument('--prediction_net', type=str, default='convnet',
                    choices=['resnet18', 'resnet34', 'resnet50', 'convnet'])

parser.add_argument('--n_context_channels', type=int, default=3, help='Used when using a convnet/resnet')
parser.add_argument('--use_context', type=int, default=0)

# Data args
parser.add_argument('--dataset', type=str, default='twitter', choices=['twitter', 'reddit'])
parser.add_argument('--data_dir', type=str, default='data/twitter')

# CelebA Data
parser.add_argument('--target_resolution', type=int, default=224,
                    help='Resize image to this size before feeding in to model')
parser.add_argument('--target_name', type=str, default='Blond_Hair',
                    help='The y value we are trying to predict')
parser.add_argument('--confounder_names', type=str, nargs='+',
                    default=['Male'],
                    help='Binary attributes from which we construct the groups. This is called confounder names \
                    for now since we are using part of Group DRO data loading')
parser.add_argument('--eval_on', type=str, nargs='+',
                    default=['test'],
                    help='Binary attributes from which we construct the groups. This is called confounder names \
                    for now since we are using part of Group DRO data loading')

# Data sampling
parser.add_argument('--meta_batch_size', type=int, default=2, help='Number of classes')
parser.add_argument('--support_size', type=int, default=50, help='Support size. What we call batch size in the appendix.')
parser.add_argument('--shuffle_train', type=int, default=1,
                    help='Only relevant when do_clustered_sampling = 0 \
                    and --uniform_over_groups 0')

# Clustered sampling
parser.add_argument('--sampling_type', type=str, default='regular',
        choices=['meta_batch_mixtures', 'meta_batch_groups', 'uniform_over_groups', 'regular'],
                    help='Sampling type')
parser.add_argument('--eval_corners_only', type=int, default=1,
                    help='Are evaluating mixtures or corners?')

parser.add_argument('--loading_type', type=str, choices=['PIL', 'jpeg'], default='jpeg',
                    help='Whether to use PIL or jpeg4py when loading images. Jpeg is faster. See README for deatiles')

# Evalaution
parser.add_argument('--n_test_dists', type=int, default=100,
                    help='Number of test distributions to evaluate on. These are sampled uniformly.')
parser.add_argument('--n_test_per_dist', type=int, default=3000,
                    help='Number of examples to evaluate on per test distribution')

# Logging
parser.add_argument('--seed', type=int, default=0, help='Seed')
parser.add_argument('--debug', type=int, default=0)

parser.add_argument('--num_workers', type=int, default=8, help='Num workers for pytorch data loader')
parser.add_argument('--pin_memory', type=int, default=1, help='Pytorch loader pin memory. \
                    Best practice is to use this')

parser.add_argument('--crop_type', type=int, default=0)
parser.add_argument('--crop_size_factor', type=float, default=1)

parser.add_argument('--ckpt_folders', type=str, nargs='+')

parser.add_argument('--context_net', type=str, default='convnet')

parser.add_argument('--experiment_name', type=str, default='')


args = parser.parse_args()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    # Cuda
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.cuda = True
    else:
        args.device = torch.device('cpu')
        args.cuda = False

    # Check if checkpoints exist
    for ckpt_folder in args.ckpt_folders:
        ckpt_path = Path('output') / 'checkpoints' / ckpt_folder / f'best_weights.pkl'
        print(ckpt_path)
        state_dict = torch.load(ckpt_path)
        print("Found: ", ckpt_path)

    all_test_stats = [] # Store all test results in list.

    for i, ckpt_folder in enumerate(args.ckpt_folders):

        args.seed = i + 10 # Mainly to make sure seed for training and testing is not the same. Not critical.
        args.ckpt_path = Path('output') / 'checkpoints' / ckpt_folder / f'best_weights.pkl'
        args.experiment_name += f'_{args.seed}'

        train_stats, val_stats, test_stats = None, None, None

        if 'train' in args.eval_on:
            _, train_stats = test(args, eval_on='train')
        if 'val' in args.eval_on:
            _, val_stats = test(args, eval_on='val')
        if 'test' in args.eval_on:
            _, test_stats = test(args, eval_on='test')
            all_test_stats.append((i, test_stats))

        print("----SEED used when evaluating -----: ", args.seed)
        print("----CKPT FOLDER -----: ", ckpt_folder)
        print("TRAIN STATS:\n ", train_stats)
        print('-----------')

        print("VAL STATS: \n ", val_stats)
        print('-----------')

        print("TEST STATS: \n ", test_stats)

    print("All test stats: ", all_test_stats)