import os
from datetime import datetime
from pathlib import Path

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
from default_param import default_dict
# Set parameters

args = parse_args()
set_seed(args.seed)
batch_size = args.meta_batch_size * args.support_size
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
if 'group' in args.sampling_type and not args.eval_corners_only:
    raise ValueError
Dict = default_dict[args.dataset]
args.data_dir = Dict['data_dir']
print('Using device', device)

# Save folder
datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
ckpt_dir = Path('output') / 'checkpoints' / f'{args.experiment_name}_{args.seed}_{datetime_now}'
args.ckpt_dir = ckpt_dir

# Get data
tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
train_loader, train_eval_loader, val_loader, test_loader = datasets.get_loaders(args, tokenizer)

# Get KG
if args.use_kg:
    from utils.kg_graph_v2 import Sentence2Graph
    kg_folder = 'data/wn18rr_kg'
    KG =  Sentence2Graph(kg_folder=kg_folder)
else: 
    KG = None

if val_loader == None:
    val_loader = train_eval_loader
args.n_groups = train_loader.dataset.n_groups
seq_len = train_loader.dataset[0][0].size(0)
# Load lots of parameters and get model
model = models.ContextualNet(
    ContextNet=models.ALBERTContextModel, PredictionNet=Dict['PredictionNet'],  # context model and predictor
    embedding_dim=Dict['embedding_dim'], pred_hidden_dim=Dict['pred_hidden_dim'], pred_output_dim=Dict['pred_output_dim'],  # dims for predictor
    context_hidden_dim=Dict['context_hidden_dim'], # dims for context model
    use_context=args.use_context,# specify whether to use context net, and the type for the context net(if YES)
    support_size=args.support_size, KG=KG# meta-learning
).to(device)


# Loss Fn
if args.use_robust_loss:
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    loss_computer = LossComputer(loss_fn, is_robust=True,
                                    dataset=train_loader.dataset,
                                    step_size=args.robust_step_size,
                                    device=device,
                                    args=args)
else:
    loss_fn = nn.CrossEntropyLoss()
# Optimizer
if args.optimizer == 'adam': # This is used for MNIST.
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=args.learning_rate)
elif args.optimizer == 'sgd':
    # From DRNN paper
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay)

# Train loop
best_worst_case_acc = 0
best_worst_case_acc_epoch = 0
avg_val_acc = 0
empirical_val_acc = 0

for epoch in range(1, args.num_epochs + 1):
    total_loss = 0
    total_accuracy = 0
    total_examples = 0

    model.train()

    for x, y, group_ids, weight, raw_x, _ in train_loader:

        # Put on GPU
        x = x.to(device)
        y = y.to(device)
        weight = weight.to(device)
        # Forward
        logits = model(sentence=x, attention_mask=weight, raw_x=raw_x)
        if args.use_robust_loss:
            group_ids = group_ids.to(device)
            loss = loss_computer.loss(logits, y, None, 
            group_ids, is_training=True)
        else: 
            loss = loss_fn(logits, y)

        # Evaluate
        preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
        accuracy = np.sum(preds == y.detach().cpu().numpy().reshape(-1))

        n_instance = y.view(-1).size(0)
        total_accuracy += accuracy
        total_loss += loss.item() * n_instance
        total_examples += n_instance
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch [%d/%d], Train Loss: %3.4f, Train Accuracy %3.4f" % \
        (epoch, args.num_epochs, total_loss/total_examples, total_accuracy/total_examples))
    
    if epoch % args.epochs_per_eval == 0:

        worst_case_acc, stats = utils.evaluate_groups(args, model, val_loader, epoch, split='val') # validation

        # Track early stopping values with respect to worst case.
        if worst_case_acc > best_worst_case_acc:
            best_worst_case_acc = worst_case_acc
            models.save_model(model, ckpt_dir, epoch, device)

        print("       Val Ave ACC %3.4f, Worst Case Acc: %3.4f" % (stats['average_val_acc'], worst_case_acc))