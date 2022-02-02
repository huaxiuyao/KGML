import argparse
import random

import ipdb
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import tqdm
from data_generator import AmazonReview, Huffpost
from protonet import Protonet
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='KGML')
parser.add_argument('--datasource', default='amazonreview', type=str,
                    help='amazonreview')
parser.add_argument('--select_data', default=-1, type=int, help='-1,0,1,2,3')
parser.add_argument('--test_dataset', default=-1, type=int,
                    help='which dataset to be test: 0: bird, 1: texture, 2: aircraft, 3: fungi, -1 is test all')
parser.add_argument('--num_classes', default=2, type=int,
                    help='number of classes used in classification (e.g. 5-way classification).')
parser.add_argument('--num_test_task', default=600, type=int, help='number of test tasks.')
parser.add_argument('--test_epoch', default=-1, type=int, help='test epoch, only work when test start')

## Training options
parser.add_argument('--metatrain_iterations', default=3000, type=int,
                    help='number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid
parser.add_argument('--meta_batch_size', default=25, type=int, help='number of tasks sampled per meta-update')
parser.add_argument('--update_lr', default=0.001, type=float, help='inner learning rate')
parser.add_argument('--meta_lr', default=2e-5, type=float, help='the base learning rate of the generator')
parser.add_argument('--num_updates', default=5, type=int, help='num_updates in maml')
parser.add_argument('--num_updates_test', default=10, type=int, help='num_updates in maml')
parser.add_argument('--update_batch_size', default=10, type=int,
                    help='number of examples used for inner gradient update (K for K-shot learning).')
parser.add_argument('--update_batch_size_eval', default=10, type=int,
                    help='number of examples used for inner gradient test (K for K-shot learning).')
parser.add_argument('--num_filters', default=64, type=int,
                    help='number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')

## Logging, saving, and testing options
parser.add_argument('--log', default=1, type=int, help='if false, do not log summaries, for debugging code.')
parser.add_argument('--logdir', default='xxx', type=str,
                    help='directory for summaries and checkpoints.')
parser.add_argument('--datadir', default='xxx', type=str, help='directory for datasets.')
parser.add_argument('--resume', default=0, type=int, help='resume training if there is a model available')
parser.add_argument('--train', default=1, type=int, help='True to train, False to test.')
parser.add_argument('--test_set', default=1, type=int,
                    help='Set to true to test on the the test set, False for the validation set.')
parser.add_argument('--use_kg', default=0, type=int, help='use mixup or not')
parser.add_argument('--trail', default=0, type=int, help='trail for each layer')
parser.add_argument('--warm_epoch', default=0, type=int, help='warm start epoch')
parser.add_argument('--ratio', default=1.0, type=float, help='warm start epoch')
parser.add_argument('--knn', default=1, type=int, help='use knn')



args = parser.parse_args()
print(args)

assert torch.cuda.is_available()

random.seed(1)
np.random.seed(2)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

exp_string = 'ProtoNet' + '.data_' + str(args.datasource) + '.cls_' + str(args.num_classes) + '.mbs_' + str(
    args.meta_batch_size) + '.ubs_' + str(
    args.update_batch_size) + '.metalr' + str(args.meta_lr) + '.innerlr' + str(args.update_lr) + '.numupdates' + str(args.num_updates)

if args.num_filters != 64:
    exp_string += '.hidden' + str(args.num_filters)
if args.trail > 0:
    exp_string += '.trail{}'.format(args.trail)
if args.use_kg:
    exp_string += '.kg'
if not args.knn:
    exp_string += '.noknn'

print(exp_string)


def train(args, protonet, optimiser, KG):
    Print_Iter = 100
    Save_Iter = 200
    print_loss, print_acc, print_loss_support = 0.0, 0.0, 0.0

    if args.datasource == 'amazonreview':
        dataset = AmazonReview(args, 'train', KG)
    elif args.datasource == 'huffpost':
        dataset = Huffpost(args, 'train', KG)
    # dataloader = DataLoader(dataset, batch_size=args.meta_batch_size, num_workers=4)

    if not os.path.exists(args.logdir + '/' + exp_string + '/'):
        os.makedirs(args.logdir + '/' + exp_string + '/')

    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(dataset):
        protonet.train()
        if step > args.metatrain_iterations:
            break
        task_losses = []
        task_acc = []

        for meta_batch in range(args.meta_batch_size):
            loss_val, acc_val = protonet(x_spt[meta_batch], y_spt[meta_batch], x_qry[meta_batch], y_qry[meta_batch])
            task_losses.append(loss_val)
            task_acc.append(acc_val)

        meta_batch_loss = torch.stack(task_losses).mean()
        meta_batch_acc = torch.stack(task_acc).mean()

        optimiser.zero_grad()
        meta_batch_loss.backward()
        optimiser.step()

        if step != 0 and step % Print_Iter == 0:
            print(
                'iter: {}, loss_all: {}, acc: {}'.format(
                    step, print_loss, print_acc))

            print_loss, print_acc = 0.0, 0.0
        else:
            print_loss += meta_batch_loss / Print_Iter
            print_acc += meta_batch_acc / Print_Iter

        if step != 0 and step % Save_Iter == 0:
            torch.save(protonet.state_dict(),
                       '{0}/{2}/model{1}'.format(args.logdir, step, exp_string))


def test(args, protonet, test_epoch, KG, type='test'):
    protonet.eval()
    res_acc = []
    args.meta_batch_size = 1

    if args.datasource == 'amazonreview':
        dataset = AmazonReview(args, type, KG)
    elif args.datasource == 'huffpost':
        dataset = Huffpost(args, type, KG)

    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(dataset):
        if step > 300:
            break
        # x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to("cuda"), y_spt.squeeze(0).to("cuda"), \
        #                              x_qry.squeeze(0).to("cuda"), y_qry.squeeze(0).to("cuda")
        with torch.no_grad():
            _, acc_val = protonet(x_spt[0], y_spt[0], x_qry[0], y_qry[0])
            res_acc.append(acc_val.item())

    res_acc = np.array(res_acc)

    print('{}_epoch is {}, acc is {}, ci95 is {}'.format(type, test_epoch, np.mean(res_acc),
                                                           1.96 * np.std(res_acc) / np.sqrt(
                                                               args.num_test_task * args.meta_batch_size)))


def main():
    if args.use_kg:
        from kg_graph import Sentence2Graph
        kg_folder = '../wn18rr_kg'
        KG = Sentence2Graph(args=args, kg_folder=kg_folder)
    else:
        KG = None

    protonet = Protonet(args, KG).to("cuda")

    if args.resume == 1 and args.train == 1:
        model_file = '{0}/{2}/model{1}'.format(args.logdir, args.test_epoch, exp_string)
        print(model_file)
        protonet.load_state_dict(torch.load(model_file))

    meta_optimiser = torch.optim.AdamW(list(protonet.parameters()),
                                      lr=args.meta_lr, weight_decay=args.weight_decay)

    if args.train == 1:
        train(args, protonet, meta_optimiser, KG)
    else:
        for test_epoch in range(200, 20000, 200):
            try:
                model_file = '{0}/{2}/model{1}'.format(args.logdir, test_epoch, exp_string)
                protonet.load_state_dict(torch.load(model_file))
                # test(args, protonet, test_epoch, KG, type='val')
                test(args, protonet, test_epoch, KG, type='test')
            except IOError:
                continue


if __name__ == '__main__':
    main()
