import argparse
import random
import numpy as np
import torch
import os
from data_generator import AmazonReview, Huffpost
from maml import MAML
import time

parser = argparse.ArgumentParser(description='KGML')
parser.add_argument('--datasource', default='amazonreview', type=str,
                    help='amazonreview')
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
parser.add_argument('--trial', default=0, type=int, help='trial')


args = parser.parse_args()
print(args)

assert torch.cuda.is_available()

random.seed(1)
np.random.seed(2)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

exp_string = 'Cross_Task' + '.data_' + str(args.datasource) + '.cls_' + str(args.num_classes) + '.mbs_' + str(
    args.meta_batch_size) + '.ubs_' + str(
    args.update_batch_size) + '.metalr' + str(args.meta_lr) + '.innerlr' + str(args.update_lr) + '.numupdates' + str(args.num_updates)

if args.trial > 0:
    exp_string += '.trial{}'.format(args.trial)
if args.use_kg:
    exp_string += '.kg'

print(exp_string)


def train(args, maml, optimiser, KG):
    Print_Iter = 100
    Save_Iter = 100
    print_loss, print_acc, print_loss_support = 0.0, 0.0, 0.0

    if args.datasource == 'amazonreview':
        dataset = AmazonReview(args, 'train', KG)
    elif args.datasource == 'huffpost':
        dataset = Huffpost(args, 'train', KG)

    if not os.path.exists(args.logdir + '/' + exp_string + '/'):
        os.makedirs(args.logdir + '/' + exp_string + '/')

    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(dataset):
        maml.train()
        if step > args.metatrain_iterations:
            break
        task_losses = []
        task_acc = []

        for meta_batch in range(args.meta_batch_size):
            loss_val, acc_val = maml(x_spt[meta_batch], y_spt[meta_batch], x_qry[meta_batch], y_qry[meta_batch])
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
            torch.save(maml.state_dict(),
                       '{0}/{2}/model{1}'.format(args.logdir, step, exp_string))


def test(args, maml, test_epoch, KG, type):

    res_acc = []
    args.meta_batch_size = 1

    if args.datasource == 'amazonreview':
        dataset = AmazonReview(args, type, KG)
    elif args.datasource == 'huffpost':
        dataset = Huffpost(args, type, KG)
    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(dataset):
        maml.eval()
        if step > args.num_test_task:
            break
        _, acc_val = maml(x_spt[0], y_spt[0], x_qry[0], y_qry[0])
        res_acc.append(acc_val.item())

    res_acc = np.array(res_acc)

    print('{}_epoch is {}, acc is {}, ci95 is {}'.format(type, test_epoch, np.mean(res_acc),
                                                           1.96 * np.std(res_acc) / np.sqrt(
                                                               args.num_test_task * args.meta_batch_size)))


def main():
    if args.use_kg:
        from kg_graph import Sentence2Graph
        kg_folder = '../wn18rr_kg'
        KG = Sentence2Graph(args, kg_folder=kg_folder)
    else:
        KG = None

    maml = MAML(args, KG).to("cuda")

    if args.resume == 1 and args.train == 1:
        model_file = '{0}/{2}/model{1}'.format(args.logdir, args.test_epoch, exp_string)
        print(model_file)
        maml.load_state_dict(torch.load(model_file))

    meta_optimiser = torch.optim.AdamW(list(maml.parameters()),
                                      lr=args.meta_lr, weight_decay=args.weight_decay)

    if args.train == 1:
        train(args, maml, meta_optimiser, KG)
    else:
        test_epoch = 200
        model_file = '{0}/{2}/model{1}'.format(args.logdir, test_epoch, exp_string)
        maml.load_state_dict(torch.load(model_file))
        test(args, maml, test_epoch, KG, type='val')
        test(args, maml, test_epoch, KG, type='test')


if __name__ == '__main__':
    main()
