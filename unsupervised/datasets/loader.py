# code from arm: https://github.com/henrikmarklund/arm

import numpy as np
import torch

from .utils.samplers import ClusteredMixSampler, ConstantMixSampler, ConstantGroupSampler, ClusteredGroupSampler
from .twitter_dataset import TwitterDataset

def get_one_hot(values, num_classes):    # putting this here for now so you can get it working in one copy paste
    return np.eye(num_classes)[values]

def get_loader(dataset, sampling_type=None, batch_size=None, meta_batch_size=None,
               support_size=None, shuffle=True, meta_distribution=None,
               pin_memory=True, args=None):
    """Returns a data loader that sample meta_batches of data where each
            meta batch contains a set of support batches. Each support batch
            contain examples all having the same angle

    """

    if sampling_type == 'meta_batch_mixtures': # Sample support batches from multiple sub distributions
        batch_sampler = ClusteredMixSampler(dataset, meta_batch_size, support_size,
                                         args=args)

        batch_size = 1
        shuffle = None
        sampler=None
        drop_last = False

    elif sampling_type == 'meta_batch_groups': # Sample support batches from multiple sub distributions
        batch_sampler = ClusteredGroupSampler(dataset, meta_batch_size, support_size,
                                          uniform_over_groups=args.uniform_over_groups)

        batch_size = 1
        shuffle = None
        sampler=None
        drop_last = False
        print("meta batch group")

    elif sampling_type == 'constant_mixture': # Sample batches from a sub distribution
        sampler = ConstantMixSampler(dataset, replacement=True)
        batch_sampler = None
        drop_last=True
        shuffle = None
        print("constant mixture")

    elif sampling_type == 'constant_group': # Sample batches from specific group
        batch_sampler = ConstantGroupSampler(dataset, batch_size, replacement=False)

        batch_size = 1
        shuffle = None
        sampler=None
        drop_last = False
        print("constant group")

    elif sampling_type == 'uniform_over_groups':
        # Sample batches from the sub distribution that is uniform over groups
        # Put each group uniformly
        sampler = ConstantMixSampler(dataset, replacement=True)
        batch_sampler = None
        sampler.set_uniform_dist_over_groups()
        drop_last = True
        shuffle = None
        print("uniform over groups")
    elif sampling_type == 'regular': # Sample each example uniformly
        sampler = None
        batch_sampler = None
        if args is not None:
            drop_last = bool(args.drop_last)
        else:
            drop_last = False
        if shuffle == 0:
            shuffle = False
    loader = torch.utils.data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  sampler=sampler,
                                  batch_sampler=batch_sampler,
                                  pin_memory=pin_memory,
                                  drop_last=drop_last)
    return loader


def get_dataset(args, tokenizer):

    if args.dataset == 'twitter':
        train_dataset = TwitterDataset('train', args.data_dir, tokenizer=tokenizer)
        test_dataset = TwitterDataset('test', args.data_dir, tokenizer=tokenizer)
        val_dataset = TwitterDataset('val', args.data_dir, tokenizer=tokenizer)
    # elif args.dataset == 'reddit':
    #     train_dataset = RedditDataset('train', args.data_dir, tokenizer=tokenizer)
    #     test_dataset = RedditDataset('test', args.data_dir, tokenizer=tokenizer)
    #     val_dataset = RedditDataset('val', args.data_dir, tokenizer=tokenizer)
    else:
        raise ValueError

    return train_dataset, val_dataset, test_dataset


def get_loaders(args, tokenizer):

    train_dataset, val_dataset, test_dataset = get_dataset(args, tokenizer=tokenizer)
    batch_size = args.meta_batch_size * args.support_size

    train_loader = get_loader(train_dataset, sampling_type=args.sampling_type,
                              batch_size=batch_size,
                              meta_batch_size=args.meta_batch_size,
                              support_size=args.support_size,
                              shuffle=args.shuffle_train,
                              pin_memory=args.pin_memory,
                              args=args)

    # The test loader will sample examples from a sub distribution that is set during evaluation
    # You can update this sub distribution during evaluation

    eval_sampling_type = 'constant_group' if args.eval_corners_only else 'constant_mixture'

    if 'eval_deterministic' in args and args.eval_deterministic:
        eval_sampling_type = 'regular'

    train_eval_loader = get_loader(train_dataset, eval_sampling_type,
                          batch_size, shuffle=False,
                          pin_memory=args.pin_memory)

    val_loader = get_loader(val_dataset, eval_sampling_type,
                            batch_size, shuffle=False,
                            pin_memory=args.pin_memory)

    test_loader = get_loader(test_dataset, eval_sampling_type,
                          batch_size, shuffle=False,
                          pin_memory=args.pin_memory)

    return train_loader, train_eval_loader, val_loader, test_loader

