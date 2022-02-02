import argparse

DATASETS = ['twitter']


def parse_args():
    parser = argparse.ArgumentParser()
    # Training / Optimization args
    parser.add_argument('--device', type=int, default=0, help='cuda device')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--drop_last', type=int, default=1)

    parser.add_argument('--learning_rate', type=float, default=2*1e-4)
    parser.add_argument('--pret_add_channels', type=int, default=1,
                        help="relevant when using context and pretrained resnet as prediction net")
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam'])

    # DRO
    parser.add_argument('--use_robust_loss', type=int, default=0,
                        help='Use robust loss algo from DRNN paper')
    parser.add_argument('--robust_step_size', type=float, default=0.01,
                        help='When using robust loss algo from DRNN paper')

    # Model args
    parser.add_argument('--model', type=str, default='ContextualConvNet',
                        choices=['ContextualMLP', 'ContextualConvNet'])

    parser.add_argument('--pretrained', type=int, default=1,
                        help='Pretrained resnet')
    # If model is Convnet
    parser.add_argument('--prediction_net', type=str, default='convnet',
                        choices=['resnet18', 'resnet34', 'resnet50', 'convnet'])

    parser.add_argument('--n_context_channels', type=int, default=3, help='Used when using a convnet/resnet')
    parser.add_argument('--use_context', type=int, default=1, help='Whether or not to condition the model.')

    # Data args
    parser.add_argument('--dataset', type=str, default='twitter', choices=DATASETS)
    parser.add_argument('--data_dir', type=str, default='data/twitter')

    # Data sampling
    parser.add_argument('--meta_batch_size', type=int, default=2, help='Number of classes')
    parser.add_argument('--support_size', type=int, default=50,
                        help='Support size: same as what we call batch size in the appendix')
    parser.add_argument('--shuffle_train', type=int, default=1,
                        help='Only relevant when no clustered sampling = 0 \
                        and --uniform_over_groups 0')

    parser.add_argument('--use_val_set', type=int, default=1,help='use validation set')
    # meta batch sampling
    parser.add_argument('--sampling_type', type=str, default='meta_batch_groups',
                        choices=['meta_batch_mixtures', 'meta_batch_groups', 'uniform_over_groups', 'regular'],
                        help='Sampling type')
    parser.add_argument('--uniform_over_groups', type=int, default=10,
                        help='Sample groups uniformly. This is relevant when sampling_type == meta_batch_groups')
    parser.add_argument('--eval_corners_only', type=int, default=1,
                        help='Are evaluating mixtures or corners?')

    # Evalaution
    parser.add_argument('--n_test_dists', type=int, default=30,
                        help='Number of test distributions to evaluate on. These are sampled uniformly.')
    parser.add_argument('--n_test_per_dist', type=int, default=2000,
                        help='Number of examples to evaluate on per test distribution')
    parser.add_argument('--crop_type', type=float, default=0)

    # Logging
    parser.add_argument('--seed', type=int, default=2021666, help='Seed')
    parser.add_argument('--use_kg', type=bool, default=False, help='Whether to use KG enhanced model')
    parser.add_argument('--plot', type=int, default=0, help='Plot or not')
    parser.add_argument('--experiment_name', type=str, default='debug')
    parser.add_argument('--epochs_per_eval', type=int, default=1)
    parser.add_argument('--debug', type=int, default=0)

    parser.add_argument('--num_workers', type=int, default=8, help='Num workers for pytorch data loader')
    parser.add_argument('--pin_memory', type=int, default=1, help='Pytorch loader pin memory. \
                        Best practice is to use this')


    return parser.parse_args()