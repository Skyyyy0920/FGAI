import torch
import argparse

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def get_args():
    parser = argparse.ArgumentParser(description="MTNet's args")
    # Operation environment
    parser.add_argument('--seed',
                        type=int,
                        default=20020310,
                        help='Random seed')
    parser.add_argument('--device',
                        type=str,
                        default=device,
                        help='Running on which device')
    # Data
    parser.add_argument('--task',
                        type=str,
                        default='node-level',
                        # default='graph-level',
                        help='task')
    parser.add_argument('--dataset',
                        type=str,
                        default='PubmedGraphDataset',
                        help='Dataset name')

    parser.add_argument('--save_path',
                        type=str,
                        default='./checkpoints/',
                        help='Checkpoints saving path')

    args = parser.parse_args()
    return args
