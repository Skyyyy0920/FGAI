import torch
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'


def get_args():
    parser = argparse.ArgumentParser(description="MTNet's args")

    # Operation environment
    parser.add_argument('--seed',
                        type=int,
                        default=20010920,
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
                        # default='PubmedGraphDataset',
                        # default='PPIDataset',
                        # default='RedditDataset',
                        # default='YelpDataset',
                        default='CoraGraphDataset',
                        help='Dataset name')

    # Experimental Setup
    parser.add_argument('--num_epochs',
                        type=int,
                        default=200,
                        help='Training epoch')

    parser.add_argument('--save_path',
                        type=str,
                        default='./checkpoints/',
                        help='Checkpoints saving path')

    args = parser.parse_args()
    return args
