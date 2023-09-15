import torch
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'


def get_args():
    parser = argparse.ArgumentParser(description="MTNet's args")

    # Operation environment
    parser.add_argument('--seed', type=int, default=20010920, help='Random seed')
    parser.add_argument('--device', type=str, default=device, help='Running on which device')

    # Data
    parser.add_argument('--task', type=str, default='node-level', help='task')  # default='graph-level'
    parser.add_argument('--dataset',
                        type=str,
                        # default='PubmedGraphDataset',
                        default='CoraGraphDataset',
                        # default='CiteseerGraphDataset',
                        # default='PPIDataset',
                        # default='RedditDataset',
                        # default='YelpDataset',
                        help='Dataset name')

    # Experimental Setup
    parser.add_argument('--num_epochs', type=int, default=300, help='Training epoch')

    parser.add_argument('--pgd_radius', type=float, default=0.05, help='Attack radius')
    parser.add_argument('--pgd_step', type=float, default=10, help='How many step to conduct PGD')
    parser.add_argument('--pgd_step_size', type=float, default=0.02, help='Coefficient of PGD')
    parser.add_argument('--pgd_norm_type', type=str, default="l-infty", help='Which norm of your noise')

    parser.add_argument('--x_pgd_radius', type=float, default=0.1, help='Attack radius')
    parser.add_argument('--x_pgd_step', type=float, default=10, help='How many step to conduct PGD')
    parser.add_argument('--x_pgd_step_size', type=float, default=0.01, help='Coefficient of PGD')
    parser.add_argument('--x_pgd_norm_type', type=str, default="l-infty", help='Which norm of your noise')

    parser.add_argument('--lambda_1', type=float, default=5e-2)
    parser.add_argument('--lambda_2', type=float, default=5e-2)
    parser.add_argument('--lambda_3', type=float, default=5e-2)
    parser.add_argument('--K', type=int, default=2)

    parser.add_argument('--save_path', type=str, default='./checkpoints/', help='Checkpoints saving path')

    args = parser.parse_args()
    return args
