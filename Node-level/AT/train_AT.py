import time
import yaml
import argparse
import torch.nn as nn
import torch.optim as optim
from utils import *
from models import GNNGuard
from load_dataset import load_dataset
from trainer import VanillaTrainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'


def get_args():
    parser = argparse.ArgumentParser(description="AT's args")

    # Data
    parser.add_argument('--dataset',
                        type=str,
                        # default='ogbn-arxiv',
                        # default='ogbn-products',
                        # default='ogbn-papers100M',
                        # default='pubmed',
                        # default='questions',
                        # default='amazon-ratings',
                        # default='roman-empire',
                        # default='amazon_photo',
                        # default='amazon_cs',
                        # default='coauthor_cs',
                        default='coauthor_phy',
                        help='Dataset name')

    # Experimental Setup
    parser.add_argument('--num_epochs', type=int, default=200, help='Training epoch')
    parser.add_argument('--n_inject_max', type=int, default=20)
    parser.add_argument('--n_edge_max', type=int, default=40)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--n_epoch_attack', type=int, default=10)

    parser.add_argument('--hid_dim', type=int, default=8)
    parser.add_argument('--n_heads', type=list, default=[8])
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--feat_drop', type=float, default=0.05)
    parser.add_argument('--attn_drop', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    load_optimized_hyperparameter_configurations = True
    if load_optimized_hyperparameter_configurations:
        with open(f"./optimized_hyperparameter_configurations/{args.dataset}.yml", 'r') as file:
            args = yaml.safe_load(file)
        args = argparse.Namespace(**args)
    args.device = device

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging_time = time.strftime('%H-%M', time.localtime())
    save_dir = os.path.join("AT_checkpoints", f"{args.dataset}_{logging_time}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s %(levelname)s]%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(save_dir, f'{args.dataset}.log'))
    console = logging.StreamHandler()  # Simultaneously output to console
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt='[%(asctime)s %(levelname)s]%(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True

    logging.info(f"Using device: {device}")
    logging.info(f"PyTorch Version: {torch.__version__}")
    logging.info(f"args: {args}")
    logging.info(f"Saving path: {save_dir}")

    adj, features, label, train_idx, valid_idx, test_idx, num_classes = load_dataset(args)
    in_feats = features.shape[1]

    criterion = nn.CrossEntropyLoss()
    AT = GNNGuard(in_feats=in_feats,
                  hid_dim=args.hid_dim,
                  n_classes=num_classes,
                  n_layers=args.n_layers,
                  n_heads=args.n_heads,
                  dropout=0.6).to(device)
    optimizer = optim.Adam(AT.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    total_params = sum(p.numel() for p in AT.parameters())
    logging.info(f"Total parameters: {total_params}")
    logging.info(f"Model: {AT}")
    logging.info(f"Optimizer: {optimizer}")

    std_trainer = VanillaTrainer(AT, criterion, optimizer, args)

    orig_outputs, orig_graph_repr, orig_att = std_trainer.train(features, adj, label, train_idx, valid_idx)

    evaluate_node_level(AT, criterion, features, adj, label, test_idx, num_classes == 2)

    torch.save(AT.state_dict(), os.path.join(save_dir, 'model_parameters.pth'))
    tensor_dict = {'orig_outputs': orig_outputs, 'orig_graph_repr': orig_graph_repr, 'orig_att': orig_att}
    torch.save(tensor_dict, os.path.join(save_dir, 'tensors.pth'))

    tim = '_00-53'
    adj_perturbed = sp.load_npz(f'./vanilla_model/{args.dataset}{tim}/adj_delta.npz')
    feats_perturbed = torch.load(f'./vanilla_model/{args.dataset}{tim}/feats_delta.pth').to(device)

    AT.eval()
    new_outputs, new_graph_repr, new_att = AT(torch.cat((features, feats_perturbed), dim=0), adj_perturbed)
    new_outputs, new_graph_repr, new_att = \
        new_outputs[:orig_outputs.shape[0]], new_graph_repr[:orig_graph_repr.shape[0]], new_att[:orig_att.shape[0]]

    TVD_score = TVD(orig_att, new_att) / len(orig_att)
    JSD_score = JSD(orig_att, new_att) / len(orig_att)
    logging.info(f"JSD: {JSD_score}")
    logging.info(f"TVD: {TVD_score}")

    fidelity_pos, fidelity_neg = compute_fidelity(AT, adj, features, label, test_idx)
    logging.info(f"fidelity_pos: {fidelity_pos}, fidelity_neg: {fidelity_neg}")
