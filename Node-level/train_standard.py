import time
import argparse
import scipy.sparse as sp
import torch.nn as nn
import torch.optim as optim
from utils import *
from models import GATNodeClassifier
from load_dataset import load_dataset
from trainer import StandardTrainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser(description="MTNet's args")

    # Operation environment
    parser.add_argument('--seed', type=int, default=20010920, help='Random seed')
    parser.add_argument('--device', type=str, default=device, help='Running on which device')

    # Data
    parser.add_argument('--dataset',
                        type=str,
                        # default='ogbn-arxiv',
                        # default='ogbn-products',
                        # default='ogbn-papers100M',
                        default='cora',
                        # default='pubmed',
                        # default='citeseer',
                        help='Dataset name')

    # Experimental Setup
    parser.add_argument('--num_epochs', type=int, default=200, help='Training epoch')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    # setup_seed(args.seed)  # make the experiment repeatable

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging_time = time.strftime('%m-%d_%H-%M', time.localtime())
    save_dir = os.path.join("standard_model", f"{args.dataset}")
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

    logging.info(f"Using device: {args.device}")
    logging.info(f"PyTorch Version: {torch.__version__}")
    logging.info(f"args: {args}")
    logging.info(f"Saving path: {save_dir}")

    g, label, train_idx, valid_idx, test_idx, num_classes = load_dataset(args)
    features = g.ndata["feat"]
    in_feats = features.shape[1]
    src, dst = g.edges()
    num_nodes = g.number_of_nodes()
    adj = sp.csr_matrix((np.ones(len(src)), (src.cpu().numpy(), dst.cpu().numpy())), shape=(num_nodes, num_nodes))
    del g

    criterion = nn.CrossEntropyLoss()
    if args.dataset == 'ogbn-arxiv':
        standard_model = GATNodeClassifier(in_feats=in_feats,
                                           hid_dim=128,
                                           n_classes=num_classes,
                                           n_layers=3,
                                           n_heads=[4, 2, 1],
                                           feat_drop=0.05,
                                           attn_drop=0).to(args.device)
        optimizer = optim.Adam(standard_model.parameters(),
                               lr=1e-2,
                               weight_decay=0)
    elif args.dataset in ['cora', 'citeseer']:
        standard_model = GATNodeClassifier(in_feats=in_feats,
                                           hid_dim=8,
                                           n_classes=num_classes,
                                           n_layers=1,
                                           n_heads=[8],
                                           feat_drop=0.6,
                                           attn_drop=0.6).to(args.device)
        optimizer = optim.Adam(standard_model.parameters(),
                               lr=5e-3,
                               weight_decay=5e-4)
    elif args.dataset == 'pubmed':
        standard_model = GATNodeClassifier(in_feats=in_feats,
                                           hid_dim=8,
                                           n_classes=num_classes,
                                           n_layers=1,
                                           n_heads=[8],
                                           feat_drop=0.6,
                                           attn_drop=0.6).to(args.device)
        optimizer = optim.Adam(standard_model.parameters(),
                               lr=1e-2,
                               weight_decay=1e-3)
    else:
        standard_model = GATNodeClassifier(in_feats=in_feats,
                                           hid_dim=128,
                                           n_classes=num_classes,
                                           n_layers=3,
                                           n_heads=[4, 2, 1],
                                           feat_drop=0.05,
                                           attn_drop=0).to(args.device)
        optimizer = optim.Adam(standard_model.parameters(),
                               lr=1e-3,
                               weight_decay=0)

    total_params = sum(p.numel() for p in standard_model.parameters())
    logging.info(f"Total parameters: {total_params}")
    logging.info(f"Model: {standard_model}")
    logging.info(f"Optimizer: {optimizer}")

    std_trainer = StandardTrainer(standard_model, criterion, optimizer, args)

    orig_outputs, orig_graph_repr, orig_att = std_trainer.train(features, adj, label, train_idx, valid_idx)

    evaluate_node_level(standard_model, criterion, features, adj, label, test_idx)

    torch.save(standard_model.state_dict(), os.path.join(save_dir, 'model_parameters.pth'))
    tensor_dict = {'orig_outputs': orig_outputs, 'orig_graph_repr': orig_graph_repr, 'orig_att': orig_att}
    torch.save(tensor_dict, os.path.join(save_dir, 'tensors.pth'))
