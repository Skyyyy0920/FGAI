import time
import argparse
import scipy.sparse as sp
import torch.nn as nn
import torch.optim as optim
from utils import *
from models import GATv2NodeClassifier
from load_dataset import load_dataset
from trainer import VanillaTrainer
from attackers import PGD

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
                        # default='cora',
                        default='pubmed',
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
    save_dir = os.path.join("GATv2_checkpoints", f"{args.dataset}")
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
        GATv2 = GATv2NodeClassifier(in_feats=in_feats,
                                    hid_dim=128,
                                    n_classes=num_classes,
                                    n_layers=3,
                                    n_heads=[4, 2, 1],
                                    feat_drop=0.05,
                                    attn_drop=0).to(args.device)
        optimizer = optim.Adam(GATv2.parameters(),
                               lr=1e-2,
                               weight_decay=0)
    elif args.dataset in ['cora', 'citeseer']:
        GATv2 = GATv2NodeClassifier(in_feats=in_feats,
                                    hid_dim=8,
                                    n_classes=num_classes,
                                    n_layers=1,
                                    n_heads=[8],
                                    feat_drop=0.6,
                                    attn_drop=0.6).to(args.device)
        optimizer = optim.Adam(GATv2.parameters(),
                               lr=5e-3,
                               weight_decay=5e-4)
    elif args.dataset == 'pubmed':
        GATv2 = GATv2NodeClassifier(in_feats=in_feats,
                                    hid_dim=8,
                                    n_classes=num_classes,
                                    n_layers=1,
                                    n_heads=[8],
                                    feat_drop=0.6,
                                    attn_drop=0.6).to(args.device)
        optimizer = optim.Adam(GATv2.parameters(),
                               lr=1e-2,
                               weight_decay=1e-3)
    else:
        GATv2 = GATv2NodeClassifier(in_feats=in_feats,
                                    hid_dim=128,
                                    n_classes=num_classes,
                                    n_layers=3,
                                    n_heads=[4, 2, 1],
                                    feat_drop=0.05,
                                    attn_drop=0).to(args.device)
        optimizer = optim.Adam(GATv2.parameters(),
                               lr=1e-3,
                               weight_decay=0)

    total_params = sum(p.numel() for p in GATv2.parameters())
    logging.info(f"Total parameters: {total_params}")
    logging.info(f"Model: {GATv2}")
    logging.info(f"Optimizer: {optimizer}")

    std_trainer = VanillaTrainer(GATv2, criterion, optimizer, args)

    orig_outputs, orig_graph_repr, orig_att = std_trainer.train(features, adj, label, train_idx, valid_idx)

    evaluate_node_level(GATv2, criterion, features, adj, label, test_idx)

    torch.save(GATv2.state_dict(), os.path.join(save_dir, 'model_parameters.pth'))
    tensor_dict = {'orig_outputs': orig_outputs, 'orig_graph_repr': orig_graph_repr, 'orig_att': orig_att}
    torch.save(tensor_dict, os.path.join(save_dir, 'tensors.pth'))

    adj_perturbed = sp.load_npz(f'./vanilla_model/{args.dataset}/adj_delta.npz')
    feats_perturbed = torch.load(f'./vanilla_model/{args.dataset}/feats_delta.pth')

    GATv2.eval()
    new_outputs, new_graph_repr, new_att = GATv2(torch.cat((features, feats_perturbed), dim=0), adj_perturbed)
    new_outputs, new_graph_repr, new_att = \
        new_outputs[:orig_outputs.shape[0]], new_graph_repr[:orig_graph_repr.shape[0]], new_att[:orig_att.shape[0]]

    TVD_score = TVD(orig_att, new_att) / len(orig_att)
    JSD_score = JSD(orig_att, new_att) / len(orig_att)
    logging.info(f"JSD: {JSD_score}")
    logging.info(f"TVD: {TVD_score}")

    fidelity_pos, fidelity_neg, TVD_pos, TVD_neg = compute_fidelity(GATv2, adj, features, label)
    logging.info(f"fidelity_pos: {fidelity_pos}, fidelity_neg: {fidelity_neg}")
    logging.info(f"TVD_pos: {TVD_pos}, TVD_neg: {TVD_neg}")
