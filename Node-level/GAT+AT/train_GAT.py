import time
import yaml
import argparse
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from utils import *
from models import GATNodeClassifier
from load_dataset import load_dataset
from trainer import AdvTrainer
from attackers import PGD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

if __name__ == '__main__':
    # dataset ='ogbn-arxiv'
    # dataset='ogbn-products'
    # dataset='ogbn-papers100M'
    # dataset='questions'
    # dataset='amazon-ratings'
    # dataset='roman-empire'
    # dataset = 'pubmed'
    # dataset = 'amazon_photo'
    # dataset = 'amazon_cs'
    # dataset = 'coauthor_cs'
    dataset = 'coauthor_phy'

    with open(f"./optimized_hyperparameter_configurations/{dataset}.yml", 'r') as file:
        args = yaml.full_load(file)
    args = argparse.Namespace(**args)
    args.device = device

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging_time = time.strftime('%H-%M', time.localtime())
    save_dir = os.path.join("GAT_checkpoints", f"{dataset}_{logging_time}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s %(levelname)s]%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(save_dir, f'{dataset}.log'))
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
    GAT_AT = GATNodeClassifier(in_feats=in_feats,
                               hid_dim=args.hid_dim,
                               n_classes=num_classes,
                               n_layers=args.n_layers,
                               n_heads=args.n_heads,
                               feat_drop=args.feat_drop,
                               attn_drop=args.attn_drop).to(device)
    optimizer = optim.Adam(GAT_AT.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
    attacker_ = PGD(epsilon=args.epsilon,
                    n_epoch=args.n_epoch_attack,
                    n_inject_max=args.n_inject_max,
                    n_edge_max=args.n_edge_max,
                    feat_lim_min=-1,
                    feat_lim_max=1,
                    device=device)

    total_params = sum(p.numel() for p in GAT_AT.parameters())
    logging.info(f"Total parameters: {total_params}")
    logging.info(f"Model: {GAT_AT}")
    logging.info(f"Optimizer: {optimizer}")

    trainer = AdvTrainer(GAT_AT, optimizer, criterion, attacker_, args)
    idx_split = train_idx, valid_idx, test_idx
    trainer.train(features, adj, label, idx_split)

    orig_outputs, orig_graph_repr, orig_att = \
        evaluate_node_level(GAT_AT, features, adj, label, test_idx, num_classes == 2)

    torch.save(GAT_AT.state_dict(), os.path.join(save_dir, 'model_parameters.pth'))

    attacker = PGD(epsilon=args.epsilon,
                   n_epoch=args.n_epoch_attack,
                   n_inject_max=args.n_inject_max,
                   n_edge_max=args.n_edge_max,
                   feat_lim_min=-1,
                   feat_lim_max=1,
                   device=device)

    GAT_AT.eval()
    adj_delta, feats_delta = attacker.attack(GAT_AT, adj, features, test_idx, None)
    new_outputs, new_graph_repr, new_att = GAT_AT(torch.cat((features, feats_delta), dim=0), adj_delta)
    new_outputs, new_graph_repr, new_att = \
        new_outputs[:orig_outputs.shape[0]], new_graph_repr[:orig_graph_repr.shape[0]], new_att[:orig_att.shape[0]]
    pred = torch.argmax(new_outputs[test_idx], dim=1)
    accuracy = accuracy_score(label[test_idx].cpu(), pred.cpu())
    logging.info(f"Accuracy after attack: {accuracy:.4f}")

    TVD_score = TVD(orig_att, new_att) / len(orig_att)
    JSD_score = JSD(orig_att, new_att) / len(orig_att)
    logging.info(f"JSD: {JSD_score}")
    logging.info(f"TVD: {TVD_score}")

    sp.save_npz(os.path.join(save_dir, 'adj_delta.npz'), adj_delta)
    torch.save(feats_delta, os.path.join(save_dir, 'feats_delta.pth'))

    fidelity_pos_list, fidelity_neg_list = compute_fidelity(GAT_AT, adj, features, label, test_idx)
    logging.info(f"fidelity_pos: {fidelity_pos_list}")
    logging.info(f"fidelity_neg: {fidelity_neg_list}")
    data = pd.DataFrame({'fidelity_pos': fidelity_pos_list, 'fidelity_neg': fidelity_neg_list})
    data.to_csv(os.path.join(save_dir, 'fidelity_data.txt'), sep=',', index=False)
