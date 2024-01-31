import time
import yaml
import argparse
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from utils import *
from models import GATv2NodeClassifier
from trainer import AdvTrainer
from attackers import PGD
from load_dataset import load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # dataset = 'amazon_photo'
    # dataset = 'amazon_cs'
    # dataset = 'coauthor_phy'
    # dataset = 'coauthor_cs'
    # dataset = 'pubmed'
    dataset = 'ogbn-arxiv'

    with open(f"./optimized_hyperparameter_configurations/{dataset}.yml", 'r') as file:
        args = yaml.full_load(file)
    args = argparse.Namespace(**args)
    args.device = device
    logging_time = time.strftime('%H-%M', time.localtime())
    save_dir = os.path.join("AT_checkpoints", f"{dataset}_{logging_time}")
    logging_config(save_dir)
    logging.info(f"Using device: {device}")
    logging.info(f"args: {args}")
    logging.info(f"Saving path: {save_dir}")

    g, adj, features, label, train_idx, valid_idx, test_idx, num_classes = load_dataset(args)
    in_feats = features.shape[1]

    criterion = nn.CrossEntropyLoss()
    GAT_AT = GATv2NodeClassifier(
        feats_size=in_feats,
        hidden_size=args.hid_dim,
        out_size=num_classes,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        feat_drop=args.feat_drop,
        attn_drop=args.attn_drop).to(device)
    optimizer = optim.Adam(
        GAT_AT.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)
    attacker_ = PGD(
        epsilon=args.epsilon,
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

    attacker = PGD(
        epsilon=args.epsilon,
        n_epoch=args.n_epoch_attack,
        n_inject_max=args.n_inject_max,
        n_edge_max=args.n_edge_max,
        feat_lim_min=-1,
        feat_lim_max=1,
        device=device)

    GAT_AT.eval()
    adj_delta, feats_delta = attacker.attack(GAT_AT, adj, features, test_idx, None)
    sp.save_npz(os.path.join(save_dir, 'adj_delta.npz'), adj_delta)
    torch.save(feats_delta, os.path.join(save_dir, 'feats_delta.pth'))

    feats_ = torch.cat((features, feats_delta), dim=0)
    new_outputs, new_graph_repr, new_att = GAT_AT(feats_, adj_delta)
    new_outputs, new_graph_repr, new_att = \
        new_outputs[:orig_outputs.shape[0]], new_graph_repr[:orig_graph_repr.shape[0]], new_att[:orig_att.shape[0]]
    pred = torch.argmax(new_outputs[test_idx], dim=1)
    accuracy = accuracy_score(label[test_idx].cpu(), pred.cpu())
    logging.info(f"Accuracy after attack: {accuracy:.4f}")

    TVD_score = TVD(orig_outputs, new_outputs) / len(orig_outputs)
    JSD_score = JSD(orig_att, new_att) / len(orig_att)
    logging.info(f"JSD: {JSD_score}")
    logging.info(f"TVD: {TVD_score}")

    f_pos_list, f_neg_list = compute_fidelity(GAT_AT, adj, features, label, test_idx, orig_att)
    logging.info(f"fidelity_pos: {f_pos_list}")
    logging.info(f"fidelity_neg: {f_neg_list}")
    data = pd.DataFrame({'fidelity_pos': f_pos_list, 'fidelity_neg': f_neg_list})
    data.to_csv(os.path.join(save_dir, 'GATv2+AT.txt'), sep=',', index=False)

    f_pos_list, f_neg_list = compute_fidelity_attacked(GAT_AT, adj, features, adj_delta, feats_, label, test_idx,
                                                       new_att)
    logging.info(f"fidelity_pos_after_attack: {f_pos_list}")
    logging.info(f"fidelity_neg_after_attack: {f_neg_list}")
    data = pd.DataFrame({'fidelity_pos': f_pos_list, 'fidelity_neg': f_neg_list})
    data.to_csv(os.path.join(save_dir, 'GATv2+AT_after_attack.txt'), sep=',', index=False)
