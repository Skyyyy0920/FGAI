import time
import yaml
import argparse
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from utils import *
from models import GATNodeClassifier
from trainer import VanillaTrainer
from attackers import PGD
from load_dataset import load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    dataset = 'amazon_photo'
    # dataset = 'amazon_cs'
    # dataset = 'coauthor_phy'
    # dataset = 'coauthor_cs'
    # dataset = 'pubmed'
    # dataset = 'ogbn-arxiv'

    with open(f"./optimized_hyperparameter_configurations/{dataset}.yml", 'r') as file:
        args = yaml.full_load(file)
    args = argparse.Namespace(**args)
    args.device = device
    logging_time = time.strftime('%H-%M', time.localtime())
    save_dir = os.path.join("vanilla_checkpoints", f"{dataset}_{logging_time}")
    logging_config(save_dir)
    logging.info(f"Using device: {device}")
    logging.info(f"args: {args}")
    logging.info(f"Saving path: {save_dir}")

    g, adj, features, label, train_idx, valid_idx, test_idx, num_classes = load_dataset(args)
    in_feats = features.shape[1]

    GAT = GATNodeClassifier(
        feats_size=in_feats,
        hidden_size=args.hid_dim,
        out_size=num_classes,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        feat_drop=args.feat_drop,
        attn_drop=args.attn_drop
    ).to(device)
    optimizer = optim.Adam(
        GAT.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in GAT.parameters())
    logging.info(f"Total parameters: {total_params}")
    logging.info(f"Model: {GAT}")
    logging.info(f"Optimizer: {optimizer}")

    std_trainer = VanillaTrainer(GAT, criterion, optimizer, args)
    std_trainer.train(features, adj, label, train_idx, valid_idx)

    orig_outputs, orig_graph_repr, orig_att = \
        evaluate_node_level(GAT, features, adj, label, test_idx, num_classes == 2)

    torch.save(GAT.state_dict(), os.path.join(save_dir, 'model_parameters.pth'))

    # args.n_inject_max = 1000
    # args.n_edge_max = 1000
    attacker = PGD(
        epsilon=args.epsilon,
        n_epoch=args.n_epoch_attack,
        n_inject_max=args.n_inject_max,
        n_edge_max=args.n_edge_max,
        feat_lim_min=-1,
        feat_lim_max=1,
        device=device
    )

    GAT.eval()
    adj_delta, feats_delta = attacker.attack(GAT, adj, features, test_idx, None)
    sp.save_npz(os.path.join(save_dir, 'adj_delta.npz'), adj_delta)
    torch.save(feats_delta, os.path.join(save_dir, 'feats_delta.pth'))

    feats_ = torch.cat((features, feats_delta), dim=0)
    new_outputs, new_graph_repr, new_att = GAT(feats_, adj_delta)
    new_outputs, new_graph_repr, new_att = \
        new_outputs[:orig_outputs.shape[0]], new_graph_repr[:orig_graph_repr.shape[0]], new_att[:orig_att.shape[0]]
    pred = torch.argmax(new_outputs[test_idx], dim=1)
    accuracy = accuracy_score(label[test_idx].cpu(), pred.cpu())
    logging.info(f"Accuracy after Injection Attack: {accuracy:.4f}")

    TVD_score = TVD(orig_outputs, new_outputs) / len(orig_outputs)
    JSD_score = JSD(orig_att, new_att) / len(orig_att)
    logging.info(f"JSD: {JSD_score}")
    logging.info(f"TVD: {TVD_score}")

    f_pos_list, f_neg_list = compute_fidelity(GAT, adj, features, label, test_idx, orig_att)
    logging.info(f"fidelity_pos: {f_pos_list}")
    logging.info(f"fidelity_neg: {f_neg_list}")
    data = pd.DataFrame({'fidelity_pos': f_pos_list, 'fidelity_neg': f_neg_list})
    data.to_csv(os.path.join(save_dir, 'GAT.txt'), sep=',', index=False)

    f_pos_list, f_neg_list = compute_fidelity_attacked(GAT, adj, features, adj_delta, feats_, label, test_idx, new_att)
    logging.info(f"fidelity_pos_after_attack: {f_pos_list}")
    logging.info(f"fidelity_neg_after_attack: {f_neg_list}")
    data = pd.DataFrame({'fidelity_pos': f_pos_list, 'fidelity_neg': f_neg_list})
    data.to_csv(os.path.join(save_dir, 'GAT_after_attack.txt'), sep=',', index=False)

    attacker_ = PGD(
        epsilon=0.00001,
        n_epoch=args.n_epoch_attack,
        n_inject_max=args.n_inject_max,
        n_edge_max=args.n_edge_max,
        feat_lim_min=-0.001,
        feat_lim_max=0.001,
        device=device,
        mode='Modification Attack'
    )
    GAT.eval()
    adj_delta, feats_delta = attacker_.attack(GAT, adj, features, test_idx, None)
    sp.save_npz(os.path.join(save_dir, 'adj_delta_.npz'), adj_delta)
    torch.save(feats_delta, os.path.join(save_dir, 'feats_delta_.pth'))

    new_outputs, new_graph_repr, new_att = GAT(feats_delta, adj_delta)
    new_outputs, new_graph_repr, new_att = \
        new_outputs[:orig_outputs.shape[0]], new_graph_repr[:orig_graph_repr.shape[0]], new_att[:orig_att.shape[0]]
    pred = torch.argmax(new_outputs[test_idx], dim=1)
    accuracy = accuracy_score(label[test_idx].cpu(), pred.cpu())
    logging.info(f"Accuracy after Modification Attack: {accuracy:.4f}")

    TVD_score = TVD(orig_outputs, new_outputs) / len(orig_outputs)
    JSD_score = JSD(orig_att, new_att) / len(orig_att)
    logging.info(f"JSD: {JSD_score}")
    logging.info(f"TVD: {TVD_score}")

    f_pos_list, f_neg_list = compute_fidelity(GAT, adj, features, label, test_idx, orig_att)
    logging.info(f"fidelity_pos: {f_pos_list}")
    logging.info(f"fidelity_neg: {f_neg_list}")
    data = pd.DataFrame({'fidelity_pos': f_pos_list, 'fidelity_neg': f_neg_list})
    data.to_csv(os.path.join(save_dir, 'GT.txt'), sep=',', index=False)

    f_pos_list, f_neg_list = compute_fidelity_attacked(GAT, adj, features, adj_delta, feats_delta, label, test_idx,
                                                       new_att)
    logging.info(f"fidelity_pos_after_attack: {f_pos_list}")
    logging.info(f"fidelity_neg_after_attack: {f_neg_list}")
    data = pd.DataFrame({'fidelity_pos': f_pos_list, 'fidelity_neg': f_neg_list})
    data.to_csv(os.path.join(save_dir, 'GT_after_attack.txt'), sep=',', index=False)
