import time
import yaml
import argparse
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from utils import *
from models import GTNodeClassifier
from load_dataset import load_dataset
from trainer import VanillaTrainer
from attackers import PGD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

if __name__ == '__main__':
    # dataset = 'amazon_photo'
    # dataset = 'amazon_cs'
    # dataset = 'coauthor_phy'
    dataset = 'coauthor_cs'
    # dataset = 'pubmed'
    # dataset = 'ogbn-arxiv'

    with open(f"./optimized_hyperparameter_configurations/{dataset}.yml", 'r') as file:
        args = yaml.full_load(file)
    args = argparse.Namespace(**args)
    args.device = device
    logging_time = time.strftime('%H-%M', time.localtime())
    save_dir = os.path.join("LN_checkpoints", f"{dataset}_{logging_time}")
    logging_config(save_dir)
    logging.info(f"Using device: {device}")
    logging.info(f"args: {args}")
    logging.info(f"Saving path: {save_dir}")

    g, adj, features, label, train_idx, valid_idx, test_idx, num_classes = load_dataset(args)
    N = len(features)
    pos_enc_size = 8
    args.hid_dim = 80

    criterion = nn.CrossEntropyLoss()
    GT_LN = GTNodeClassifier(
        feats_size=features.shape[1],
        hidden_size=args.hid_dim,
        out_size=num_classes,
        pos_enc_size=pos_enc_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        LayerNorm=True
    ).to(device)
    optimizer = optim.Adam(
        GT_LN.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)

    total_params = sum(p.numel() for p in GT_LN.parameters())
    logging.info(f"Total parameters: {total_params}")
    logging.info(f"Model: {GT_LN}")
    logging.info(f"Optimizer: {optimizer}")

    pos_enc_path = f'./{dataset}_pos_enc.pth'
    if os.path.exists(pos_enc_path):
        pos_enc = torch.load(pos_enc_path)
    else:
        in_degrees = torch.tensor(adj.sum(axis=0)).squeeze()
        pos_enc = laplacian_pe(adj, in_degrees, k=pos_enc_size, padding=True).to(device)
        torch.save(pos_enc, pos_enc_path)
    GT_LN.pos_enc = pos_enc

    if os.path.exists(f'./{dataset}_pos_enc_perturbed.pth'):
        GT_LN.pos_enc_ = torch.load(f'./{dataset}_pos_enc_perturbed.pth')
        need_update = False
    else:
        need_update = True

    std_trainer = VanillaTrainer(GT_LN, criterion, optimizer, args)
    std_trainer.train(features, adj, label, train_idx, valid_idx)

    orig_outputs, orig_graph_repr, orig_att = \
        evaluate_node_level(GT_LN, features, adj, label, test_idx, num_classes == 2)

    torch.save(GT_LN.state_dict(), os.path.join(save_dir, 'model_parameters.pth'))

    attacker = PGD(
        epsilon=args.epsilon,
        n_epoch=args.n_epoch_attack,
        n_inject_max=args.n_inject_max,
        n_edge_max=args.n_edge_max,
        feat_lim_min=-1,
        feat_lim_max=1,
        device=device)

    GT_LN.eval()
    adj_delta, feats_delta = attacker.attack(GT_LN, adj, features, test_idx, None)
    new_outputs, new_graph_repr, new_att = GT_LN(torch.cat((features, feats_delta), dim=0), adj_delta)
    new_outputs, new_graph_repr, new_att = \
        new_outputs[:orig_outputs.shape[0]], new_graph_repr[:orig_graph_repr.shape[0]], new_att[:orig_att.shape[0]]
    pred = torch.argmax(new_outputs[test_idx], dim=1)
    accuracy = accuracy_score(label[test_idx].cpu(), pred.cpu())
    logging.info(f"Accuracy after attack: {accuracy:.4f}")

    TVD_score = TVD(orig_outputs, new_outputs) / len(orig_outputs)
    JSD_score = JSD(orig_att, new_att) / len(orig_att)
    logging.info(f"JSD: {JSD_score}")
    logging.info(f"TVD: {TVD_score}")

    sp.save_npz(os.path.join(save_dir, 'adj_delta.npz'), adj_delta)
    torch.save(feats_delta, os.path.join(save_dir, 'feats_delta.pth'))

    avg_att = torch.mean(orig_att, dim=1)
    fidelity_pos_list, fidelity_neg_list = compute_fidelity(GT_LN, adj, features, label, test_idx, avg_att)
    logging.info(f"fidelity_pos: {fidelity_pos_list}")
    logging.info(f"fidelity_neg: {fidelity_neg_list}")
    data = pd.DataFrame({'fidelity_pos': fidelity_pos_list, 'fidelity_neg': fidelity_neg_list})
    data.to_csv(os.path.join(save_dir, 'fidelity_data.txt'), sep=',', index=False)
