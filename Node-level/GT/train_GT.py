import time
import yaml
import argparse
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from utils import *
from models import GTNodeClassifier
from load_dataset import load_dataset
from attackers import PGD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # dataset = 'amazon_photo'
    dataset = 'amazon_cs'
    # dataset = 'coauthor_phy'
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
    N = len(features)
    pos_enc_size = 8
    args.hid_dim = 80

    criterion = nn.CrossEntropyLoss()
    GT = GTNodeClassifier(
        feats_size=features.shape[1],
        hidden_size=args.hid_dim,
        out_size=num_classes,
        pos_enc_size=pos_enc_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads
    ).to(device)
    optimizer = optim.Adam(
        GT.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    total_params = sum(p.numel() for p in GT.parameters())
    logging.info(f"Total parameters: {total_params}")
    logging.info(f"Model: {GT}")
    logging.info(f"Optimizer: {optimizer}")

    pos_enc = laplacian_pe(N, adj, g.degree[0], k=pos_enc_size, padding=True).to(device)
    X = features, pos_enc

    for epoch in range(args.num_epochs):
        GT.train()

        logits, graph_repr, att = GT(X, adj)
        loss = criterion(logits[train_idx], label[train_idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        GT.eval()
        with torch.no_grad():
            val_logits, val_graph_repr, val_att = GT(X, adj)
            val_loss = criterion(val_logits[valid_idx], label[valid_idx])
            val_pred = torch.argmax(val_logits[valid_idx], dim=1)
            val_accuracy = accuracy_score(label[valid_idx].cpu(), val_pred.cpu())

        logging.info(f'Epoch [{epoch + 1}/{args.num_epochs}] | Train Loss: {loss.item():.4f} | '
                     f'Val Loss: {val_loss.item():.4f} | Val Accuracy: {val_accuracy:.4f}')

    GT.eval()
    with torch.no_grad():
        orig_outputs, orig_graph_repr, orig_att = GT(X, adj)
        test_pred = torch.argmax(orig_outputs[test_idx], dim=1)
        test_accuracy = accuracy_score(label[test_idx].cpu(), test_pred.cpu())

    logging.info(f'Test Accuracy: {test_accuracy:.4f}')

    torch.save(GT.state_dict(), os.path.join(save_dir, 'model_parameters.pth'))

    attacker = PGD(
        epsilon=args.epsilon,
        n_epoch=args.n_epoch_attack,
        n_inject_max=args.n_inject_max,
        n_edge_max=args.n_edge_max,
        feat_lim_min=-1,
        feat_lim_max=1,
        device=device
    )

    GT.eval()
    adj_delta, feats_delta = attacker.attack(GT, adj, features, test_idx, None)
    new_outputs, new_graph_repr, new_att = GT(torch.cat((features, feats_delta), dim=0), adj_delta)
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

    fidelity_pos_list, fidelity_neg_list = compute_fidelity(GT, adj, features, label, test_idx)
    logging.info(f"fidelity_pos: {fidelity_pos_list}")
    logging.info(f"fidelity_neg: {fidelity_neg_list}")
    data = pd.DataFrame({'fidelity_pos': fidelity_pos_list, 'fidelity_neg': fidelity_neg_list})
    data.to_csv(os.path.join(save_dir, 'fidelity_data.txt'), sep=',', index=False)
