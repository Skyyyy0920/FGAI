import yaml
import argparse
import pandas as pd

from models import *
from trainer import *
from attackers import *

import os.path as osp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import negative_sampling
import time
import scipy.sparse as sp
import torch
import torch.nn as nn

import torch.optim as optim
from deeprobust.graph.defense.pgd import PGD, prox_operators

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == '__main__':
    # ==================================================================================================
    # 1. Choose the dataset, base model
    # ==================================================================================================
    dataset = 'cora'
    # dataset = 'pubmed'
    # dataset = 'citeseer'

    base_model = 'GAT'
    # base_model = 'GATv2'
    # base_model = 'GT'

    # ==================================================================================================
    # 2. Get experiment args and seed
    # ==================================================================================================
    current_dir = os.getcwd()
    print("Current work dir：", current_dir)
    new_dir = current_dir + "/Link Prediction"
    os.chdir(new_dir)
    with open(f"./hyperparameter_configurations/{base_model}/{dataset}.yml", 'r') as file:
        args = yaml.full_load(file)
    args = argparse.Namespace(**args)
    args.device = device
    logging_time = time.strftime('%H-%M', time.localtime())
    save_dir = os.path.join("checkpoints", f"{base_model}+vanilla", f"{dataset}_{logging_time}")
    logging_config(save_dir)
    logging.info(f"args: {args}")
    logging.info(f"Saving path: {save_dir}")
    logging.info(f"base model: {base_model}")

    # ==================================================================================================
    # 3. Prepare data
    # ==================================================================================================
    g, adj, features, label, train_idx, valid_idx, test_idx, num_classes = load_dataset(args)
    idx_split = train_idx, valid_idx, test_idx


    adj_coo = adj.tocoo()
    row_indices = adj_coo.row
    col_indices = adj_coo.col
    edges = torch.tensor([row_indices, col_indices], dtype=torch.long)

    train_edges, temp_edges = train_test_split(edges.T, test_size=0.2, random_state=42)
    valid_edges, test_edges = train_test_split(temp_edges, test_size=0.75, random_state=42)
    train_edges = train_edges.T
    valid_edges = valid_edges.T
    test_edges = test_edges.T

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          add_negative_train_samples=False),
    ])
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
    dataset = Planetoid(path, name='Cora', transform=transform)
    train_data, val_data, test_data = dataset[0]

    in_feats = dataset.num_features
    pos_enc_size = 8
    out_size = 64

    # ==================================================================================================
    # 4. Build models, define overall loss and optimizer
    # ==================================================================================================
    if base_model == 'GAT':
        model = GATLinkPredictor(
            feats_size=in_feats,
            hidden_size=args.hid_dim,
            out_size=out_size,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            feat_drop=args.feat_drop,
            attn_drop=args.attn_drop,
            layer_norm=False
        ).to(device)

    elif base_model == 'GATv2':
        model = GATv2NodeClassifier(
            feats_size=in_feats,
            hidden_size=args.hid_dim,
            out_size=out_size,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            feat_drop=args.feat_drop,
            attn_drop=args.attn_drop,
            layer_norm=False
        ).to(device)

    elif base_model == 'GT':
        model = GTNodeClassifier(
            feats_size=features.shape[1],
            hidden_size=args.hid_dim,
            out_size=out_size,
            pos_enc_size=pos_enc_size,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            layer_norm=False
        ).to(device)
        pos_enc_path = f"./GT_pos_encoding/{dataset}_pos_enc.pth"
        if os.path.exists(pos_enc_path):
            pos_enc = torch.load(pos_enc_path)
        else:
            in_degrees = torch.tensor(adj.sum(axis=0)).squeeze()
            pos_enc = laplacian_pe(adj, in_degrees, k=pos_enc_size, padding=True).to(device)
            torch.save(pos_enc, pos_enc_path)
        model.pos_enc = pos_enc
        pos_enc_per_path = f"./GT_pos_encoding/{dataset}_pos_enc_perturbed.pth"
        if os.path.exists(pos_enc_per_path):
            model.pos_enc_ = torch.load(pos_enc_per_path)
            need_update = False
        else:
            need_update = True

    else:
        raise ValueError(f"Unknown base model name: {base_model}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params}")
    logging.info(f"Model: {model}")
    logging.info(f"Optimizer: {optimizer}")


    def train(features, adj, train_edges, valid_edges, ):
        model.train()
        optimizer.zero_grad()
        z, _, _ = model.encode(features, train_edges)

        # We perform a new round of negative sampling for every training epoch:
        neg_edges = negative_sampling(
            edge_index=edges, num_nodes=len(features),
            num_neg_samples=train_edges.size(1), method='sparse')

        edge_label_index = torch.cat(
            [train_edges, neg_edges],
            dim=-1,
        )
        edge_label = torch.cat([
            torch.ones(train_edges.size(1)),
            torch.zeros(train_edges.size(1))
        ], dim=0).to(features.device)

        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()
        return loss


    @torch.no_grad()
    def test(features, adj, test_edges, ):
        model.eval()
        z, _, _ = model.encode(features, test_edges)
        out = model.decode(z, test_edges).view(-1).sigmoid()
        # precision_k = precision_at_k(data.edge_label.cpu().numpy(), out.cpu().numpy(), k=10)
        # print("@10:", precision_k)
        return roc_auc_score(torch.ones(test_edges.size(1)).numpy(), out.cpu().numpy())


    def train():
        model.train()
        optimizer.zero_grad()
        z, _, _ = model.encode(train_data.x, train_data.edge_index)

        # We perform a new round of negative sampling for every training epoch:
        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

        edge_label_index = torch.cat(
            [train_data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()
        return loss


    @torch.no_grad()
    def test(data):
        model.eval()
        z, _, _ = model.encode(data.x, data.edge_index)
        out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
        # precision_k = precision_at_k(data.edge_label.cpu().numpy(), out.cpu().numpy(), k=10)
        # print("@10:", precision_k)
        return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


    best_val_auc = final_test_auc = 0
    for epoch in range(1, 101):
        # loss = train(features, adj, train_edges, valid_edges, )
        loss = train()
        # val_auc = test(features, adj, valid_edges)
        # test_auc = test(features, adj, test_edges)
        val_auc = test(val_data)
        test_auc = test(test_data)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, Test: {test_auc:.4f}')

    print(f'Final Test: {final_test_auc:.4f}')

    z = model.encode(test_data.x, test_data.edge_index)
    final_edge_index = model.decode_all(z)

    # ==================================================================================================
    # 5. Training
    # ==================================================================================================
    trainer = VanillaTrainer(model, criterion, optimizer, args)
    trainer.train(features, adj, label, idx_split)

    orig_outputs, _, orig_att = evaluate_node_level(model, features, adj, label, test_idx)
    torch.save(model.state_dict(), os.path.join(save_dir, 'model_parameters.pth'))

    attacker = PGD(
        epsilon=args.epsilon,
        n_epoch=args.n_epoch_attack,
        n_inject_max=args.n_inject_max,
        n_edge_max=args.n_edge_max,
        feat_lim_min=-1,
        feat_lim_max=1,
        device=device
    )
    adj_delta, feats_delta = attacker.attack(model, adj, features, test_idx, None)
    sp.save_npz(os.path.join(save_dir, 'adj_delta.npz'), adj_delta)
    torch.save(feats_delta, os.path.join(save_dir, 'feats_delta.pth'))

    if base_model == 'GT' and need_update:
        in_degrees = torch.tensor(adj_delta.sum(axis=0)).squeeze()
        pos_enc_ = laplacian_pe(adj_delta, in_degrees, k=pos_enc_size, padding=True).to(device)
        torch.save(pos_enc_, f'./{dataset}_pos_enc_perturbed.pth')
        model.pos_enc_ = pos_enc_

    feats_ = torch.cat((features, feats_delta), dim=0)
    new_outputs, _, new_att = model(feats_, adj_delta)
    new_outputs, new_att = new_outputs[:orig_outputs.shape[0]], new_att[:orig_att.shape[0]]
    pred = torch.argmax(new_outputs[test_idx], dim=1)
    accuracy = accuracy_score(label[test_idx].cpu(), pred.cpu())
    logging.info(f"Accuracy after Injection Attack: {accuracy:.4f}")

    TVD_score = TVD(orig_outputs, new_outputs) / len(orig_outputs)
    JSD_score = JSD(orig_att, new_att) / len(orig_att)
    logging.info(f"JSD: {JSD_score}")
    logging.info(f"TVD: {TVD_score}")

    f_pos_list, f_neg_list = compute_fidelity(model, adj, features, label, test_idx, orig_att)
    logging.info(f"fidelity_pos: {f_pos_list}")
    logging.info(f"fidelity_neg: {f_neg_list}")
    data = pd.DataFrame({'fidelity_pos': f_pos_list, 'fidelity_neg': f_neg_list})
    data.to_csv(os.path.join(save_dir, f'{base_model}_F.txt'), sep=',', index=False)

    f_pos_list, f_neg_list = compute_fidelity_attacked(model, adj, features, adj_delta, feats_, label, test_idx,
                                                       new_att)
    logging.info(f"fidelity_pos_after_attack: {f_pos_list}")
    logging.info(f"fidelity_neg_after_attack: {f_neg_list}")
    data = pd.DataFrame({'fidelity_pos': f_pos_list, 'fidelity_neg': f_neg_list})
    data.to_csv(os.path.join(save_dir, f'{base_model}_F_after_attack.txt'), sep=',', index=False)

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
    adj_delta_, feats_delta_ = attacker_.attack(model, adj, features, test_idx, None)
    sp.save_npz(os.path.join(save_dir, 'adj_delta_.npz'), adj_delta_)
    torch.save(feats_delta_, os.path.join(save_dir, 'feats_delta_.pth'))

    new_outputs_, new_graph_repr, new_att_ = model(feats_delta_, adj_delta_)
    pred = torch.argmax(new_outputs_[test_idx], dim=1)
    accuracy = accuracy_score(label[test_idx].cpu(), pred.cpu())
    logging.info(f"Accuracy after Modification Attack: {accuracy:.4f}")

    TVD_score = TVD(orig_outputs, new_outputs_) / len(orig_outputs)
    JSD_score = JSD(orig_att, new_att_) / len(orig_att)
    logging.info(f"JSD: {JSD_score}")
    logging.info(f"TVD: {TVD_score}")

    f_pos_list, f_neg_list = compute_fidelity_attacked(model, adj, features, adj_delta_, feats_delta_,
                                                       label, test_idx, new_att)
    logging.info(f"fidelity_pos_after_attack: {f_pos_list}")
    logging.info(f"fidelity_neg_after_attack: {f_neg_list}")
    data = pd.DataFrame({'fidelity_pos': f_pos_list, 'fidelity_neg': f_neg_list})
    data.to_csv(os.path.join(save_dir, f'{base_model}_F_after_attack_.txt'), sep=',', index=False)
