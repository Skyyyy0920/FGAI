import yaml
import argparse
import pandas as pd

from models import *
from trainer import *
from attackers import *
from load_dataset import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == '__main__':
    # ==================================================================================================
    # 1. Choose the dataset, base model
    # ==================================================================================================
    # dataset = 'amazon_photo'
    dataset = 'amazon_cs'
    # dataset = 'coauthor_cs'
    # dataset = 'coauthor_phy'
    # dataset = 'pubmed'
    # dataset = 'ogbn-arxiv'

    base_model = 'GAT'
    # base_model = 'GATv2'
    # base_model = 'GT'

    # ==================================================================================================
    # 2. Get experiment args and seed
    # ==================================================================================================
    with open(f"./optimized_hyperparameter_configurations/{base_model}/{dataset}.yml", 'r') as file:
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
    in_feats = features.shape[1]
    pos_enc_size = 8

    # ==================================================================================================
    # 4. Build models, define overall loss and optimizer
    # ==================================================================================================
    if base_model == 'GAT':
        model = GATNodeClassifier(
            feats_size=in_feats,
            hidden_size=args.hid_dim,
            out_size=num_classes,
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
            out_size=num_classes,
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
            out_size=num_classes,
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

    import os.path as osp

    import torch
    from sklearn.metrics import roc_auc_score

    import torch_geometric.transforms as T
    from torch_geometric.datasets import Planetoid
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import negative_sampling

    import dgl
    import time
    import numpy as np
    import scipy.sparse as sp
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from copy import deepcopy
    import dgl.function as fn
    from dgl.nn import GATv2Conv
    from dgl.base import DGLError
    from dgl.ops import edge_softmax
    from dgl.utils import expand_as_pair
    from dgl.nn.pytorch.utils import Identity
    import dgl.sparse as dglsp
    import torch.optim as optim
    from deeprobust.graph.utils import accuracy
    from deeprobust.graph.defense.pgd import PGD, prox_operators

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          add_negative_train_samples=False),
    ])
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
    dataset = Planetoid(path, name='Cora', transform=transform)
    # After applying the `RandomLinkSplit` transform, the data is transformed from
    # a data object to a list of tuples (train_data, val_data, test_data), with
    # each element representing the corresponding split.
    train_data, val_data, test_data = dataset[0]


    def precision_at_k(logits, labels, k):
        # 按照logits排序，选出得分最高的前K个
        indices = np.argsort(logits)[::-1][:k]  # 降序排列
        selected_labels = np.array(labels)[indices]

        # 计算 Precision@K
        precision = np.sum(selected_labels) / k
        return precision


    class Net(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
            # self.conv3 = GATConv(in_channels, hidden_channels, 8, 0.6, 0.6, activation=F.elu)

        def encode(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            return self.conv2(x, edge_index)

        def decode(self, z, edge_label_index):
            return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

        def decode_all(self, z):
            prob_adj = z @ z.t()
            return (prob_adj > 0).nonzero(as_tuple=False).t()


    model = Net(dataset.num_features, 128, 64).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()


    def train():
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)

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
        z = model.encode(data.x, data.edge_index)
        out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
        # precision_k = precision_at_k(data.edge_label.cpu().numpy(), out.cpu().numpy(), k=10)
        # print("@10:", precision_k)
        return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


    best_val_auc = final_test_auc = 0
    for epoch in range(1, 101):
        loss = train()
        val_auc = test(val_data)
        test_auc = test(test_data)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
              f'Test: {test_auc:.4f}')

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
