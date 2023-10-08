import matplotlib.pyplot as plt
import networkx as nx
import argparse
import torch
import logging
import scipy.sparse as sp
from dgl.data import *
import yaml
from ogb.nodeproppred import DglNodePropPredDataset
from supplement_dataset import *
from graphgallery.datasets import NPZDataset
from models import GATNodeClassifier
import dgl
import yaml
import time
import zipfile
import argparse
import pandas as pd
from pathlib import Path
from models import GATNodeClassifier
from utils import *
from trainer import FGAITrainer
from load_dataset import load_dataset
from attackers import PGD
import torch.optim as optim


def load_dataset(args):
    if args.dataset in ['cora', 'pubmed', 'citeseer']:
        if args.dataset == 'cora':
            dataset = CoraGraphDataset()
        elif args.dataset == 'pubmed':
            dataset = PubmedGraphDataset()
        else:
            dataset = CiteseerGraphDataset()
        g = dataset[0]
        print(f"Total edges before adding self-loop {g.number_of_edges()}")
        g = g.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {g.number_of_edges()}")
        num_classes = dataset.num_classes
        train_idx, valid_idx, test_idx = g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']
        feats, label = g.ndata["feat"], g.ndata['label']
        src, dst = g.edges()
        num_nodes = g.number_of_nodes()
        adj = sp.csr_matrix((np.ones(len(src)), (src.cpu().numpy(), dst.cpu().numpy())), shape=(num_nodes, num_nodes))
    elif args.dataset in ['questions', 'amazon-ratings', 'roman-empire']:
        if args.dataset == 'questions':
            dataset = QuestionsDataset()
        elif args.dataset == 'amazon-ratings':
            dataset = AmazonRatingsDataset()
        else:
            dataset = RomanEmpireDataset()
        g = dataset[0]
        print(f"Total edges before adding self-loop {g.number_of_edges()}")
        g = g.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {g.number_of_edges()}")
        num_classes = dataset.num_classes
        tra_idx, val_idx, test_idx = g.ndata['train_mask'][:, 0], g.ndata['val_mask'][:, 0], g.ndata['test_mask'][:, 0]
        train_idx, valid_idx, test_idx = tra_idx.squeeze(), val_idx.squeeze(), test_idx.squeeze()
        feats, label = g.ndata["feat"], g.ndata['label']
        src, dst = g.edges()
        num_nodes = g.number_of_nodes()
        adj = sp.csr_matrix((np.ones(len(src)), (src.cpu().numpy(), dst.cpu().numpy())), shape=(num_nodes, num_nodes))
    elif args.dataset in ['amazon_photo', 'amazon_cs', 'coauthor_cs', 'coauthor_phy']:
        dataset = NPZDataset(args.dataset, root="./dataset/", verbose=False)
        g = dataset.graph
        splits = dataset.split_nodes()
        train_idx, valid_idx, test_idx = splits.train_nodes, splits.val_nodes, splits.test_nodes
        train_idx, valid_idx, test_idx = torch.tensor(train_idx), torch.tensor(valid_idx), torch.tensor(test_idx)
        feats, label = torch.tensor(g.x), torch.tensor(g.y, dtype=torch.int64)
        num_classes = g.num_classes
        adj = g.adj_matrix
        adj = adj + adj.transpose()
        num_nodes = g.num_nodes
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset}")

    logging.info(f"num_nodes: {num_nodes}")
    return adj, feats.to(args.device), label.to(args.device), train_idx, valid_idx, test_idx, num_classes, g


# dataset ='ogbn-arxiv'
# dataset='ogbn-products'
# dataset='ogbn-papers100M'
# dataset='questions'
# dataset='amazon-ratings'
# dataset='roman-empire'
# dataset = 'pubmed'
dataset = 'amazon_photo'
# dataset = 'amazon_cs'
# dataset='coauthor_cs'
# dataset = 'coauthor_phy'

exp = 'GAT'
with open(f"./{exp}/optimized_hyperparameter_configurations/FGAI_{dataset}.yml", 'r') as file:
    args = yaml.safe_load(file)
args = argparse.Namespace(**args)
args.device = 'cpu'

adj, feats, label, train_idx, valid_idx, test_idx, num_classes, g = load_dataset(args)
in_feats = feats.shape[1]
g = dgl.from_scipy(adj).to(args.device)

vanilla_model = GATNodeClassifier(in_feats=in_feats,
                                  hid_dim=args.hid_dim,
                                  n_classes=num_classes,
                                  n_layers=args.n_layers,
                                  n_heads=args.n_heads,
                                  feat_drop=args.feat_drop,
                                  attn_drop=args.attn_drop).to(args.device)
FGAI = GATNodeClassifier(in_feats=in_feats,
                         hid_dim=args.hid_dim,
                         n_classes=num_classes,
                         n_layers=args.n_layers,
                         n_heads=args.n_heads,
                         feat_drop=args.feat_drop,
                         attn_drop=args.attn_drop).to(args.device)

tim1 = '_10-54'
tim2 = '_10-06_11-18'

vanilla_model.load_state_dict(torch.load(f'./{exp}/GAT_checkpoints/{dataset}{tim1}/model_parameters.pth'))
FGAI.load_state_dict(torch.load(f'./{exp}/FGAI_checkpoints/{dataset}{tim2}/FGAI_parameters.pth'))

orig_outputs, orig_graph_repr, orig_att = \
    evaluate_node_level(vanilla_model, feats, adj, label, test_idx, num_classes == 2)
pred = torch.argmax(orig_outputs[test_idx], dim=1)
accuracy = accuracy_score(label[test_idx].cpu(), pred.cpu())
print(f"vanilla accuracy: {accuracy:.4f}")
FGAI_outputs, FGAI_graph_repr, FGAI_att = \
    evaluate_node_level(FGAI, feats, adj, label, test_idx, num_classes == 2)
pred = torch.argmax(FGAI_outputs[test_idx], dim=1)
accuracy = accuracy_score(label[test_idx].cpu(), pred.cpu())
print(f"FGAI accuracy: {accuracy:.4f}")

g.edata['a'] = orig_att
g.ndata['feat'] = feats[:, 0]
nx_g = dgl.to_networkx(g, node_attrs=['feat'], edge_attrs=['a'])
print(nx_g)
pos = nx.spring_layout(nx_g)  # Seed layout for reproducibility
colors = range(200)
edge_colors = [d['attention'] for u, v, d in nx_g.edges(data=True)]
cmap = plt.cm.jet  # 使用jet颜色映射
norm = plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors))
edge_colors = [cmap(norm(d['attention'])) for u, v, d in nx_g.edges(data=True)]

nx.draw(nx_g, pos, node_color='lightblue', node_size=feats.shape[0], edge_color=edge_colors, width=2.0, edge_cmap=cmap)
plt.show()

# 显示颜色条
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, label='Attention Value')

plt.show()
