import torch
import logging
import scipy.sparse as sp
from dgl.data import *
from ogb.nodeproppred import DglNodePropPredDataset
from supplement_dataset import *
from graphgallery.datasets import NPZDataset


def load_dataset(args):
    print(f"{args.dataset} info: ")
    if args.dataset in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']:
        dataset = DglNodePropPredDataset(args.dataset)
        g, label = dataset[0]
        label = label.squeeze()
        if args.dataset in ['ogbn-arxiv', 'ogbn-papers100M']:
            srcs, dsts = g.all_edges()
            g.add_edges(dsts, srcs)
        print(f"Total edges before adding self-loop {g.number_of_edges()}")
        g = g.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {g.number_of_edges()}")
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        num_classes = dataset.num_classes
        feats = g.ndata["feat"]
        src, dst = g.edges()
        num_nodes = g.number_of_nodes()
        adj = sp.csr_matrix((np.ones(len(src)), (src.cpu().numpy(), dst.cpu().numpy())), shape=(num_nodes, num_nodes))
    elif args.dataset in ['cora', 'pubmed', 'citeseer']:
        if args.dataset == 'cora':
            dataset = CoraGraphDataset()
        elif args.dataset == 'pubmed':
            dataset = PubmedGraphDataset()
        elif args.dataset == 'questions':
            dataset = QuestionsDataset()
        elif args.dataset == 'amazon-ratings':
            dataset = AmazonRatingsDataset()
        elif args.dataset == 'roman-empire':
            dataset = RomanEmpireDataset()
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
    elif args.dataset in ['amazon_photo', 'amazon_cs', '']:
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

    del g
    logging.info(f"num_nodes: {num_nodes}")
    return adj, feats.to(args.device), label.to(args.device), train_idx, valid_idx, test_idx, num_classes
