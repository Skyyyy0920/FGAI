from dgl.data import GINDataset
from ogb.graphproppred import DglGraphPropPredDataset
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import split_dataset
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset


def load_dataset(args):
    print(f"{args.dataset} info: ")
    if args.dataset in ['MUTAG', 'PROTEINS', 'IMDBBINARY', 'IMDBMULTI']:
        dataset = GINDataset(args.dataset, self_loop=True, degree_as_nlabel=False)
        in_feats = dataset.dim_nfeats
        num_classes = dataset.num_classes
        train_dataset, valid_dataset, test_dataset = split_dataset(dataset)

    elif args.dataset in ['ogbg-ppa', 'ogbg-molhiv']:
        dataset = DglGraphPropPredDataset(name=args.dataset)
        in_feats = dataset[0][0].ndata['feat'].shape[1]
        num_classes = dataset.num_classes
        split_idx = dataset.get_idx_split()
        train_dataset, valid_dataset, test_dataset = \
            dataset[split_idx["train"]], dataset[split_idx["valid"]], dataset[split_idx["test"]]

    else:
        raise ValueError(f"Unknown dataset name: {args.dataset}")

    # dataset = TUDataset('./dataset/', name=args.dataset, use_node_attr=True, use_edge_attr=True)
    # n = (len(dataset) + 9) // 10
    # in_feats = dataset.num_features
    # num_classes = dataset.num_classes
    # dataset = dataset.shuffle()
    # test_dataset = dataset[:n]
    # valid_dataset = dataset[n:2 * n]
    # train_dataset = dataset[2 * n:]

    train_loader = GraphDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    valid_loader = GraphDataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = GraphDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    # valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    return train_loader, valid_loader, test_loader, in_feats, num_classes
