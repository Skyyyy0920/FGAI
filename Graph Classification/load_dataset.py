from dgl.data import GINDataset, QM7bDataset, LegacyTUDataset
from ogb.graphproppred import DglGraphPropPredDataset
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import split_dataset


def load_dataset(args):
    print(f"{args.dataset} info: ")
    if args.dataset in ['MUTAG', 'PROTEINS', 'IMDBBINARY', 'IMDBMULTI']:
        dataset = GINDataset(args.dataset, self_loop=True, degree_as_nlabel=False)
        in_feats = dataset.dim_nfeats
        num_classes = dataset.num_classes
        train_dataset, valid_dataset, test_dataset = split_dataset(dataset, frac_list=[0.6, 0.02, 0.38], shuffle=True)

    elif args.dataset in ['ogbg-ppa', 'ogbg-molhiv']:
        dataset = DglGraphPropPredDataset(name=args.dataset)
        in_feats = dataset[0][0].ndata['feat'].shape[1]
        num_classes = dataset.num_classes
        split_idx = dataset.get_idx_split()
        train_dataset, valid_dataset, test_dataset = \
            dataset[split_idx["train"]], dataset[split_idx["valid"]], dataset[split_idx["test"]]

    elif args.dataset in ['DD', 'ENZYMES', 'COLLAB']:
        dataset = LegacyTUDataset(args.dataset)
        num_classes = dataset.num_classes
        in_feats = dataset[0][0].ndata['feat'].shape[1]
        train_dataset, valid_dataset, test_dataset = split_dataset(dataset, frac_list=[0.6, 0.02, 0.38], shuffle=True)

    else:
        raise ValueError(f"Unknown dataset name: {args.dataset}")

    train_loader = GraphDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    valid_loader = GraphDataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = GraphDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    print(f"Classes: {num_classes}")
    print(f"num_graphs train: {len(train_dataset)}")

    return train_loader, valid_loader, test_loader, in_feats, int(num_classes)
