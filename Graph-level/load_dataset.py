from torch.utils.data import Dataset, DataLoader
from dgl.data import GINDataset
from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl
from utils import get_idx_split


def load_dataset(args):
    print(f"{args.dataset} info: ")
    if args.dataset in ['MUTAG', 'PROTEINS', 'IMDBBINARY', 'IMDBMULTI']:
        dataset = GINDataset(args.dataset, False)
        in_feats = dataset[0][0].ndata['attr'].shape[1]
        num_classes = dataset.num_classes
        split_idx = get_idx_split(len(dataset[0]))
        train_dataset, train_labels = zip(*[dataset[i] for i in split_idx["train"]])
        sss = dataset[split_idx["valid"]]
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_dgl)
        for ss in train_loader:
            s, a = ss
            print(s)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False, collate_fn=collate_dgl)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False, collate_fn=collate_dgl)
    elif args.dataset in ['ogbg-ppa', 'ogbg-molhiv']:
        dataset = DglGraphPropPredDataset(name=args.dataset)
        in_feats = dataset[0][0].ndata['feat'].shape[1]
        num_classes = dataset.num_classes
        split_idx = dataset.get_idx_split()
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True, collate_fn=collate_dgl)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False, collate_fn=collate_dgl)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False, collate_fn=collate_dgl)
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset}")

    return train_loader, valid_loader, test_loader, in_feats, num_classes
