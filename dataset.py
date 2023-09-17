from torch.utils.data import Dataset, DataLoader
from dgl.data import YelpDataset, RedditDataset  # Node Classification
from dgl.data import GINDataset  # Graph Classification
from ogb.nodeproppred import DglNodePropPredDataset


def load_dataset(args):
    print(f"{args.dataset} info: ")
    if args.task == 'node-level':
        if args.dataset == 'ogbn-arxiv':
            dataset = DglNodePropPredDataset(name='ogbn-arxiv')
            g, label = dataset[0]
            label = label.squeeze()
            srcs, dsts = g.all_edges()
            g.add_edges(dsts, srcs)
            print(f"Total edges before adding self-loop {g.number_of_edges()}")
            g = g.remove_self_loop().add_self_loop()
            print(f"Total edges after adding self-loop {g.number_of_edges()}")
            split_idx = dataset.get_idx_split()
            train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
            num_classes = dataset.num_classes
        else:
            raise ValueError(f"Unknown dataset name: {args.dataset}")
    elif args.task == 'graph-level':
        if args.dataset == 'PROTEINS':
            dataset = GINDataset('PROTEINS', False)
        elif args.dataset == 'MUTAG':
            dataset = GINDataset('MUTAG', False)
        elif args.dataset == 'IMDBBINARY':
            dataset = GINDataset('IMDBBINARY', False)
        elif args.dataset == 'IMDBMULTI':
            dataset = GINDataset('IMDBMULTI', False)
        else:
            raise ValueError(f"Unknown dataset name: {args.dataset}")
    else:
        raise ValueError(f"Unknown task: {args.task}")

    return g.to(args.device), label.to(args.device), train_idx, valid_idx, test_idx, num_classes
