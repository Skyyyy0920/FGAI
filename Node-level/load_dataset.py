from dgl.data import YelpDataset, RedditDataset, CoraGraphDataset, PubmedGraphDataset, CiteseerGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset


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
    elif args.dataset in ['cora', 'pubmed', 'citeseer']:
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
        label = g.ndata['label']
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset}")

    return g.to(args.device), label.to(args.device), train_idx, valid_idx, test_idx, num_classes
