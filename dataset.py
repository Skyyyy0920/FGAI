import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from dgl.data import PubmedGraphDataset, PPIDataset, RedditDataset, YelpDataset  # Node Classification
from dgl.data import GINDataset  # Graph Classification PROTEINS, MUTAG, IMDBBINARY, IMDBMULTI


def load_dataset(args):
    print(f"{args.dataset} info: ")
    if args.task == 'node-level':
        if args.dataset == 'PubmedGraphDataset':
            dataset = PubmedGraphDataset()
        elif args.dataset == 'PPIDataset':
            dataset = PPIDataset()
        elif args.dataset == 'RedditDataset':
            dataset = RedditDataset()
        elif args.dataset == 'YelpDataset':
            dataset = YelpDataset()
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

    return dataset
