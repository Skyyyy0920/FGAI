import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from dgl.data import PubmedGraphDataset, PPIDataset, RedditDataset, YelpDataset  # Node Classification
from dgl.data import GINDataset  # Graph Classification PROTEINS, MUTAG, IMDBBINARY, IMDBMULTI


def load_dataset(args):
    print(f"{args.dataset} info: ")
    dataset = PubmedGraphDataset()

    return dataset
