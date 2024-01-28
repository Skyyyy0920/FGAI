import os
import numpy as np
from dgl.convert import graph
from dgl.transforms.functional import to_bidirected
from dgl.data.dgl_dataset import DGLBuiltinDataset
from dgl.data.utils import download


class HeterophilousGraphDataset(DGLBuiltinDataset):
    def __init__(
            self,
            name,
            raw_dir=None,
            force_reload=False,
            verbose=True,
            transform=None,
    ):
        name = name.lower().replace("-", "_")
        url = f"https://github.com/yandex-research/heterophilous-graphs/raw/main/data/{name}.npz"
        super(HeterophilousGraphDataset, self).__init__(
            name=name,
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def download(self):
        download(
            url=self.url, path=os.path.join(self.raw_path, f"{self.name}.npz")
        )

    def process(self):
        """Load and process the data."""
        try:
            import torch
        except ImportError:
            raise ModuleNotFoundError(
                "This dataset requires PyTorch to be the backend."
            )

        data = np.load(os.path.join(self.raw_path, f"{self.name}.npz"))
        src = torch.from_numpy(data["edges"][:, 0])
        dst = torch.from_numpy(data["edges"][:, 1])
        features = torch.from_numpy(data["node_features"])
        labels = torch.from_numpy(data["node_labels"])
        train_masks = torch.from_numpy(data["train_masks"].T)
        val_masks = torch.from_numpy(data["val_masks"].T)
        test_masks = torch.from_numpy(data["test_masks"].T)
        num_nodes = len(labels)
        num_classes = len(labels.unique())

        self._num_classes = num_classes

        self._g = to_bidirected(graph((src, dst), num_nodes=num_nodes))
        self._g.ndata["feat"] = features
        self._g.ndata["label"] = labels
        self._g.ndata["train_mask"] = train_masks
        self._g.ndata["val_mask"] = val_masks
        self._g.ndata["test_mask"] = test_masks

    def has_cache(self):
        return os.path.exists(self.raw_path)

    def load(self):
        self.process()

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph."
        if self._transform is None:
            return self._g
        else:
            return self._transform(self._g)

    def __len__(self):
        return 1

    @property
    def num_classes(self):
        return self._num_classes


class RomanEmpireDataset(HeterophilousGraphDataset):
    def __init__(
            self, raw_dir=None, force_reload=False, verbose=True, transform=None
    ):
        super(RomanEmpireDataset, self).__init__(
            name="roman-empire",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )


class AmazonRatingsDataset(HeterophilousGraphDataset):
    def __init__(
            self, raw_dir=None, force_reload=False, verbose=True, transform=None
    ):
        super(AmazonRatingsDataset, self).__init__(
            name="amazon-ratings",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )


class MinesweeperDataset(HeterophilousGraphDataset):
    def __init__(
            self, raw_dir=None, force_reload=False, verbose=True, transform=None
    ):
        super(MinesweeperDataset, self).__init__(
            name="minesweeper",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )


class TolokersDataset(HeterophilousGraphDataset):
    def __init__(
            self, raw_dir=None, force_reload=False, verbose=True, transform=None
    ):
        super(TolokersDataset, self).__init__(
            name="tolokers",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )


class QuestionsDataset(HeterophilousGraphDataset):
    def __init__(
            self, raw_dir=None, force_reload=False, verbose=True, transform=None
    ):
        super(QuestionsDataset, self).__init__(
            name="questions",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )
