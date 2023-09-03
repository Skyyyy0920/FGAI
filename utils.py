import os
import re
import dgl
import glob
import torch
import random
import collections
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def zipdir(path, zipf, include_format):
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[-1] in include_format:
                filename = os.path.join(root, file)
                arcname = os.path.relpath(os.path.join(root, file), os.path.join(path, '..'))
                zipf.write(filename, arcname)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_node_degrees(adj_matrix):
    # Calculate node degrees by summing the rows of the adjacency matrix
    node_degrees = np.sum(adj_matrix.detach().cpu().numpy(), axis=1)
    return node_degrees


def k_shell_algorithm(adj_matrix):
    degrees = compute_node_degrees(adj_matrix)
    k_values = np.zeros(len(degrees), dtype=int)
    remaining_nodes = np.arange(len(degrees))

    k = 1
    while len(remaining_nodes) > 0:
        nodes_to_remove = np.intersect1d(remaining_nodes[degrees[remaining_nodes] <= k],
                                         remaining_nodes[degrees[remaining_nodes] >= 0])
        if len(nodes_to_remove) > 0:
            k_values[nodes_to_remove] = k
            degrees[nodes_to_remove] = -1  # Mark nodes as processed
            for node_id in nodes_to_remove:
                degrees[adj_matrix[node_id] == 1] -= 1
            remaining_nodes = np.setdiff1d(remaining_nodes, nodes_to_remove)
        if all(x < 0 or x > k for x in degrees):
            k += 1

    return k_values
