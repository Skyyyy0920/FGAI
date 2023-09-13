import os
import dgl
import torch
import random
import logging
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score

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


def TVD_numpy(predictions, targets):  # accepts two numpy arrays of dimension: (num. instances, )
    return (0.5 * np.abs(predictions - targets)).sum()


def TVD(predictions: torch.Tensor, targets: torch.Tensor, reduce=True):
    if not reduce:
        return 0.5 * torch.abs(predictions - targets)
    else:
        return (0.5 * torch.abs(predictions - targets)).sum()


def kl(a, b):
    return torch.nn.functional.kl_div(a.log(), b, reduction="batchmean")


def JSD(a, b):
    loss = kl(a, b) + kl(b, a)
    loss /= 2
    return loss


def topK_overlap_loss(new_att, old_att, g: dgl.DGLGraph, K=2, metric='l1'):
    new_att, old_att = new_att.squeeze(), old_att.squeeze()
    src, dst = g.edges()
    loss = 0

    for node_id in g.nodes():
        # neighbors = g.successors(node_id)
        # neighbor_ids = neighbors.numpy()
        indices = torch.where(src == node_id)[0]
        if len(indices) < K:
            continue
        old_neighbor_att = old_att[indices]  # 1维tensor, 假设邻居节点有n个, 则为(n, )的tensor
        new_neighbor_att = new_att[indices]

        idx_1 = torch.argsort(new_neighbor_att, descending=True)
        idx_1 = idx_1[:K]
        old_topK_1 = old_neighbor_att.gather(0, idx_1)
        new_topK_1 = new_neighbor_att.gather(0, idx_1)

        idx_2 = torch.argsort(old_neighbor_att, descending=True)
        idx_2 = idx_2[:K]
        old_topK_2 = old_neighbor_att.gather(0, idx_2)
        new_topK_2 = new_neighbor_att.gather(0, idx_2)

        if metric == 'l1':
            loss += (torch.norm(old_topK_1 - new_topK_1, p=1) + torch.norm(new_topK_2 - old_topK_2, p=1)) / (2 * K)
        elif metric == 'l2':
            loss += (torch.norm(old_topK_1 - new_topK_1, p=2) + torch.norm(new_topK_2 - old_topK_2, p=2)) / (2 * K)
        elif metric == "kl-full":
            loss += kl(new_att, old_att)
        elif metric == "jsd-full":
            loss += JSD(new_att, old_att)
        elif metric == "kl-topk":
            gt_Topk_1_normed = torch.nn.functional.softmax(new_topK_1, dim=-1)
            pred_TopK_1_normed = torch.nn.functional.softmax(old_topK_1, dim=-1)
            gt_TopK_2_normed = torch.nn.functional.softmax(new_topK_2, dim=-1)
            pred_TopK_2_normed = torch.nn.functional.softmax(old_topK_2, dim=-1)
            loss += (kl(gt_Topk_1_normed, pred_TopK_1_normed) + kl(gt_TopK_2_normed, pred_TopK_2_normed)) / 2
        elif metric == "jsd-topk":
            gt_Topk_1_normed = torch.nn.functional.softmax(new_topK_1, dim=-1)
            pred_TopK_1_normed = torch.nn.functional.softmax(old_topK_1, dim=-1)
            gt_TopK_2_normed = torch.nn.functional.softmax(new_topK_2, dim=-1)
            pred_TopK_2_normed = torch.nn.functional.softmax(old_topK_2, dim=-1)
            loss += (JSD(gt_Topk_1_normed, pred_TopK_1_normed) + JSD(gt_TopK_2_normed, pred_TopK_2_normed)) / 2
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return loss


def evaluate(model, criterion, g, features, test_mask, test_label):
    model.eval()
    with torch.no_grad():
        test_outputs, graph_rep, _ = model(g, features)
        test_loss = criterion(test_outputs[test_mask], test_label)
        test_pred = torch.argmax(test_outputs[test_mask], dim=1)
        test_accuracy = accuracy_score(test_label.cpu(), test_pred.cpu())

    logging.info(f'Test Loss: {test_loss.item():.4f} | Test Accuracy: {test_accuracy:.4f}')
