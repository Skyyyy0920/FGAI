import os
import torch
import random
import logging
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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


def get_idx_split(dataset_len):
    idx = list(range(dataset_len))
    random.shuffle(idx)

    split_point_1 = int(dataset_len * 0.6)
    split_point_2 = int(dataset_len * 0.8)

    split_idx = {'train': idx[:split_point_1], 'valid': idx[split_point_1:split_point_2], 'test': idx[split_point_2:]}
    return split_idx


def k_shell_algorithm(adj_matrix):
    flag = 1
    if isinstance(adj_matrix, torch.Tensor):
        node_degrees = np.sum(adj_matrix.detach().cpu().numpy(), axis=1)
    elif isinstance(adj_matrix, sp.csr_matrix):
        node_degrees = np.array(adj_matrix.sum(axis=1)).squeeze()
        flag = 0
    else:
        raise ValueError(f"Unknown adj type: {type(adj_matrix)}")
    k_values = np.zeros(len(node_degrees), dtype=int)
    remaining_nodes = np.arange(len(node_degrees))

    k = 1
    while len(remaining_nodes) > 0:
        nodes_to_remove = np.intersect1d(remaining_nodes[node_degrees[remaining_nodes] <= k],
                                         remaining_nodes[node_degrees[remaining_nodes] >= 0])
        if len(nodes_to_remove) > 0:
            k_values[nodes_to_remove] = k
            node_degrees[nodes_to_remove] = -1  # Mark nodes as processed
            for node_id in nodes_to_remove:
                if flag:
                    node_degrees[adj_matrix[node_id] == 1] -= 1
                else:
                    neighbors = set(adj_matrix[node_id].indices)
                    for neighbor in neighbors:
                        node_degrees[neighbor] -= 1
            remaining_nodes = np.setdiff1d(remaining_nodes, nodes_to_remove)
        if all(x < 0 or x > k for x in node_degrees):
            k += 1

    return k_values - k_values.min() + 1


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
    a[a == 0] = 1e-10
    b[b == 0] = 1e-10
    loss = kl(a, b) + kl(b, a)
    loss /= 2
    return loss


def topK_overlap_loss(new_att, old_att, adj, K=2, metric='l1'):
    new_att, old_att = new_att.squeeze(), old_att.squeeze()

    loss = 0

    idx_1 = torch.argsort(new_att, dim=-1, descending=True)
    idx_1 = idx_1[:K]
    old_topK_1 = old_att.gather(-1, idx_1)
    new_topK_1 = new_att.gather(-1, idx_1)

    idx_2 = torch.argsort(old_att, dim=-1, descending=True)
    idx_2 = idx_2[:K]
    old_topK_2 = old_att.gather(-1, idx_2)
    new_topK_2 = new_att.gather(-1, idx_2)

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


def evaluate_node_level(model, criterion, features, adj, label, test_idx):
    model.eval()
    with torch.no_grad():
        test_outputs, graph_rep, _ = model(features, adj)
        test_loss = criterion(test_outputs[test_idx], label[test_idx])
        test_pred = torch.argmax(test_outputs[test_idx], dim=1)
        test_accuracy = accuracy_score(label[test_idx].cpu(), test_pred.cpu())
        test_f1_score = f1_score(label[test_idx].cpu(), test_pred.cpu(), average='micro')

    logging.info(
        f'Test Loss: {test_loss.item():.4f} | Accuracy: {test_accuracy:.4f} | F1 Score: {test_f1_score:.4f}')


def evaluate_graph_level(model, criterion, test_loader, device):
    model.eval()
    with torch.no_grad():
        loss_list = []
        pred_list, label_list = [], []
        for batched_graph, labels in test_loader:
            labels = labels.to(device)
            feats = batched_graph.ndata['attr'].to(device)

            logits, _, _ = model(feats, batched_graph.to(device))
            loss = criterion(logits, labels)
            loss_list.append(loss.item())

            predicted = logits.argmax(dim=1)
            pred_list = pred_list + predicted.tolist()
            label_list = label_list + labels.tolist()

        accuracy = accuracy_score(label_list, pred_list)
        precision = precision_score(label_list, pred_list)
        recall = recall_score(label_list, pred_list)
        f1 = f1_score(label_list, pred_list)

    logging.info(f'Test Loss: {np.mean(loss_list):.4f} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f}'
                 f' | Recall: {recall:.4f} | F1: {f1:.4f}')
