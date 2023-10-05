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


def evaluate_node_level(model, features, adj, label, test_idx, roc_auc=True):
    model.eval()
    with torch.no_grad():
        orig_outputs, orig_graph_repr, orig_att = model(features, adj)
        pred = torch.argmax(orig_outputs[test_idx], dim=1)
        accuracy = accuracy_score(label[test_idx].cpu(), pred.cpu())
        f1 = f1_score(label[test_idx].cpu(), pred.cpu(), average='micro')
        if roc_auc:
            roc_auc_ = roc_auc_score(label[test_idx].cpu(), pred.cpu())
            logging.info(f'ROC_AUC: {roc_auc_}')

    logging.info(f'Test Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f}')
    return orig_outputs, orig_graph_repr, orig_att


def evaluate_graph_level(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        pred_list, label_list = [], []
        for batched_graph, labels in test_loader:
            labels = labels.to(device)
            feats = batched_graph.ndata['attr'].to(device)

            logits, _, _ = model(feats, batched_graph.to(device))

            predicted = logits.argmax(dim=1)
            pred_list = pred_list + predicted.tolist()
            label_list = label_list + labels.tolist()

        accuracy = accuracy_score(label_list, pred_list)
        precision = precision_score(label_list, pred_list)
        recall = recall_score(label_list, pred_list)
        f1 = f1_score(label_list, pred_list)

    logging.info(f'Test Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}')


def compute_fidelity(model, adj, feats, labels, test_idx):
    model.eval()

    fidelity_pos_list, fidelity_neg_list = [], []
    for split in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        variances = torch.var(feats, dim=0)
        imp_indices = torch.argsort(variances)[-int(feats.shape[1] * split):]
        unimp_indices = torch.argsort(variances)[:int(feats.shape[1] * split)]
        feats_imp = torch.zeros_like(feats)
        feats_imp[:, imp_indices] = feats[:, imp_indices]
        feats_unimp = torch.zeros_like(feats)
        feats_unimp[:, unimp_indices] = feats[:, unimp_indices]

        outputs, _, _ = model(feats, adj)
        outputs_wo_imp, _, _ = model(feats_unimp, adj)
        outputs_wo_unimp, _, _ = model(feats_imp, adj)
        pred = torch.argmax(outputs, dim=1)[test_idx]
        pred_wo_imp = torch.argmax(outputs_wo_imp, dim=1)[test_idx]
        pred_wo_unimp = torch.argmax(outputs_wo_unimp, dim=1)[test_idx]
        labels_test = labels[test_idx]

        corr_idx = torch.where(pred == labels_test)[0]
        fidelity_pos = torch.sum(pred_wo_unimp[corr_idx] == labels_test[corr_idx]) / len(corr_idx)
        fidelity_neg = torch.sum(pred_wo_imp[corr_idx] == labels_test[corr_idx]) / len(corr_idx)
        fidelity_pos_list.append(round(fidelity_pos.item(), 4))
        fidelity_neg_list.append(round(fidelity_neg.item(), 4))

    return fidelity_pos_list, fidelity_neg_list
