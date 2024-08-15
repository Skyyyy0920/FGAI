import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from grb.attack.base import InjectionAttack
from grb.evaluator import metric
from grb.utils import utils
from scipy.sparse import csr_matrix

from utils import topK_overlap_loss, laplacian_pe
from models import GTNodeClassifier


class PGD(InjectionAttack):
    def __init__(self,
                 epsilon,
                 n_epoch,
                 n_inject_max,
                 n_edge_max,
                 feat_lim_min,
                 feat_lim_max,
                 loss=F.nll_loss,
                 eval_metric=metric.eval_acc,
                 K=5,
                 device='cpu',
                 verbose=True,
                 dataset='amazon_cs',
                 mode='Injection Attack'):
        self.device = device
        self.epsilon = epsilon
        self.n_epoch = n_epoch
        self.n_inject_max = n_inject_max
        self.n_edge_max = n_edge_max
        self.feat_lim_min = feat_lim_min
        self.feat_lim_max = feat_lim_max
        self.loss = loss
        self.eval_metric = eval_metric
        self.K = K
        self.verbose = verbose
        self.dataset = dataset
        self.mode = mode

    def attack(self, model, test_data, features, target_mask):
        edges = test_data.edge_index
        row_indices = edges[0].cpu().numpy()
        col_indices = edges[1].cpu().numpy()
        data = np.ones(len(row_indices))
        num_nodes = len(features)
        adj = csr_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))
        model.to(self.device)
        model.eval()
        n_total, n_feat = features.shape
        features = utils.feat_preprocess(features=features, device=self.device)
        z, _, att_orig = model.encode(features, edges)
        pred_orig = model.decode(z, test_data.edge_label_index).view(-1).sigmoid()

        if self.mode == 'Injection Attack':
            adj_attack = self.injection(adj=adj,
                                        n_inject=self.n_inject_max,
                                        n_node=n_total,
                                        target_mask=target_mask)

            features_attack = np.random.normal(loc=0, scale=self.feat_lim_max / 10, size=(self.n_inject_max, n_feat))
            features_attack = self.update_features(model=model,
                                                   adj=adj,
                                                   adj_attack=adj_attack,
                                                   features=features,
                                                   features_attack=features_attack,
                                                   pred_orig=pred_orig,
                                                   att_orig=att_orig,
                                                   target_mask=target_mask,
                                                   adj_norm_func=None,
                                                   test_data=test_data)

        elif self.mode == 'Modification Attack':
            adj_attack, features_attack = self.modification(model, adj, features, pred_orig, att_orig, target_mask)

        else:
            raise ValueError(f"Unknown attack mode: {self.mode}")

        return adj_attack, features_attack

    def injection(self, adj, n_inject, n_node, target_mask):
        test_index = torch.where(target_mask)[0].cpu()
        n_test = test_index.shape[0]
        new_edges_x = []
        new_edges_y = []
        new_data = []
        for i in range(n_inject):
            x = i + n_node
            for j in range(self.n_edge_max):
                yy = random.randint(0, n_test - 1)
                y = test_index[yy]
                new_edges_x.extend([x, y])
                new_edges_y.extend([y, x])
                new_data.extend([1, 1])

        add1 = sp.csr_matrix((n_inject, n_node))
        add2 = sp.csr_matrix((n_node + n_inject, n_inject))
        adj_attack = sp.vstack([adj, add1])
        adj_attack = sp.hstack([adj_attack, add2])

        adj_attack = adj_attack.tocoo()
        adj_attack.row = np.hstack([adj_attack.row, new_edges_x])
        adj_attack.col = np.hstack([adj_attack.col, new_edges_y])
        adj_attack.data = np.hstack([adj_attack.data, new_data])
        adj_attack = adj_attack.tocsr()

        return adj_attack

    def update_features(self, model, adj, adj_attack, features, features_attack, pred_orig, att_orig, target_mask,
                        adj_norm_func, test_data):
        feat_lim_min, feat_lim_max = self.feat_lim_min, self.feat_lim_max

        n_total = features.shape[0]
        adj_attacked_tensor = utils.adj_preprocess(adj=adj_attack,
                                                   adj_norm_func=adj_norm_func,
                                                   model_type='dgl',
                                                   device=self.device)
        features_attack = utils.feat_preprocess(features=features_attack, device=self.device)
        model.eval()

        if isinstance(model, GTNodeClassifier) and not os.path.exists(
                f'./GT_pos_encoding/{self.dataset}_pos_enc_perturbed.pth'):
            in_degrees = torch.tensor(adj_attacked_tensor.sum(axis=0)).squeeze()
            model.pos_enc_ = laplacian_pe(adj_attacked_tensor, in_degrees, padding=True).to(features.device)

        for i in range(self.n_epoch):
            features_attack.requires_grad_(True)
            features_attack.retain_grad()
            features_concat = torch.cat((features, features_attack), dim=0)
            row_indices, col_indices = adj_attacked_tensor.nonzero()
            edges_attacked = torch.tensor([row_indices, col_indices])
            z, _, att = model.encode(features_concat, edges_attacked)
            pred = model.decode(z, test_data.edge_label_index).view(-1).sigmoid()
            orig_labels = torch.argmax(pred_orig.unsqueeze(-1), dim=1)

            if self.loss == F.nll_loss:
                pred_loss = self.loss(pred, orig_labels).to(self.device)
            elif self.loss == topK_overlap_loss:
                pred_loss = self.loss(att[:att_orig.shape[0]], att_orig, adj, self.K).to(self.device)
            else:
                pred_loss = self.loss(pred, pred_orig).to(self.device)

            model.zero_grad()
            pred_loss.backward(retain_graph=True)
            grad = features_attack.grad.data
            features_attack = features_attack.clone() + self.epsilon * grad.sign()
            features_attack = torch.clamp(features_attack, feat_lim_min, feat_lim_max)
            features_attack = features_attack.detach()

            # test_score = self.eval_metric(orig_labels.detach().numpy(), pred.detach().numpy())
            #
            # if self.verbose:
            #     print("Attacking: Epoch {}, Loss: {:.5f}, test score: {:.5f}".format(i, pred_loss, test_score))

        return features_attack

    def modification(self, model, adj, features_attack, pred_orig, att_orig, target_mask):
        feat_lim_min, feat_lim_max = self.feat_lim_min, self.feat_lim_max

        n_total = features_attack.shape[0]
        features_attack = utils.feat_preprocess(features=features_attack, device=self.device)
        model.eval()

        if isinstance(model, GTNodeClassifier) and not os.path.exists(
                f'./GT_pos_encoding/{self.dataset}_pos_enc_perturbed.pth'):
            in_degrees = torch.tensor(adj.sum(axis=0)).squeeze()
            model.pos_enc_ = laplacian_pe(adj, in_degrees, padding=True).to(features_attack.device)

        for i in range(self.n_epoch):
            features_attack.requires_grad_(True)
            features_attack.retain_grad()
            pred, graph_repr, att = model(features_attack, adj)
            orig_labels = torch.argmax(pred_orig, dim=1)

            if self.loss == F.nll_loss:
                pred_loss = self.loss(pred[:n_total][target_mask], orig_labels[target_mask]).to(self.device)
            elif self.loss == topK_overlap_loss:
                pred_loss = self.loss(att[:att_orig.shape[0]], att_orig, adj, self.K).to(self.device)
            else:
                pred_loss = self.loss(pred[:n_total][target_mask], pred_orig[target_mask]).to(self.device)

            model.zero_grad()
            pred_loss.backward(retain_graph=True)
            grad = features_attack.grad.data
            features_attack = features_attack.clone() + self.epsilon * grad.sign()
            features_attack = torch.clamp(features_attack, feat_lim_min, feat_lim_max)
            features_attack = features_attack.detach()

            test_score = self.eval_metric(pred[:n_total][target_mask], orig_labels[target_mask])

            if self.verbose:
                print("Attacking: Epoch {}, Loss: {:.5f}, test score: {:.5f}".format(i, pred_loss, test_score))

        return adj, features_attack
