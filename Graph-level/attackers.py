import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from grb.attack.base import InjectionAttack, EarlyStop
from grb.evaluator import metric
from grb.utils import utils

from utils import topK_overlap_loss


class PGD(InjectionAttack):
    r"""

    Description
    -----------
    Graph injection attack version of Projected Gradient Descent attack (`PGD <https://arxiv.org/abs/1706.06083>`__).

    Parameters
    ----------
    epsilon : float
        Perturbation level on features.
    n_epoch : int
        Epoch of perturbations.
    n_inject_max : int
        Maximum number of injected nodes.
    n_edge_max : int
        Maximum number of edges of injected nodes.
    feat_lim_min : float
        Minimum limit of features.
    feat_lim_max : float
        Maximum limit of features.
    loss : func of torch.nn.functional, optional
        Loss function compatible with ``torch.nn.functional``. Default: ``F.nll_loss``.
    eval_metric : func of grb.evaluator.metric, optional
        Evaluation metric. Default: ``metric.eval_acc``.
    device : str, optional
        Device used to host data. Default: ``cpu``.
    early_stop : bool, optional
        Whether to early stop. Default: ``False``.
    verbose : bool, optional
        Whether to display logs. Default: ``True``.

    """

    def __init__(self,
                 epsilon,
                 n_epoch,
                 n_inject_max,
                 n_edge_max,
                 feat_lim_min,
                 feat_lim_max,
                 loss=F.nll_loss,
                 eval_metric=metric.eval_acc,
                 device='cpu',
                 early_stop=True,
                 verbose=True):
        self.device = device
        self.epsilon = epsilon
        self.n_epoch = n_epoch
        self.n_inject_max = n_inject_max
        self.n_edge_max = n_edge_max
        self.feat_lim_min = feat_lim_min
        self.feat_lim_max = feat_lim_max
        self.loss = loss
        self.eval_metric = eval_metric
        self.verbose = verbose

        if early_stop:
            self.early_stop = EarlyStop(patience=1000, epsilon=1e-4)
        else:
            self.early_stop = early_stop

    def attack(self, model, g, features, target_mask, adj_norm_func):
        r"""

        Description
        -----------
        Attack process consists of injection and feature update.

        Parameters
        ----------
        model : torch.nn.module
            Model implemented based on ``torch.nn.module``.
        adj : scipy.sparse.csr.csr_matrix
            Adjacency matrix in form of ``N * N`` sparse matrix.
        features : torch.FloatTensor
            Features in form of ``N * D`` torch float tensor.
        target_mask : torch.Tensor
            Mask of attack target nodes in form of ``N * 1`` torch bool tensor.
        adj_norm_func : func of utils.normalize
            Function that normalizes adjacency matrix.

        Returns
        -------
        adj_attack : scipy.sparse.csr.csr_matrix
            Adversarial adjacency matrix in form of :math:`(N + N_{inject})\times(N + N_{inject})` sparse matrix.
        features_attack : torch.FloatTensor
            Features of nodes after attacks in form of :math:`N_{inject}` * D` torch float tensor.

        """

        model.to(self.device)
        n_total, n_feat = features.shape
        features = utils.feat_preprocess(features=features, device=self.device)

        pred_orig, _, att_orig = model(features, g)

        features_attack = np.random.normal(loc=0, scale=self.feat_lim_max / 10, size=(self.n_inject_max, n_feat))
        features_attack = self.update_features(model=model,
                                               adj=adj,
                                               adj_attack=adj_attack,
                                               features=features,
                                               features_attack=features_attack,
                                               pred_orig=pred_orig,
                                               att_orig=att_orig,
                                               target_mask=target_mask,
                                               adj_norm_func=adj_norm_func)

        return features_attack

    def update_features(self, model, adj, adj_attack, features, features_attack, pred_orig, att_orig, target_mask,
                        adj_norm_func):
        r"""

        Description
        -----------
        Update features of injected nodes.

        Parameters
        ----------
        model : torch.nn.module
            Model implemented based on ``torch.nn.module``.
        adj : scipy.sparse.csr.csr_matrix
            Original adjacency matrix in form of :math:`(N)\times(N)` sparse matrix.
        adj_attack :  scipy.sparse.csr.csr_matrix
            Adversarial adjacency matrix in form of :math:`(N + N_{inject})\times(N + N_{inject})` sparse matrix.
        features : torch.FloatTensor
            Features in form of ``N * D`` torch float tensor.
        features_attack : torch.FloatTensor
            Features of nodes after attacks in form of :math:`N_{inject}` * D` torch float tensor.
        pred_orig : torch.LongTensor
            Labels of target nodes originally predicted by the model.
        att_orig : torch.FloatTensor
            Attention list.
        target_mask : torch.Tensor
            Mask of target nodes in form of ``N * 1`` torch bool tensor.
        adj_norm_func : func of utils.normalize
            Function that normalizes adjacency matrix.

        Returns
        -------
        features_attack : torch.FloatTensor
            Updated features of nodes after attacks in form of :math:`N_{inject}` * D` torch float tensor.

        """

        epsilon = self.epsilon
        n_epoch = self.n_epoch
        feat_lim_min, feat_lim_max = self.feat_lim_min, self.feat_lim_max

        n_total = features.shape[0]
        adj_attacked_tensor = utils.adj_preprocess(adj=adj_attack,
                                                   adj_norm_func=adj_norm_func,
                                                   model_type='dgl',
                                                   device=self.device)
        features_attack = utils.feat_preprocess(features=features_attack, device=self.device)
        model.eval()

        for i in range(n_epoch):
            features_attack.requires_grad_(True)
            features_attack.retain_grad()
            features_concat = torch.cat((features, features_attack), dim=0)
            pred, graph_repr, att = model(features_concat, adj_attacked_tensor)
            origin_labels = torch.argmax(pred_orig, dim=1)

            if self.loss == F.nll_loss:
                pred_loss = self.loss(pred[:n_total][target_mask], origin_labels[target_mask]).to(self.device)
            elif self.loss == topK_overlap_loss:
                pred_loss = self.loss(att, att_orig, adj).to(self.device)
            else:
                pred_loss = self.loss(pred[:n_total][target_mask], origin_labels[target_mask]).to(self.device)

            model.zero_grad()
            pred_loss.backward(retain_graph=True)
            grad = features_attack.grad.data
            features_attack = features_attack.clone() + epsilon * grad.sign()
            features_attack = torch.clamp(features_attack, feat_lim_min, feat_lim_max)
            features_attack = features_attack.detach()

            test_score = self.eval_metric(pred[:n_total][target_mask], origin_labels[target_mask])

            if self.early_stop:
                self.early_stop(test_score)
                if self.early_stop.stop:
                    print("Attacking: Early stopped.")
                    self.early_stop = EarlyStop()
                    return features_attack

            if self.verbose:
                print(
                    "Attacking: Epoch {}, Loss: {:.5f}, Surrogate test score: {:.5f}".format(i, pred_loss, test_score))

        return features_attack
