import dgl
import time
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import dgl.function as fn
from dgl.nn import GATv2Conv
from dgl.base import DGLError
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from dgl.nn.pytorch.utils import Identity
import dgl.sparse as dglsp
from ogb.graphproppred.mol_encoder import AtomEncoder
from utils import k_shell_algorithm, feature_normalize
import torch.optim as optim
from deeprobust.graph.utils import accuracy
from deeprobust.graph.defense.pgd import PGD, prox_operators


class GATConv(nn.Module):
    def __init__(
            self,
            in_feats,
            out_feats,
            num_heads,
            feat_drop=0.6,
            attn_drop=0.6,
            negative_slope=0.2,
            residual=False,
            activation=None,
            allow_zero_in_degree=True,
            bias=True,
    ):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.has_linear_res = False
        self.has_explicit_bias = False
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=bias)
                self.has_linear_res = True
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)

        if bias and not self.has_linear_res:
            self.bias = nn.Parameter(
                torch.FloatTensor(size=(num_heads * out_feats,))
            )
            self.has_explicit_bias = True
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.has_explicit_bias:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.res_fc.bias is not None:
                nn.init.constant_(self.res_fc.bias, 0)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_weight=None, get_attention=False):
        with graph.local_scope():  # graph.local_scope()是为了避免意外覆盖现有的特征数据
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError("There are 0-in-degree nodes in the graph")

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    feat_src = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(*dst_prefix_shape, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(*dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                # Wh_i(src)、Wh_j(dst)在各head的特征组成的矩阵: (1, num_heads, out_feats)
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                    h_dst = h_dst[: graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            # e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
            e = self.leaky_relu(graph.edata.pop("e"))
            # compute softmax, \alpha_i,j = softmax e_ij
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            if edge_weight is not None:
                graph.edata["a"] = graph.edata["a"] * edge_weight.tile(1, self._num_heads, 1).transpose(0, 2)
            # message passing, 'm' = \alpha * Wh_j, feature = \sum(\alpha_i,j * Wh_j)
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]
            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.has_explicit_bias:
                rst = rst + self.bias.view(*((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst


class GATNodeClassifier(nn.Module):
    def __init__(self, feats_size, hidden_size, out_size, n_layers, n_heads, feat_drop, attn_drop, LayerNorm=False):
        super(GATNodeClassifier, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            GATConv(feats_size, hidden_size, n_heads[0], feat_drop, attn_drop, activation=F.elu)
        )
        for i in range(0, n_layers - 1):
            in_hid_dim = hidden_size * n_heads[i]
            self.layers.append(
                GATConv(in_hid_dim, hidden_size, n_heads[i + 1], feat_drop, attn_drop, activation=F.elu)
            )
        self.out_layer = GATConv(hidden_size * n_heads[-1], out_size, 1, feat_drop, attn_drop, activation=F.elu)
        self.dropout = nn.Dropout(0.6)
        self.LayerNorm = LayerNorm

    def forward(self, x, adj):
        if self.LayerNorm:
            x = feature_normalize(x)
        g = dgl.from_scipy(adj).to(x.device)
        g.ndata['features'] = x

        for layer in self.layers:
            x, att = layer(g, x, get_attention=True)
            x = F.elu(x, alpha=1)
            x = x.flatten(1)  # use concat to handle multi-head. for mean method, use x = x.mean(1)
            x = self.dropout(x)
        graph_representation = x.mean(dim=0)
        x, att = self.out_layer(g, x, get_attention=True)
        logits = x.flatten(1)

        return logits, graph_representation, att.squeeze()


class GATv2NodeClassifier(nn.Module):
    def __init__(self, feats_size, hidden_size, out_size, n_layers, n_heads, feat_drop, attn_drop):
        super(GATv2NodeClassifier, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            GATv2Conv(feats_size, hidden_size, n_heads[0], feat_drop, attn_drop, activation=F.elu,
                      allow_zero_in_degree=True)
        )
        for i in range(0, n_layers - 1):
            in_hid_dim = hidden_size * n_heads[i]
            self.layers.append(
                GATv2Conv(in_hid_dim, hidden_size, n_heads[i + 1], feat_drop, attn_drop, activation=F.elu,
                          allow_zero_in_degree=True)
            )
        self.out_layer = GATv2Conv(hidden_size * n_heads[-1], out_size, 1, feat_drop, attn_drop, activation=F.elu,
                                   allow_zero_in_degree=True)
        self.dropout = nn.Dropout(0.6)

    def forward(self, x, adj):
        g = dgl.from_scipy(adj).to(x.device)
        g.ndata['features'] = x

        for layer in self.layers:
            x, att = layer(g, x, get_attention=True)
            x = F.elu(x, alpha=1)
            x = x.flatten(1)  # use concat to handle multi-head. for mean method, use x = x.mean(1)
            x = self.dropout(x)
        graph_representation = x.mean(dim=0)
        x = self.out_layer(g, x)
        logits = x.flatten(1)

        return logits, graph_representation, att


class GATGraphClassifier(nn.Module):
    def __init__(self, feats_size, hidden_size, n_classes, n_layers, n_heads, feat_drop, attn_drop, readout_type=''):
        super(GATGraphClassifier, self).__init__()
        self.readout_type = readout_type
        self.layers = nn.ModuleList()
        self.layers.append(
            GATConv(feats_size, hidden_size, n_heads[0], feat_drop, attn_drop, activation=F.elu)
        )
        for i in range(0, n_layers - 1):
            in_hid_dim = hidden_size * n_heads[i]
            self.layers.append(
                GATConv(in_hid_dim, hidden_size, n_heads[i + 1], feat_drop, attn_drop, activation=F.elu)
            )
        self.out_layer = nn.Linear(hidden_size * n_heads[-1], n_classes)

    def forward(self, x, g):
        for layer in self.layers:
            x, att = layer(g, x, get_attention=True)
            x = x.flatten(1)  # use concat to handle multi-head. for mean method, use h = h.mean(1)
        attention = att
        x = self.out_layer(x)
        g.ndata['h'] = x

        if self.readout_type == 'K-shell':
            src, dst = g.edges()
            num_nodes = g.number_of_nodes()
            adj = sp.csr_matrix((np.ones(len(src)), (src.cpu().numpy(), dst.cpu().numpy())),
                                shape=(num_nodes, num_nodes))
            k_values = torch.tensor(k_shell_algorithm(adj), dtype=torch.float32).to(x.device)
            k_values /= k_values.sum()
            g.ndata['w'] = k_values.view(-1, 1).repeat(1, x.shape[1])
            graph_representation = dgl.readout_nodes(g, 'h', weight='w')
        elif self.readout_type == 'mean':
            graph_representation = dgl.readout_nodes(g, 'h', op='mean')
        elif self.readout_type == 'max':
            graph_representation = dgl.readout_nodes(g, 'h', op='max')
        elif self.readout_type == 'min':
            graph_representation = dgl.readout_nodes(g, 'h', op='min')
        else:
            raise ValueError(f"Unknown readout type: {self.readout_type}")

        return graph_representation, graph_representation, attention


class ProGNN:
    def __init__(self, model, adj, args, device, debug=True):
        self.device = device
        self.epochs = args.num_epochs
        self.outer_steps = 1
        self.inner_steps = 2
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_graph = None
        self.weights = None
        self.estimator = None
        self.alpha = 5e-4  # weight of l1 norm
        self.beta = 1.5  # weight of nuclear norm
        self.gamma = 1  # weight of l2 norm
        self.lambda_ = 0  # weight of feature smoothing
        self.phi = 0  # weight of symmetric loss
        self.model = model.to(device)
        self.debug = debug
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.estimator = EstimateAdj(adj, device=device).to(device)
        self.optimizer_adj = optim.SGD(self.estimator.parameters(), momentum=0.9, lr=0.01)

        self.optimizer_l1 = PGD(self.estimator.parameters(), proxs=[prox_operators.prox_l1],
                                lr=0.01, alphas=[self.alpha])
        self.optimizer_nuclear = PGD(self.estimator.parameters(), proxs=[prox_operators.prox_nuclear],
                                     lr=0.01, alphas=[self.beta])

    def fit(self, features, adj, labels, train_idx, valid_idx, **kwargs):
        t_total = time.time()
        for epoch in range(self.epochs):
            for i in range(int(self.outer_steps)):
                self.train_adj(epoch, features, adj, labels, train_idx, valid_idx)
            for i in range(int(self.inner_steps)):
                self.train_gcn(epoch, features, self.estimator.estimated_adj, labels, train_idx, valid_idx)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        print("picking the best model according to validation performance")
        self.model.load_state_dict(self.weights)

    def train_gcn(self, epoch, features, adj, labels, train_idx, idx_val):
        estimator = self.estimator
        adj = estimator.normalize()

        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()

        if isinstance(adj, torch.Tensor):
            adj_ = adj.to_sparse(layout=torch.sparse_csr).detach().cpu()
            adj_ = sp.csr_matrix((adj_.values(), (adj_.crow_indices(), adj_.col_indices())), shape=adj_.size())
        else:
            adj_ = adj
        output = self.model(features, adj_)
        loss_train = F.cross_entropy(output[train_idx], labels[train_idx])
        acc_train = accuracy(output[train_idx], labels[train_idx])
        loss_train.backward()
        self.optimizer.step()

        self.model.eval()
        output = self.model(features, adj_)

        loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if self.debug:
                print('\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if self.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())

        if self.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'acc_train: {:.4f}'.format(acc_train.item()),
                      'loss_val: {:.4f}'.format(loss_val.item()),
                      'acc_val: {:.4f}'.format(acc_val.item()),
                      'time: {:.4f}s'.format(time.time() - t))

    def train_adj(self, epoch, features, adj, labels, idx_train, idx_val):
        estimator = self.estimator
        if self.debug:
            print("\n=== train_adj ===")
        t = time.time()
        estimator.train()
        self.optimizer_adj.zero_grad()

        loss_l1 = torch.norm(estimator.estimated_adj, 1)
        loss_fro = torch.norm(estimator.estimated_adj - adj, p='fro')
        normalized_adj = estimator.normalize()

        if self.lambda_:
            loss_smooth_feat = self.feature_smoothing(estimator.estimated_adj, features)
        else:
            loss_smooth_feat = 0 * loss_l1

        if isinstance(normalized_adj, torch.Tensor):
            adj_ = normalized_adj.to_sparse(layout=torch.sparse_csr).detach().cpu()
            adj_ = sp.csr_matrix((adj_.values(), (adj_.crow_indices(), adj_.col_indices())), shape=adj_.size())
            print(111)
        else:
            adj_ = normalized_adj
        output = self.model(features, adj_)
        loss_gcn = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

        loss_symmetric = torch.norm(estimator.estimated_adj - estimator.estimated_adj.t(), p="fro")

        loss_diffiential = loss_fro + self.gamma * loss_gcn + self.lambda_ * loss_smooth_feat + self.phi * loss_symmetric

        loss_diffiential.backward()

        self.optimizer_adj.step()
        loss_nuclear = 0 * loss_fro
        if self.beta != 0:
            self.optimizer_nuclear.zero_grad()
            self.optimizer_nuclear.step()
            loss_nuclear = prox_operators.nuclear_norm

        self.optimizer_l1.zero_grad()
        self.optimizer_l1.step()

        total_loss = loss_fro \
                     + self.gamma * loss_gcn \
                     + self.alpha * loss_l1 \
                     + self.beta * loss_nuclear \
                     + self.phi * loss_symmetric

        estimator.estimated_adj.data.copy_(torch.clamp(
            estimator.estimated_adj.data, min=0, max=1))

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        normalized_adj = estimator.normalize()
        output = self.model(features, adj_)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch + 1),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = normalized_adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if self.debug:
                print(f'\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = normalized_adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            if self.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())

        if self.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_fro: {:.4f}'.format(loss_fro.item()),
                      'loss_gcn: {:.4f}'.format(loss_gcn.item()),
                      'loss_feat: {:.4f}'.format(loss_smooth_feat.item()),
                      'loss_symmetric: {:.4f}'.format(loss_symmetric.item()),
                      'delta_l1_norm: {:.4f}'.format(torch.norm(estimator.estimated_adj - adj, 1).item()),
                      'loss_l1: {:.4f}'.format(loss_l1.item()),
                      'loss_total: {:.4f}'.format(total_loss.item()),
                      'loss_nuclear: {:.4f}'.format(loss_nuclear.item()))

    def feature_smoothing(self, adj, X):
        adj = (adj.t() + adj) / 2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj

        r_inv = r_inv + 1e-3
        r_inv = r_inv.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        # L = r_mat_inv @ L
        L = r_mat_inv @ L @ r_mat_inv

        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)
        return loss_smooth_feat


class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, adj, symmetric=False, device='cpu'):
        super(EstimateAdj, self).__init__()
        n = adj.shape[0]
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n))
        self._init_estimation(adj)
        self.symmetric = symmetric
        self.device = device

    def _init_estimation(self, adj):
        with torch.no_grad():
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

    def normalize(self):
        if self.symmetric:
            adj = (self.estimated_adj + self.estimated_adj.t()) / 2
        else:
            adj = self.estimated_adj

        normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx


class SparseMHA(nn.Module):  # Sparse Multi-head Attention Module
    def __init__(self, hidden_size=80, n_heads=8):
        super(SparseMHA, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, A, h):
        N = len(h)
        # [N, dh, nh]
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        q *= self.scaling
        # [N, dh, nh]
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        # [N, dh, nh]
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)

        # (HIGHLIGHT) Compute the multi-head attention with Sparse Matrix API
        attn = dglsp.bsddmm(A, q, k.transpose(1, 0))  # (sparse) [N, N, nh]
        # Sparse softmax by default applies on the last sparse dimension.
        attn = attn.softmax()  # (sparse) [N, N, nh]
        out = dglsp.bspmm(attn, v)  # [N, dh, nh]

        return self.out_proj(out.reshape(N, -1)), attn


class GTLayer(nn.Module):
    def __init__(self, hidden_size=80, n_heads=8):
        super(GTLayer, self).__init__()
        self.MHA = SparseMHA(hidden_size=hidden_size, n_heads=n_heads)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.FFN1 = nn.Linear(hidden_size, hidden_size * 2)
        self.FFN2 = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, h, adj):
        h1 = h
        h, att = self.MHA(adj, h)
        h = self.batchnorm1(h + h1)

        h2 = h
        h = self.FFN2(F.relu(self.FFN1(h)))
        h = h2 + h

        return self.batchnorm2(h), att


class GTNodeClassifier(nn.Module):
    def __init__(self, out_size, hidden_size=80, pos_enc_size=2, n_layers=8, n_heads=8):
        super(GTNodeClassifier, self).__init__()
        self.atom_encoder = AtomEncoder(hidden_size)
        self.pos_linear = nn.Linear(pos_enc_size, hidden_size)
        self.layers = nn.ModuleList(
            [GTLayer(hidden_size, n_heads) for _ in range(n_layers)]
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, out_size),
        )

    def forward(self, X, adj, pos_enc):
        h = self.atom_encoder(X) + self.pos_linear(pos_enc)
        for layer in self.layers:
            h, att = layer(h, adj)
        graph_representation = h.mean(dim=0)
        logits = self.predictor(h)

        return logits, graph_representation, att
