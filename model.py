import dgl
import dgl.function as fn
from dgl.base import DGLError
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from dgl.nn.pytorch.utils import Identity
import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import scipy.sparse as sp
import torch.nn.functional as F
from utils import k_shell_algorithm


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input_feature, adj):
        h = torch.mm(input_feature, self.W)  # [N, out_features]
        N = h.size(0)

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        # shape[N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # [N, N, 1] -> [N, N]

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        # adj_sparse = dgl.from_scipy(sp.csr_matrix(adj.numpy()))
        # attention_x = adj_sparse.multiply(e)
        #
        # print(attention_x == attention)

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)  # [N, N], [N, out_features] --> [N, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GATGraphClassifier(nn.Module):
    def __init__(self, in_feats, hidden_dim, n_classes, dropout, alpha, n_heads, readout_type='mean'):
        super(GATGraphClassifier, self).__init__()
        self.dropout = dropout
        self.readout_type = readout_type

        self.attentions = [GraphAttentionLayer(in_feats, hidden_dim, dropout, alpha, True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(hidden_dim * n_heads, n_classes, dropout, alpha, False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)  # N * hidden_dim
        if self.readout_type == 'top-k':
            k_values = torch.tensor(k_shell_algorithm(adj), dtype=torch.float32)
            k_values /= k_values.sum()
            x = F.elu(self.out_att(x, adj))
            x = torch.matmul(k_values, x).unsqueeze(0)  # 1 * n_classes
        else:
            x = F.elu(self.out_att(x, adj))  # 1 * hidden_dim
            x = torch.mean(x, dim=0).unsqueeze(0)  # 1 * n_classes
        return x


class GATNodeClassifier(nn.Module):
    def __init__(self, in_feats, hidden_dim, n_classes, dropout, alpha, n_heads):
        super(GATNodeClassifier, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(in_feats, hidden_dim, dropout, alpha, True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(hidden_dim * n_heads, n_classes, dropout, alpha, False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)  # N * hidden_dim
        x = F.elu(self.out_att(x, adj))
        return x


class GATConv(nn.Module):
    def __init__(
            self,
            in_feats,
            out_feats,
            num_heads,
            feat_drop=0.0,
            attn_drop=0.0,
            negative_slope=0.2,
            residual=False,
            activation=None,
            allow_zero_in_degree=False,
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

            def message_func(edges):
                print(1)
                return {'a': edges.data['a']}

            def reduce_func(nodes):
                print(2)
                return {'attention_list': nodes.mailbox['a'].sum(1)}

            graph.update_all(message_func=message_func, reduce_func=reduce_func)

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst


class GATNodeClassifier_DGL(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes, num_heads):
        super(GATNodeClassifier_DGL, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_feats, hidden_feats, num_heads, feat_drop=0.6, attn_drop=0.6))
        for _ in range(1, num_heads):
            self.layers.append(GATConv(hidden_feats * num_heads, hidden_feats, num_heads, feat_drop=0.6, attn_drop=0.6))
        self.fc = nn.Linear(hidden_feats * num_heads, num_classes)

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h, att = layer(g, h, get_attention=True)
            h = h.flatten(1)
        logits = self.fc(h)
        return logits


class GraphClassifierExample(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GraphClassifierExample, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h):
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        with g.local_scope():
            g.ndata['h'] = h
            # 使用平均读出计算图表示
            hg = dgl.mean_nodes(g, 'h')  # N * hid_dim
            out = self.classify(hg)  # num_graph * n_classes
            return out
