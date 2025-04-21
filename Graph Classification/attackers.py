import os
import dgl
import torch
import torch.nn as nn
from torch import Tensor


class GraphRandomAttacker:
    def __init__(self,
                 feat_noise_ratio=0.1,
                 edge_add_ratio=0.1,
                 inject_node_max=3,
                 feat_lim=(-1., 1.),
                 attack_mode='mixed'):
        self.feat_ratio = feat_noise_ratio
        self.edge_add_ratio = edge_add_ratio
        self.inject_max = inject_node_max
        self.feat_min, self.feat_max = feat_lim
        self.mode = attack_mode

    def _perturb_features(self, feat: Tensor) -> Tensor:
        noise_mask = torch.rand_like(feat) < self.feat_ratio
        noise = torch.empty_like(feat).uniform_(self.feat_min, self.feat_max)
        return torch.where(noise_mask, noise, feat)

    def _perturb_edges(self, graph: dgl.DGLGraph) -> (dgl.DGLGraph, torch.Tensor):
        src, dst = graph.edges()
        n_edges = len(src)
        device = graph.device

        # 生成新边
        n_add = int(n_edges * self.edge_add_ratio)
        new_src = torch.randint(0, graph.num_nodes(), (n_add,), device=device)
        new_dst = torch.randint(0, graph.num_nodes(), (n_add,), device=device)

        # 合并新旧边
        total_src = torch.cat([src, new_src])
        total_dst = torch.cat([dst, new_dst])

        # 创建边掩码（原始边为True，新增边为False）
        edge_mask = torch.cat([
            torch.ones(n_edges, dtype=torch.bool, device=device),
            torch.zeros(n_add, dtype=torch.bool, device=device)
        ])

        return dgl.graph((total_src, total_dst), device=device), edge_mask

    def _inject_nodes(self, graph: dgl.DGLGraph, features: Tensor) -> (dgl.DGLGraph, Tensor, torch.Tensor):
        n_inject = min(self.inject_max, graph.num_nodes())
        device = graph.device

        # 添加新节点
        new_g = dgl.add_nodes(graph, n_inject)
        new_feat = torch.rand(n_inject, features.shape[1], device=device)
        new_feat = new_feat * (self.feat_max - self.feat_min) + self.feat_min

        # 生成新连接
        src_list, dst_list = [], []
        for i in range(graph.num_nodes(), graph.num_nodes() + n_inject):
            targets = torch.randint(0, graph.num_nodes(), (2,), device=device)
            # 双向连接
            src_list.extend([i, i, targets[0], targets[1]])
            dst_list.extend([targets[0], targets[1], i, i])

        # 添加边并生成掩码
        if src_list:
            new_g.add_edges(torch.tensor(src_list, device=device),
                            torch.tensor(dst_list, device=device))
            injected_mask = torch.zeros(len(src_list), dtype=torch.bool, device=device)
        else:
            injected_mask = torch.tensor([], dtype=torch.bool, device=device)

        return new_g, torch.cat([features, new_feat.to(features.device)]), injected_mask

    def _perturb_single(self, graph: dgl.DGLGraph, features: Tensor) -> tuple:
        new_g = graph.clone()
        new_feat = features.clone()
        device = features.device

        # 初始化掩码
        node_mask = torch.ones(new_g.num_nodes(), dtype=torch.bool, device=device)
        edge_mask = torch.ones(new_g.num_edges(), dtype=torch.bool, device=device)

        # 特征扰动
        if self.mode in ['feat', 'mixed']:
            new_feat = self._perturb_features(new_feat)

        # 结构扰动
        if self.mode in ['struct', 'mixed']:
            new_g, struct_mask = self._perturb_edges(new_g)
            edge_mask = struct_mask

        # 节点注入
        if self.mode in ['inject', 'mixed'] and self.inject_max > 0:
            new_g, extended_feat, inject_mask = self._inject_nodes(new_g, new_feat)
            new_feat = extended_feat
            node_mask = torch.cat([
                node_mask,
                torch.zeros(new_g.num_nodes() - node_mask.size(0),
                            dtype=torch.bool, device=device)
            ])
            edge_mask = torch.cat([edge_mask, inject_mask])

        return new_g, new_feat, node_mask, edge_mask

    def perturb_batch(self, graphs: dgl.DGLGraph, features: Tensor) -> tuple:
        graph_list = dgl.unbatch(graphs)
        mod_graphs, mod_feats, node_masks, edge_masks = [], [], [], []
        current_idx = 0

        for g in graph_list:
            # 提取对应子图特征
            num_nodes = g.num_nodes()
            sub_feat = features[current_idx:current_idx + num_nodes]
            current_idx += num_nodes

            # 执行扰动
            perturb_g, perturb_feat, n_mask, e_mask = self._perturb_single(g, sub_feat)

            mod_graphs.append(perturb_g)
            mod_feats.append(perturb_feat)
            node_masks.append(n_mask)
            edge_masks.append(e_mask)

        return (
            dgl.batch(mod_graphs),
            torch.cat(mod_feats),
            torch.cat(node_masks),
            torch.cat(edge_masks)
        )

    @property
    def device(self):
        return next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cpu')


class GraphPGDAttacker:
    def __init__(self,
                 epsilon=0.1,
                 n_epoch=10,
                 feat_lim=(-1., 1.),
                 edge_perturb_ratio=0.1,
                 inject_node_max=0,
                 device='cpu'):
        """
        图级别PGD攻击器

        参数:
        - epsilon: 扰动强度
        - n_epoch: 攻击迭代次数
        - feat_lim: 特征扰动范围
        - edge_perturb_ratio: 边修改比例
        - inject_node_max: 最大注入节点数
        """
        self.epsilon = epsilon
        self.n_epoch = n_epoch
        self.feat_min, self.feat_max = feat_lim
        self.edge_ratio = edge_perturb_ratio
        self.inject_max = inject_node_max
        self.device = device

    def perturb_batch(self, model, graphs, features):
        """生成对抗样本 (保持原始图结构)"""
        model.eval()
        perturbed_graphs = []
        perturbed_feats = []

        # 分解批处理图
        graph_list = dgl.unbatch(graphs)
        total_nodes = features.shape[0]
        current_idx = 0

        for g in graph_list:
            # 提取当前图的特征
            num_nodes = g.num_nodes()
            feat_slice = slice(current_idx, current_idx + num_nodes)
            orig_feat = features[feat_slice].clone().detach().requires_grad_(True)
            current_idx += num_nodes

            # 执行单图攻击
            adv_g, adv_feat = self._attack_single(model, g.clone(), orig_feat)
            perturbed_graphs.append(adv_g)
            perturbed_feats.append(adv_feat)

        # 重新批处理并合并特征
        return dgl.batch(perturbed_graphs), torch.cat(perturbed_feats, dim=0)

    def _attack_single(self, model, graph, features):
        """确保adj_grad和feat_grad存在的攻击方法"""
        device = self.device
        model.eval()

        # 初始化可导参数 --------------------------------
        graph = graph.to(device)

        # 邻接矩阵（必须为浮点型）
        adj = graph.adjacency_matrix().to_dense().float().to(device)
        adj_param = nn.Parameter(adj, requires_grad=True)

        # 特征（确保requires_grad=True）
        features = features.clone().detach().to(device).requires_grad_(True)

        # 攻击循环 --------------------------------
        for _ in range(self.n_epoch):
            # 构建可导图结构（关键步骤：保留梯度链接）
            src, dst = (adj_param.sigmoid() > 0.5).nonzero(as_tuple=True)
            current_g = dgl.graph((src, dst), device=device)

            # 前向传播（必须使用参数化的adj_param和features）
            logits, _ = model(features, current_g)
            loss = -logits.mean()  # 攻击目标：最大化损失

            # 显式梯度计算 ----------------------------
            grad_adj, grad_feat = torch.autograd.grad(
                outputs=loss,
                inputs=[adj_param, features],
                create_graph=False,  # 避免二阶梯度
                retain_graph=False,  # 单次梯度计算
                allow_unused=False  # 强制所有输入被使用
            )

            # 断言梯度存在性（调试用）
            assert grad_adj is not None, "邻接矩阵梯度未计算！"
            assert grad_feat is not None, "特征梯度未计算！"

            # 更新参数 ----------------------------
            with torch.no_grad():
                # 邻接矩阵更新（带投影）
                adj_param.data += self.epsilon * grad_adj.sign()
                adj_param.data.clamp_(0, 1)

                # 特征更新（带投影）
                features.data += self.epsilon * grad_feat.sign()
                features.data.clamp_(self.feat_min, self.feat_max)

        # 生成最终对抗图 ----------------------------
        final_adj = (adj_param > 0.5).float()
        src, dst = final_adj.nonzero(as_tuple=True)
        return dgl.graph((src, dst), device=device), features.detach()

    def save_adv_samples(self, graphs, features, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        for i, (g, feat) in enumerate(zip(dgl.unbatch(graphs), features)):
            dgl.save_graphs(f"{save_dir}/graph_{i}.bin", [g])
            torch.save(feat, f"{save_dir}/feat_{i}.pt")
