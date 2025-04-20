import dgl
import torch
import numpy as np
from torch import Tensor
from typing import List


class GraphRandomAttacker:
    def __init__(self,
                 feat_noise_ratio=0.1,
                 edge_add_ratio=0.1,
                 edge_del_ratio=0.1,
                 inject_node_max=3,
                 feat_lim=(-1., 1.),
                 attack_mode='mixed'):
        """
        图级别随机攻击器

        参数:
        - feat_noise_ratio: 特征扰动比例 (0-1)
        - edge_add_ratio: 边添加比例 (相对于原边数)
        - edge_del_ratio: 边删除比例 (相对于原边数)
        - inject_node_max: 最大注入节点数
        - feat_lim: (min, max) 特征扰动范围
        - attack_mode: 攻击模式组合，可选 ['feat', 'struct', 'inject', 'mixed']
        """
        self.feat_ratio = feat_noise_ratio
        self.edge_add_ratio = edge_add_ratio
        self.edge_del_ratio = edge_del_ratio
        self.inject_max = inject_node_max
        self.feat_min, self.feat_max = feat_lim
        self.mode = attack_mode

    def _perturb_features(self, feat: Tensor) -> Tensor:
        """随机特征扰动"""
        noise_mask = torch.rand_like(feat) < self.feat_ratio
        noise = torch.empty_like(feat).uniform_(self.feat_min, self.feat_max)
        return torch.where(noise_mask, noise, feat)

    def _perturb_edges(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        """随机边扰动"""
        src, dst = graph.edges()
        n_edges = len(src)

        # 删除边
        del_idx = np.random.choice(n_edges, int(n_edges * self.edge_del_ratio), replace=False)
        remaining = np.setdiff1d(np.arange(n_edges), del_idx)
        new_src = src[remaining]
        new_dst = dst[remaining]

        # 添加边
        n_nodes = graph.num_nodes()
        n_add = int(len(new_src) * self.edge_add_ratio)
        new_src = torch.cat([new_src, torch.randint(0, n_nodes, (n_add,))])
        new_dst = torch.cat([new_dst, torch.randint(0, n_nodes, (n_add,))])

        return dgl.graph((new_src, new_dst))

    def perturb_batch(self, graphs: dgl.DGLGraph, features: Tensor) -> (dgl.DGLGraph, Tensor):
        """
        修复维度不匹配问题的批处理扰动方法
        """
        graph_list = dgl.unbatch(graphs)
        mod_graphs = []
        mod_features_list = []
        current_idx = 0  # 跟踪原始特征索引

        for i, single_graph in enumerate(graph_list):
            # 获取当前图的原始节点数
            num_orig_nodes = single_graph.num_nodes()

            # 提取对应的原始特征
            orig_feat = features[current_idx:current_idx + num_orig_nodes]
            current_idx += num_orig_nodes  # 更新索引

            # 执行单图扰动
            perturb_g, perturb_feat = self._perturb_single(single_graph, orig_feat)

            mod_graphs.append(perturb_g)
            mod_features_list.append(perturb_feat)

        # 合并所有特征和图表
        batched_graph = dgl.batch(mod_graphs)
        perturbed_features = torch.cat(mod_features_list, dim=0)

        return batched_graph, perturbed_features

    def _perturb_single(self, graph: dgl.DGLGraph, features: Tensor) -> (dgl.DGLGraph, Tensor):
        """修复后的单图扰动逻辑"""
        new_g = graph.clone()
        new_feat = features.clone()

        # 特征扰动（不会改变节点数）
        if self.mode in ['feat', 'mixed']:
            new_feat = self._perturb_features(new_feat)

        # 结构扰动（不会改变节点数）
        if self.mode in ['struct', 'mixed']:
            new_g = self._perturb_edges(new_g)

        # 节点注入（会改变节点数）
        if self.mode in ['inject', 'mixed'] and self.inject_max > 0:
            new_g, extended_feat = self._inject_nodes(new_g, new_feat)
            new_feat = extended_feat  # 更新为包含新节点的特征

        return new_g, new_feat

    def _inject_nodes(self, graph: dgl.DGLGraph, features: Tensor) -> (dgl.DGLGraph, Tensor):
        """修复节点注入后的特征返回"""
        n_inject = min(self.inject_max, graph.num_nodes())

        # 生成新节点特征
        new_feat = torch.rand(n_inject, features.shape[1],
                              device=features.device)
        new_feat = new_feat * (self.feat_max - self.feat_min) + self.feat_min

        # 添加新节点到图
        new_g = dgl.add_nodes(graph, n_inject)

        # 连接新节点
        n_connect = int(graph.num_edges() / graph.num_nodes()) + 1
        for i in range(graph.num_nodes(), graph.num_nodes() + n_inject):
            targets = np.random.choice(graph.num_nodes(), n_connect, replace=False)
            new_g.add_edges(i, targets)
            new_g.add_edges(targets, i)

        # 合并特征
        total_feat = torch.cat([features, new_feat])
        return new_g, total_feat


class GraphRandomAttacker:
    def __init__(self,
                 feat_noise_ratio=0.1,
                 edge_add_ratio=0.1,
                 edge_del_ratio=0.1,
                 inject_node_max=3,
                 feat_lim=(-1., 1.),
                 attack_mode='mixed'):
        """
        图级别随机攻击器

        参数:
        - feat_noise_ratio: 特征扰动比例 (0-1)
        - edge_add_ratio: 边添加比例 (相对于原边数)
        - edge_del_ratio: 边删除比例 (相对于原边数)
        - inject_node_max: 最大注入节点数
        - feat_lim: (min, max) 特征扰动范围
        - attack_mode: 攻击模式组合，可选 ['feat', 'struct', 'inject', 'mixed']
        """
        self.feat_ratio = feat_noise_ratio
        self.edge_add_ratio = edge_add_ratio
        self.edge_del_ratio = edge_del_ratio
        self.inject_max = inject_node_max
        self.feat_min, self.feat_max = feat_lim
        self.mode = attack_mode

    def _perturb_features(self, feat: Tensor) -> Tensor:
        """随机特征扰动"""
        noise_mask = torch.rand_like(feat) < self.feat_ratio
        noise = torch.empty_like(feat).uniform_(self.feat_min, self.feat_max)
        return torch.where(noise_mask, noise, feat)

    def _perturb_edges(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        """随机边扰动"""
        src, dst = graph.edges()
        n_edges = len(src)

        # 删除边
        del_idx = np.random.choice(n_edges, int(n_edges * self.edge_del_ratio), replace=False)
        remaining = np.setdiff1d(np.arange(n_edges), del_idx)
        new_src = src[remaining]
        new_dst = dst[remaining]

        # 添加边
        n_nodes = graph.num_nodes()
        n_add = int(len(new_src) * self.edge_add_ratio)
        new_src = torch.cat([new_src, torch.randint(0, n_nodes, (n_add,))])
        new_dst = torch.cat([new_dst, torch.randint(0, n_nodes, (n_add,))])

        return dgl.graph((new_src, new_dst))

    def perturb_batch(self, graphs: dgl.DGLGraph, features: Tensor) -> (dgl.DGLGraph, Tensor, Tensor):
        """
        修复维度不匹配问题的批处理扰动方法，返回包含节点掩码的元组
        """
        graph_list = dgl.unbatch(graphs)
        mod_graphs = []
        mod_features_list = []
        mask_list = []  # 新增：收集各图的节点掩码
        current_idx = 0  # 跟踪原始特征索引

        for i, single_graph in enumerate(graph_list):
            # 获取当前图的原始节点数
            num_orig_nodes = single_graph.num_nodes()

            # 提取对应的原始特征
            orig_feat = features[current_idx:current_idx + num_orig_nodes]
            current_idx += num_orig_nodes  # 更新索引

            # 执行单图扰动
            perturb_g, perturb_feat, node_mask = self._perturb_single(
                single_graph,
                orig_feat
            )

            mod_graphs.append(perturb_g)
            mod_features_list.append(perturb_feat)
            mask_list.append(node_mask)  # 收集掩码

        # 合并所有特征、图表和掩码
        batched_graph = dgl.batch(mod_graphs)
        perturbed_features = torch.cat(mod_features_list, dim=0)
        total_node_mask = torch.cat(mask_list, dim=0)

        return batched_graph, perturbed_features, total_node_mask

    def _perturb_single(self, graph: dgl.DGLGraph, features: Tensor) -> (dgl.DGLGraph, Tensor, Tensor):
        """修复后的单图扰动逻辑，返回包含节点掩码的元组"""
        new_g = graph.clone()
        new_feat = features.clone()
        original_nodes = graph.num_nodes()  # 原始节点数
        device = features.device

        # 初始化节点掩码（全为原始节点）
        node_mask = torch.ones(original_nodes, dtype=torch.bool, device=device)

        # 特征扰动（不会改变节点数）
        if self.mode in ['feat', 'mixed']:
            new_feat = self._perturb_features(new_feat)

        # 结构扰动（不会改变节点数）
        if self.mode in ['struct', 'mixed']:
            new_g = self._perturb_edges(new_g)

        # 节点注入（会改变节点数）
        if self.mode in ['inject', 'mixed'] and self.inject_max > 0:
            new_g, extended_feat, injected_mask = self._inject_nodes(new_g, new_feat)
            new_feat = extended_feat
            # 合并节点掩码（原始节点为True，新增为False）
            node_mask = torch.cat([node_mask, injected_mask])

        return new_g, new_feat, node_mask

    def _inject_nodes(self, graph: dgl.DGLGraph, features: Tensor) -> (dgl.DGLGraph, Tensor, Tensor):
        """节点注入，返回新图、扩展特征和新增节点掩码"""
        n_inject = min(self.inject_max, graph.num_nodes())

        # 生成新节点特征
        new_feat = torch.rand(n_inject, features.shape[1],
                              device=features.device)
        new_feat = new_feat * (self.feat_max - self.feat_min) + self.feat_min

        # 添加新节点到图
        new_g = dgl.add_nodes(graph, n_inject)

        # 连接新节点
        n_connect = int(graph.num_edges() / graph.num_nodes()) + 1
        for i in range(graph.num_nodes(), graph.num_nodes() + n_inject):
            targets = np.random.choice(graph.num_nodes(), n_connect, replace=False)
            new_g.add_edges(i, targets)
            new_g.add_edges(targets, i)

        # 合并特征
        total_feat = torch.cat([features, new_feat])
        # 生成新增节点的掩码（False表示新增）
        injected_mask = torch.zeros(n_inject, dtype=torch.bool, device=features.device)

        return new_g, total_feat, injected_mask


class GraphRandomAttacker:
    def __init__(self,
                 feat_noise_ratio=0.1,
                 edge_add_ratio=0.1,
                 edge_del_ratio=0.1,
                 inject_node_max=3,
                 feat_lim=(-1., 1.),
                 attack_mode='mixed'):
        self.feat_ratio = feat_noise_ratio
        self.edge_add_ratio = edge_add_ratio
        self.edge_del_ratio = edge_del_ratio
        self.inject_max = inject_node_max
        self.feat_min, self.feat_max = feat_lim
        self.mode = attack_mode

    def _perturb_features(self, feat: Tensor) -> Tensor:
        """随机特征扰动"""
        noise_mask = torch.rand_like(feat) < self.feat_ratio
        noise = torch.empty_like(feat).uniform_(self.feat_min, self.feat_max)
        return torch.where(noise_mask, noise, feat)

    def _perturb_edges(self, graph: dgl.DGLGraph) -> (dgl.DGLGraph, torch.Tensor):
        """返回扰动后的图和边掩码"""
        src, dst = graph.edges()
        n_edges = len(src)
        device = graph.device

        # 删除边（标记保留的边）
        del_idx = np.random.choice(n_edges, int(n_edges * self.edge_del_ratio), replace=False)
        remaining = np.setdiff1d(np.arange(n_edges), del_idx)
        new_src = src[remaining]
        new_dst = dst[remaining]
        edge_mask = torch.ones(len(remaining), dtype=torch.bool, device=device)  # 保留边为True

        # 添加随机边（标记新增边）
        n_add = int(len(new_src) * self.edge_add_ratio)
        new_src_add = torch.randint(0, graph.num_nodes(), (n_add,), device=device)
        new_dst_add = torch.randint(0, graph.num_nodes(), (n_add,), device=device)
        edge_mask = torch.cat([edge_mask, torch.zeros(n_add, dtype=torch.bool, device=device)])

        new_g = dgl.graph((torch.cat([new_src, new_src_add]),
                           torch.cat([new_dst, new_dst_add])), device=device)
        return new_g, edge_mask

    # def _inject_nodes(self, graph: dgl.DGLGraph, features: Tensor) -> (dgl.DGLGraph, Tensor, torch.Tensor):
    #     """返回注入节点后的图、特征和新增边掩码"""
    #     n_inject = min(self.inject_max, graph.num_nodes())
    #     device = features.device
    #
    #     # 添加新节点
    #     new_g = dgl.add_nodes(graph, n_inject)
    #     new_feat = torch.rand(n_inject, features.shape[1], device=device) * (
    #                 self.feat_max - self.feat_min) + self.feat_min
    #
    #     # 连接新节点并记录边数
    #     added_edges = 0
    #     src_list, dst_list = [], []
    #     for i in range(graph.num_nodes(), graph.num_nodes() + n_inject):
    #         targets = np.random.choice(graph.num_nodes(), 2, replace=False)
    #         # 双向连接
    #         src_list.extend([i, i] + list(targets) * 2)
    #         dst_list.extend(list(targets) * 2 + [i, i])
    #         added_edges += 4
    #
    #     if src_list:
    #         new_g.add_edges(torch.tensor(src_list, device=device),
    #                         torch.tensor(dst_list, device=device))
    #
    #     # 生成新增边掩码（全为False）
    #     injected_edge_mask = torch.zeros(added_edges, dtype=torch.bool, device=device)
    #     return new_g, torch.cat([features, new_feat]), injected_edge_mask

    def _inject_nodes(self, graph: dgl.DGLGraph, features: Tensor) -> (dgl.DGLGraph, Tensor, torch.Tensor):
        """节点注入，返回新图、扩展特征和新增边掩码"""
        n_inject = min(self.inject_max, graph.num_nodes())
        device = graph.device  # 关键修复：从原图获取设备信息

        # 生成新节点特征（保持设备一致）
        new_feat = torch.rand(n_inject, features.shape[1],
                              device=device)  # 添加设备参数
        new_feat = new_feat * (self.feat_max - self.feat_min) + self.feat_min

        # 添加新节点到图
        new_g = dgl.add_nodes(graph, n_inject)

        # 连接新节点（确保张量在正确设备）
        src_list, dst_list = [], []
        for i in range(graph.num_nodes(), graph.num_nodes() + n_inject):
            targets = np.random.choice(graph.num_nodes(), 2, replace=False)
            # 转换为Tensor时指定设备
            src_list.append(torch.tensor([i, i], device=device))
            dst_list.append(torch.tensor(targets, device=device))
            src_list.append(torch.tensor(targets, device=device))
            dst_list.append(torch.tensor([i, i], device=device))

        if src_list:
            new_src = torch.cat(src_list)
            new_dst = torch.cat(dst_list)
            new_g.add_edges(new_src, new_dst)

        # 生成新增边掩码（全为False）
        injected_edge_mask = torch.zeros(new_g.num_edges() - graph.num_edges(),
                                         dtype=torch.bool, device=device)
        new_feat = new_feat.to('cuda')
        return new_g, torch.cat([features, new_feat]), injected_edge_mask

    def _perturb_single(self, graph: dgl.DGLGraph, features: Tensor) -> (dgl.DGLGraph, Tensor, Tensor, Tensor):
        """返回扰动后的图、特征、节点掩码和边掩码"""
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
            new_g, struct_edge_mask = self._perturb_edges(new_g)
            edge_mask = struct_edge_mask

        # 节点注入
        if self.mode in ['inject', 'mixed'] and self.inject_max > 0:
            new_g, extended_feat, injected_edge_mask = self._inject_nodes(new_g, new_feat)
            new_feat = extended_feat
            node_mask = torch.cat([node_mask, torch.zeros(new_g.num_nodes() - node_mask.size(0),
                                                          dtype=torch.bool, device=device)])
            edge_mask = torch.cat([edge_mask, injected_edge_mask])

        return new_g, new_feat, node_mask, edge_mask

    def perturb_batch(self, graphs: dgl.DGLGraph, features: Tensor) -> tuple:
        """返回批处理后的图、特征、节点掩码和边掩码"""
        graph_list = dgl.unbatch(graphs)
        batched_graphs, all_feats, node_masks, edge_masks = [], [], [], []
        current_idx = 0

        for g in graph_list:
            # 提取对应子图的特征
            num_nodes = g.num_nodes()
            sub_feat = features[current_idx:current_idx + num_nodes]
            current_idx += num_nodes

            # 扰动单图
            perturb_g, perturb_feat, n_mask, e_mask = self._perturb_single(g, sub_feat)

            batched_graphs.append(perturb_g)
            all_feats.append(perturb_feat)
            node_masks.append(n_mask)
            edge_masks.append(e_mask)

        return (
            dgl.batch(batched_graphs),
            torch.cat(all_feats),
            torch.cat(node_masks),
            torch.cat(edge_masks)
        )



class GraphRandomAttacker:
    def __init__(self,
                 feat_noise_ratio=0.1,
                 edge_add_ratio=0.1,
                 inject_node_max=3,
                 feat_lim=(-1., 1.),
                 attack_mode='mixed'):
        """
        修改后的图随机攻击器（只能增加边）

        参数:
        - feat_noise_ratio: 特征扰动比例 (0-1)
        - edge_add_ratio: 边添加比例 (相对于原边数)
        - inject_node_max: 最大注入节点数
        - feat_lim: 特征扰动范围 (min, max)
        - attack_mode: 攻击模式 ['feat', 'struct', 'inject', 'mixed']
        """
        self.feat_ratio = feat_noise_ratio
        self.edge_add_ratio = edge_add_ratio
        self.inject_max = inject_node_max
        self.feat_min, self.feat_max = feat_lim
        self.mode = attack_mode

    def _perturb_features(self, feat: Tensor) -> Tensor:
        """特征随机扰动"""
        noise_mask = torch.rand_like(feat) < self.feat_ratio
        noise = torch.empty_like(feat).uniform_(self.feat_min, self.feat_max)
        return torch.where(noise_mask, noise, feat)

    def _perturb_edges(self, graph: dgl.DGLGraph) -> (dgl.DGLGraph, torch.Tensor):
        """边添加扰动（不删除边）"""
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
        """节点注入（含边掩码跟踪）"""
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
        """单图扰动逻辑"""
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
        """批量扰动方法"""
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
        """获取设备信息"""
        return next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cpu')
