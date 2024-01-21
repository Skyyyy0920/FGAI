import math
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from tqdm import tqdm
from dgl import batch, ETYPE, khop_in_subgraph, to_homogeneous, to_heterogeneous, EID, NID
from dgl.convert import to_networkx
from dgl.subgraph import node_subgraph
from dgl.transforms.functional import remove_nodes


class MCTSNode:
    def __init__(self, nodes):
        self.nodes = nodes
        self.num_visit = 0
        self.total_reward = 0.0
        self.immediate_reward = 0.0
        self.children = []

    def __repr__(self):
        return str(self.nodes)


class SubgraphX(nn.Module):
    def __init__(
            self,
            model,
            num_hops,
            coef=10.0,
            high2low=True,
            num_child=12,
            num_rollouts=20,
            node_min=3,
            shapley_steps=100,
            log=False,
    ):
        super().__init__()
        self.num_hops = num_hops
        self.coef = coef
        self.high2low = high2low
        self.num_child = num_child
        self.num_rollouts = num_rollouts
        self.node_min = node_min
        self.shapley_steps = shapley_steps
        self.log = log

        self.model = model

    def shapley(self, subgraph_nodes):
        num_nodes = self.graph.num_nodes()
        subgraph_nodes = subgraph_nodes.tolist()

        # Obtain neighboring nodes of the subgraph g_i, P'.
        local_region = subgraph_nodes
        for _ in range(self.num_hops - 1):
            in_neighbors, _ = self.graph.in_edges(local_region)
            _, out_neighbors = self.graph.out_edges(local_region)
            neighbors = torch.cat([in_neighbors, out_neighbors]).tolist()
            local_region = list(set(local_region + neighbors))

        split_point = num_nodes
        coalition_space = list(set(local_region) - set(subgraph_nodes)) + [
            split_point
        ]

        marginal_contributions = []
        device = self.feat.device
        for _ in range(self.shapley_steps):
            permuted_space = np.random.permutation(coalition_space)
            split_idx = int(np.where(permuted_space == split_point)[0])

            selected_nodes = permuted_space[:split_idx]

            # Mask for coalition set S_i
            exclude_mask = torch.ones(num_nodes)
            exclude_mask[local_region] = 0.0
            exclude_mask[selected_nodes] = 1.0

            # Mask for set S_i and g_i
            include_mask = exclude_mask.clone()
            include_mask[subgraph_nodes] = 1.0

            exclude_feat = self.feat * exclude_mask.unsqueeze(1).to(device)
            include_feat = self.feat * include_mask.unsqueeze(1).to(device)

            with torch.no_grad():
                exclude_probs = self.model(
                    self.graph, exclude_feat, **self.kwargs
                ).softmax(dim=-1)
                exclude_value = exclude_probs[:, self.target_class]
                include_probs = self.model(
                    self.graph, include_feat, **self.kwargs
                ).softmax(dim=-1)
                include_value = include_probs[:, self.target_class]
            marginal_contributions.append(include_value - exclude_value)

        return torch.cat(marginal_contributions).mean().item()

    def get_mcts_children(self, mcts_node):
        if len(mcts_node.children) > 0:
            return mcts_node.children

        subg = node_subgraph(self.graph, mcts_node.nodes)
        node_degrees = subg.out_degrees() + subg.in_degrees()
        k = min(subg.num_nodes(), self.num_child)
        chosen_nodes = torch.topk(
            node_degrees, k, largest=self.high2low
        ).indices

        mcts_children_maps = dict()

        for node in chosen_nodes:
            new_subg = remove_nodes(subg, node.to(subg.idtype), store_ids=True)
            # Get the largest weakly connected component in the subgraph.
            nx_graph = to_networkx(new_subg.cpu())
            largest_cc_nids = list(
                max(nx.weakly_connected_components(nx_graph), key=len)
            )
            # Map to the original node IDs.
            largest_cc_nids = new_subg.ndata[NID][largest_cc_nids].long()
            largest_cc_nids = subg.ndata[NID][largest_cc_nids].sort().values
            if str(largest_cc_nids) not in self.mcts_node_maps:
                child_mcts_node = MCTSNode(largest_cc_nids)
                self.mcts_node_maps[str(child_mcts_node)] = child_mcts_node
            else:
                child_mcts_node = self.mcts_node_maps[str(largest_cc_nids)]

            if str(child_mcts_node) not in mcts_children_maps:
                mcts_children_maps[str(child_mcts_node)] = child_mcts_node

        mcts_node.children = list(mcts_children_maps.values())
        for child_mcts_node in mcts_node.children:
            if child_mcts_node.immediate_reward == 0:
                child_mcts_node.immediate_reward = self.shapley(
                    child_mcts_node.nodes
                )

        return mcts_node.children

    def mcts_rollout(self, mcts_node):
        if len(mcts_node.nodes) <= self.node_min:
            return mcts_node.immediate_reward

        children_nodes = self.get_mcts_children(mcts_node)
        children_visit_sum = sum([child.num_visit for child in children_nodes])
        children_visit_sum_sqrt = math.sqrt(children_visit_sum)
        chosen_child = max(
            children_nodes,
            key=lambda c: c.total_reward / max(c.num_visit, 1)
                          + self.coef
                          * c.immediate_reward
                          * children_visit_sum_sqrt
                          / (1 + c.num_visit),
        )
        reward = self.mcts_rollout(chosen_child)
        chosen_child.num_visit += 1
        chosen_child.total_reward += reward

        return reward

    def explain_graph(self, graph, feat, target_class, **kwargs):
        self.model.eval()
        assert (
                graph.num_nodes() > self.node_min
        ), f"The number of nodes in the\
            graph {graph.num_nodes()} should be bigger than {self.node_min}."

        self.graph = graph
        self.feat = feat
        self.target_class = target_class
        self.kwargs = kwargs

        # book all nodes in MCTS
        self.mcts_node_maps = dict()

        root = MCTSNode(graph.nodes())
        self.mcts_node_maps[str(root)] = root

        for i in range(self.num_rollouts):
            if self.log:
                print(
                    f"Rollout {i}/{self.num_rollouts}, \
                    {len(self.mcts_node_maps)} subgraphs have been explored."
                )
            self.mcts_rollout(root)

        best_leaf = None
        best_immediate_reward = float("-inf")
        for mcts_node in self.mcts_node_maps.values():
            if len(mcts_node.nodes) > self.node_min:
                continue

            if mcts_node.immediate_reward > best_immediate_reward:
                best_leaf = mcts_node
                best_immediate_reward = best_leaf.immediate_reward

        return best_leaf.nodes


class HeteroSubgraphX(nn.Module):
    def __init__(
            self,
            model,
            num_hops,
            coef=10.0,
            high2low=True,
            num_child=12,
            num_rollouts=20,
            node_min=3,
            shapley_steps=100,
            log=False,
    ):
        super().__init__()
        self.num_hops = num_hops
        self.coef = coef
        self.high2low = high2low
        self.num_child = num_child
        self.num_rollouts = num_rollouts
        self.node_min = node_min
        self.shapley_steps = shapley_steps
        self.log = log

        self.model = model

    def shapley(self, subgraph_nodes):
        # Obtain neighboring nodes of the subgraph g_i, P'.
        local_regions = {
            ntype: nodes.tolist() for ntype, nodes in subgraph_nodes.items()
        }
        for _ in range(self.num_hops - 1):
            for c_etype in self.graph.canonical_etypes:
                src_ntype, _, dst_ntype = c_etype
                if (
                        src_ntype not in local_regions
                        or dst_ntype not in local_regions
                ):
                    continue

                in_neighbors, _ = self.graph.in_edges(
                    local_regions[dst_ntype], etype=c_etype
                )
                _, out_neighbors = self.graph.out_edges(
                    local_regions[src_ntype], etype=c_etype
                )
                local_regions[src_ntype] = list(
                    set(local_regions[src_ntype] + in_neighbors.tolist())
                )
                local_regions[dst_ntype] = list(
                    set(local_regions[dst_ntype] + out_neighbors.tolist())
                )

        split_point = self.graph.num_nodes()
        coalition_space = {
            ntype: list(
                set(local_regions[ntype]) - set(subgraph_nodes[ntype].tolist())
            )
                   + [split_point]
            for ntype in subgraph_nodes.keys()
        }

        marginal_contributions = []
        for _ in range(self.shapley_steps):
            selected_node_map = dict()
            for ntype, nodes in coalition_space.items():
                permuted_space = np.random.permutation(nodes)
                split_idx = int(np.where(permuted_space == split_point)[0])
                selected_node_map[ntype] = permuted_space[:split_idx]

            # Mask for coalition set S_i
            exclude_mask = {
                ntype: torch.ones(self.graph.num_nodes(ntype))
                for ntype in self.graph.ntypes
            }
            for ntype, region in local_regions.items():
                exclude_mask[ntype][region] = 0.0
            for ntype, selected_nodes in selected_node_map.items():
                exclude_mask[ntype][selected_nodes] = 1.0

            # Mask for set S_i and g_i
            include_mask = {
                ntype: exclude_mask[ntype].clone()
                for ntype in self.graph.ntypes
            }
            for ntype, subgn in subgraph_nodes.items():
                exclude_mask[ntype][subgn] = 1.0

            exclude_feat = {
                ntype: self.feat[ntype]
                       * exclude_mask[ntype].unsqueeze(1).to(self.feat[ntype].device)
                for ntype in self.graph.ntypes
            }
            include_feat = {
                ntype: self.feat[ntype]
                       * include_mask[ntype].unsqueeze(1).to(self.feat[ntype].device)
                for ntype in self.graph.ntypes
            }

            with torch.no_grad():
                exclude_probs = self.model(
                    self.graph, exclude_feat, **self.kwargs
                ).softmax(dim=-1)
                exclude_value = exclude_probs[:, self.target_class]
                include_probs = self.model(
                    self.graph, include_feat, **self.kwargs
                ).softmax(dim=-1)
                include_value = include_probs[:, self.target_class]
            marginal_contributions.append(include_value - exclude_value)

        return torch.cat(marginal_contributions).mean().item()

    def get_mcts_children(self, mcts_node):
        if len(mcts_node.children) > 0:
            return mcts_node.children

        subg = node_subgraph(self.graph, mcts_node.nodes)
        # Choose k nodes based on the highest degree in the subgraph
        node_degrees_map = {
            ntype: torch.zeros(
                subg.num_nodes(ntype), device=subg.nodes(ntype).device
            )
            for ntype in subg.ntypes
        }
        for c_etype in subg.canonical_etypes:
            src_ntype, _, dst_ntype = c_etype
            node_degrees_map[src_ntype] += subg.out_degrees(etype=c_etype)
            node_degrees_map[dst_ntype] += subg.in_degrees(etype=c_etype)

        node_degrees_list = [
            ((ntype, i), degree)
            for ntype, node_degrees in node_degrees_map.items()
            for i, degree in enumerate(node_degrees)
        ]
        node_degrees = torch.stack([v for _, v in node_degrees_list])
        k = min(subg.num_nodes(), self.num_child)
        chosen_node_indicies = torch.topk(
            node_degrees, k, largest=self.high2low
        ).indices
        chosen_nodes = [node_degrees_list[i][0] for i in chosen_node_indicies]

        mcts_children_maps = dict()

        for ntype, node in chosen_nodes:
            new_subg = remove_nodes(subg, node, ntype, store_ids=True)

            if new_subg.num_edges() > 0:
                new_subg_homo = to_homogeneous(new_subg)
                # Get the largest weakly connected component in the subgraph.
                nx_graph = to_networkx(new_subg_homo.cpu())
                largest_cc_nids = list(
                    max(nx.weakly_connected_components(nx_graph), key=len)
                )
                largest_cc_homo = node_subgraph(new_subg_homo, largest_cc_nids)
                largest_cc_hetero = to_heterogeneous(
                    largest_cc_homo, new_subg.ntypes, new_subg.etypes
                )

                # Follow steps for backtracking to original graph node ids
                # 1. retrieve instanced homograph from connected-component homograph
                # 2. retrieve instanced heterograph from instanced homograph
                # 3. retrieve hetero-subgraph from instanced heterograph
                # 4. retrieve orignal graph ids from subgraph node ids
                cc_nodes = {
                    ntype: subg.ndata[NID][ntype][
                        new_subg.ndata[NID][ntype][
                            new_subg_homo.ndata[NID][
                                largest_cc_homo.ndata[NID][indicies]
                            ]
                        ]
                    ]
                    for ntype, indicies in largest_cc_hetero.ndata[NID].items()
                }
            else:
                available_ntypes = [
                    ntype
                    for ntype in new_subg.ntypes
                    if new_subg.num_nodes(ntype) > 0
                ]
                chosen_ntype = np.random.choice(available_ntypes)
                # backtrack from subgraph node ids to entire graph
                chosen_node = subg.ndata[NID][chosen_ntype][
                    np.random.choice(new_subg.nodes[chosen_ntype].data[NID])
                ]
                cc_nodes = {
                    chosen_ntype: torch.tensor(
                        [chosen_node],
                        device=subg.device,
                    )
                }

            if str(cc_nodes) not in self.mcts_node_maps:
                child_mcts_node = MCTSNode(cc_nodes)
                self.mcts_node_maps[str(child_mcts_node)] = child_mcts_node
            else:
                child_mcts_node = self.mcts_node_maps[str(cc_nodes)]

            if str(child_mcts_node) not in mcts_children_maps:
                mcts_children_maps[str(child_mcts_node)] = child_mcts_node

        mcts_node.children = list(mcts_children_maps.values())
        for child_mcts_node in mcts_node.children:
            if child_mcts_node.immediate_reward == 0:
                child_mcts_node.immediate_reward = self.shapley(
                    child_mcts_node.nodes
                )

        return mcts_node.children

    def mcts_rollout(self, mcts_node):
        if (
                sum(len(nodes) for nodes in mcts_node.nodes.values())
                <= self.node_min
        ):
            return mcts_node.immediate_reward

        children_nodes = self.get_mcts_children(mcts_node)
        children_visit_sum = sum([child.num_visit for child in children_nodes])
        children_visit_sum_sqrt = math.sqrt(children_visit_sum)
        chosen_child = max(
            children_nodes,
            key=lambda c: c.total_reward / max(c.num_visit, 1)
                          + self.coef
                          * c.immediate_reward
                          * children_visit_sum_sqrt
                          / (1 + c.num_visit),
        )
        reward = self.mcts_rollout(chosen_child)
        chosen_child.num_visit += 1
        chosen_child.total_reward += reward

        return reward

    def explain_graph(self, graph, feat, target_class, **kwargs):
        self.model.eval()
        assert (
                graph.num_nodes() > self.node_min
        ), f"The number of nodes in the\
            graph {graph.num_nodes()} should be bigger than {self.node_min}."

        self.graph = graph
        self.feat = feat
        self.target_class = target_class
        self.kwargs = kwargs

        # book all nodes in MCTS
        self.mcts_node_maps = dict()

        root_dict = {ntype: graph.nodes(ntype) for ntype in graph.ntypes}
        root = MCTSNode(root_dict)
        self.mcts_node_maps[str(root)] = root

        for i in range(self.num_rollouts):
            if self.log:
                print(
                    f"Rollout {i}/{self.num_rollouts}, \
                    {len(self.mcts_node_maps)} subgraphs have been explored."
                )
            self.mcts_rollout(root)

        best_leaf = None
        best_immediate_reward = float("-inf")
        for mcts_node in self.mcts_node_maps.values():
            if len(mcts_node.nodes) > self.node_min:
                continue

            if mcts_node.immediate_reward > best_immediate_reward:
                best_leaf = mcts_node
                best_immediate_reward = best_leaf.immediate_reward

        return best_leaf.nodes


class PGExplainer(nn.Module):
    def __init__(
            self,
            model,
            num_features,
            num_hops=None,
            explain_graph=True,
            coff_budget=0.01,
            coff_connect=5e-4,
            sample_bias=0.0,
    ):
        super(PGExplainer, self).__init__()

        self.model = model
        self.graph_explanation = explain_graph
        # Node explanation requires additional self-embedding data.
        self.num_features = num_features * (2 if self.graph_explanation else 3)
        self.num_hops = num_hops

        # training hyperparameters for PGExplainer
        self.coff_budget = coff_budget
        self.coff_connect = coff_connect
        self.sample_bias = sample_bias

        self.init_bias = 0.0

        # Explanation network in PGExplainer
        self.elayers = nn.Sequential(
            nn.Linear(self.num_features, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def set_masks(self, graph, edge_mask=None):
        if edge_mask is None:
            num_nodes = graph.num_nodes()
            num_edges = graph.num_edges()

            init_bias = self.init_bias
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (2 * num_nodes)
            )
            self.edge_mask = torch.randn(num_edges) * std + init_bias
        else:
            self.edge_mask = edge_mask

        self.edge_mask = self.edge_mask.to(graph.device)

    def clear_masks(self):
        self.edge_mask = None

    def parameters(self):
        return self.elayers.parameters()

    def loss(self, prob, ori_pred):
        target_prob = prob.gather(-1, ori_pred.unsqueeze(-1))
        # 1e-6 added to prob to avoid taking the logarithm of zero
        target_prob += 1e-6
        # computing the log likelihood for a single prediction
        pred_loss = torch.mean(-torch.log(target_prob))

        # size
        edge_mask = self.sparse_mask_values
        if self.coff_budget <= 0:
            size_loss = self.coff_budget * torch.sum(edge_mask)
        else:
            size_loss = self.coff_budget * F.relu(
                torch.sum(edge_mask) - self.coff_budget
            )

        # entropy
        scale = 0.99
        edge_mask = self.edge_mask * (2 * scale - 1.0) + (1.0 - scale)
        mask_ent = -edge_mask * torch.log(edge_mask) - (
                1 - edge_mask
        ) * torch.log(1 - edge_mask)
        mask_ent_loss = self.coff_connect * torch.mean(mask_ent)

        loss = pred_loss + size_loss + mask_ent_loss
        return loss

    def concrete_sample(self, w, beta=1.0, training=True):
        if training:
            bias = self.sample_bias
            random_noise = torch.rand(w.size()).to(w.device)
            random_noise = bias + (1 - 2 * bias) * random_noise
            gate_inputs = torch.log(random_noise) - torch.log(
                1.0 - random_noise
            )
            gate_inputs = (gate_inputs + w) / beta
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(w)

        return gate_inputs

    def train_step(self, graph, feat, temperature, **kwargs):
        assert (
            self.graph_explanation
        ), '"explain_graph" must be True when initializing the module.'

        self.model = self.model.to(graph.device)
        self.elayers = self.elayers.to(graph.device)

        pred = self.model(graph, feat, embed=False, **kwargs)
        pred = pred.argmax(-1).data

        prob, _ = self.explain_graph(
            graph, feat, temperature, training=True, **kwargs
        )

        loss = self.loss(prob, pred)
        return loss

    def train_step_node(self, nodes, graph, feat, temperature, **kwargs):
        assert (
            not self.graph_explanation
        ), '"explain_graph" must be False when initializing the module.'

        self.model = self.model.to(graph.device)
        self.elayers = self.elayers.to(graph.device)

        if isinstance(nodes, torch.Tensor):
            nodes = nodes.tolist()
        if isinstance(nodes, int):
            nodes = [nodes]

        prob, _, batched_graph, inverse_indices = self.explain_node(
            nodes, graph, feat, temperature, training=True, **kwargs
        )

        pred = self.model(batched_graph, self.batched_feats)
        pred = pred.argmax(-1).data

        loss = self.loss(prob[inverse_indices], pred[inverse_indices])
        return loss

    def explain_graph(
            self, graph, feat, temperature=1.0, training=False, **kwargs
    ):
        assert (
            self.graph_explanation
        ), '"explain_graph" must be True when initializing the module.'

        self.model = self.model.to(graph.device)
        self.elayers = self.elayers.to(graph.device)

        embed = self.model(graph, feat)
        embed = embed.data

        col, row = graph.edges()
        col_emb = embed[col.long()]
        row_emb = embed[row.long()]
        emb = torch.cat([col_emb, row_emb], dim=-1)
        emb = self.elayers(emb)
        values = emb.reshape(-1)

        values = self.concrete_sample(
            values, beta=temperature, training=training
        )
        self.sparse_mask_values = values

        reverse_eids = graph.edge_ids(row, col).long()
        edge_mask = (values + values[reverse_eids]) / 2

        self.set_masks(graph, edge_mask)

        # the model prediction with the updated edge mask
        logits = self.model(graph, feat)
        probs = F.softmax(logits, dim=-1)

        if training:
            probs = probs.data
        else:
            self.clear_masks()

        return (probs, edge_mask)

    def explain_node(self, nodes, graph, feat, temperature=1.0, training=False, **kwargs):
        assert (
            not self.graph_explanation
        ), '"explain_graph" must be False when initializing the module.'
        assert (
                self.num_hops is not None
        ), '"num_hops" must be provided when initializing the module.'

        if isinstance(nodes, torch.Tensor):
            nodes = nodes.tolist()
        if isinstance(nodes, int):
            nodes = [nodes]

        self.model = self.model.to(feat.device)
        self.elayers = self.elayers.to(feat.device)

        batched_graph = []
        batched_embed = []
        for node_id in nodes:
            sg, inverse_indices = khop_in_subgraph(graph, node_id, self.num_hops)
            sg.ndata["feat"] = feat[sg.ndata[NID].long()]
            sg.ndata["train"] = torch.tensor(
                [nid in inverse_indices for nid in sg.nodes()], device=sg.device
            )

            embed = self.model(sg, sg.ndata["feat"])
            embed = embed.data

            col, row = sg.edges()
            col_emb = embed[col.long()]
            row_emb = embed[row.long()]
            self_emb = embed[inverse_indices[0]].repeat(sg.num_edges(), 1)
            emb = torch.cat([col_emb, row_emb, self_emb], dim=-1)
            batched_embed.append(emb)
            batched_graph.append(sg)

        batched_graph = batch(batched_graph)

        batched_embed = torch.cat(batched_embed)
        batched_embed = self.elayers(batched_embed)
        values = batched_embed.reshape(-1)

        values = self.concrete_sample(
            values, beta=temperature, training=training
        )
        self.sparse_mask_values = values

        col, row = batched_graph.edges()
        reverse_eids = batched_graph.edge_ids(row, col).long()
        edge_mask = (values + values[reverse_eids]) / 2

        self.set_masks(batched_graph, edge_mask)

        batched_feats = batched_graph.ndata["feat"]
        # the model prediction with the updated edge mask
        logits = self.model(batched_graph, batched_feats)
        probs = F.softmax(logits, dim=-1)

        batched_inverse_indices = (
            batched_graph.ndata["train"].nonzero().squeeze(1)
        )

        if training:
            self.batched_feats = batched_feats
            probs = probs.data
        else:
            self.clear_masks()

        return (
            probs,
            edge_mask,
            batched_graph,
            batched_inverse_indices,
        )


class HeteroPGExplainer(PGExplainer):
    def train_step(self, graph, feat, temperature, **kwargs):
        # pylint: disable=useless-super-delegation
        return super().train_step(graph, feat, temperature, **kwargs)

    def train_step_node(self, nodes, graph, feat, temperature, **kwargs):
        assert (
            not self.graph_explanation
        ), '"explain_graph" must be False when initializing the module.'

        self.model = self.model.to(graph.device)
        self.elayers = self.elayers.to(graph.device)

        prob, _, batched_graph, inverse_indices = self.explain_node(
            nodes, graph, feat, temperature, training=True, **kwargs
        )

        pred = self.model(batched_graph, self.batched_feats)
        pred = {ntype: pred[ntype].argmax(-1).data for ntype in pred.keys()}

        loss = self.loss(
            torch.cat(
                [prob[ntype][nid] for ntype, nid in inverse_indices.items()]
            ),
            torch.cat(
                [pred[ntype][nid] for ntype, nid in inverse_indices.items()]
            ),
        )
        return loss

    def explain_graph(
            self, graph, feat, temperature=1.0, training=False, **kwargs
    ):
        assert (
            self.graph_explanation
        ), '"explain_graph" must be True when initializing the module.'

        self.model = self.model.to(graph.device)
        self.elayers = self.elayers.to(graph.device)

        embed = self.model(graph, feat)
        for ntype, emb in embed.items():
            graph.nodes[ntype].data["emb"] = emb.data
        homo_graph = to_homogeneous(graph, ndata=["emb"])
        homo_embed = homo_graph.ndata["emb"]

        col, row = homo_graph.edges()
        col_emb = homo_embed[col.long()]
        row_emb = homo_embed[row.long()]
        emb = torch.cat([col_emb, row_emb], dim=-1)
        emb = self.elayers(emb)
        values = emb.reshape(-1)

        values = self.concrete_sample(
            values, beta=temperature, training=training
        )
        self.sparse_mask_values = values

        reverse_eids = homo_graph.edge_ids(row, col).long()
        edge_mask = (values + values[reverse_eids]) / 2

        self.set_masks(homo_graph, edge_mask)

        # convert the edge mask back into heterogeneous format
        hetero_edge_mask = self._edge_mask_to_heterogeneous(
            edge_mask=edge_mask,
            homograph=homo_graph,
            heterograph=graph,
        )

        # the model prediction with the updated edge mask
        logits = self.model(graph, feat)
        probs = F.softmax(logits, dim=-1)

        if training:
            probs = probs.data
        else:
            self.clear_masks()

        return (probs, hetero_edge_mask)

    def explain_node(
            self, nodes, graph, feat, temperature=1.0, training=False, **kwargs
    ):
        assert (
            not self.graph_explanation
        ), '"explain_graph" must be False when initializing the module.'
        assert (
                self.num_hops is not None
        ), '"num_hops" must be provided when initializing the module.'

        self.model = self.model.to(graph.device)
        self.elayers = self.elayers.to(graph.device)

        batched_embed = []
        batched_homo_graph = []
        batched_hetero_graph = []
        for target_ntype, target_nids in nodes.items():
            if isinstance(target_nids, torch.Tensor):
                target_nids = target_nids.tolist()

            for target_nid in target_nids:
                sg, inverse_indices = khop_in_subgraph(
                    graph, {target_ntype: target_nid}, self.num_hops
                )

                for sg_ntype in sg.ntypes:
                    sg_feat = feat[sg_ntype][sg.ndata[NID][sg_ntype].long()]
                    train_mask = [
                        sg_ntype in inverse_indices
                        and node_id in inverse_indices[sg_ntype]
                        for node_id in sg.nodes(sg_ntype)
                    ]

                    sg.nodes[sg_ntype].data["feat"] = sg_feat
                    sg.nodes[sg_ntype].data["train"] = torch.tensor(
                        train_mask, device=sg.device
                    )

                embed = self.model(sg, sg.ndata["feat"])
                for ntype in embed.keys():
                    sg.nodes[ntype].data["emb"] = embed[ntype].data

                homo_sg = to_homogeneous(sg, ndata=["emb"])
                homo_sg_embed = homo_sg.ndata["emb"]

                col, row = homo_sg.edges()
                col_emb = homo_sg_embed[col.long()]
                row_emb = homo_sg_embed[row.long()]
                self_emb = homo_sg_embed[
                    inverse_indices[target_ntype][0]
                ].repeat(sg.num_edges(), 1)
                emb = torch.cat([col_emb, row_emb, self_emb], dim=-1)
                batched_embed.append(emb)
                batched_homo_graph.append(homo_sg)
                batched_hetero_graph.append(sg)

        batched_homo_graph = batch(batched_homo_graph)
        batched_hetero_graph = batch(batched_hetero_graph)

        batched_embed = torch.cat(batched_embed)
        batched_embed = self.elayers(batched_embed)
        values = batched_embed.reshape(-1)

        values = self.concrete_sample(
            values, beta=temperature, training=training
        )
        self.sparse_mask_values = values

        col, row = batched_homo_graph.edges()
        reverse_eids = batched_homo_graph.edge_ids(row, col).long()
        edge_mask = (values + values[reverse_eids]) / 2

        self.set_masks(batched_homo_graph, edge_mask)

        # Convert the edge mask back into heterogeneous format.
        hetero_edge_mask = self._edge_mask_to_heterogeneous(
            edge_mask=edge_mask,
            homograph=batched_homo_graph,
            heterograph=batched_hetero_graph,
        )

        batched_feats = {
            ntype: batched_hetero_graph.nodes[ntype].data["feat"]
            for ntype in batched_hetero_graph.ntypes
        }

        # The model prediction with the updated edge mask.
        logits = self.model(batched_hetero_graph, batched_feats)
        probs = {
            ntype: F.softmax(logits[ntype], dim=-1) for ntype in logits.keys()
        }

        batched_inverse_indices = {
            ntype: batched_hetero_graph.nodes[ntype]
            .data["train"]
            .nonzero()
            .squeeze(1)
            for ntype in batched_hetero_graph.ntypes
        }

        if training:
            self.batched_feats = batched_feats
            probs = {ntype: probs[ntype].data for ntype in probs.keys()}
        else:
            self.clear_masks()

        return (
            probs,
            hetero_edge_mask,
            batched_hetero_graph,
            batched_inverse_indices,
        )

    def _edge_mask_to_heterogeneous(self, edge_mask, homograph, heterograph):
        return {
            etype: edge_mask[
                (homograph.edata[ETYPE] == heterograph.get_etype_id(etype))
                .nonzero()
                .squeeze(1)
            ]
            for etype in heterograph.canonical_etypes
        }


class GNNExplainer(nn.Module):
    def __init__(
            self,
            model,
            num_hops,
            lr=0.01,
            num_epochs=100,
            *,
            alpha1=0.005,
            alpha2=1.0,
            beta1=1.0,
            beta2=0.1,
            log=True,
    ):
        super(GNNExplainer, self).__init__()
        self.model = model
        self.num_hops = num_hops
        self.lr = lr
        self.num_epochs = num_epochs
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2
        self.log = log

    def _init_masks(self, graph, feat):
        num_nodes, feat_size = feat.size()
        num_edges = graph.num_edges()
        device = feat.device

        std = 0.1
        feat_mask = nn.Parameter(torch.randn(1, feat_size, device=device) * std)

        std = nn.init.calculate_gain("relu") * sqrt(2.0 / (2 * num_nodes))
        edge_mask = nn.Parameter(torch.randn(num_edges, device=device) * std)

        return feat_mask, edge_mask

    def _loss_regularize(self, loss, feat_mask, edge_mask):
        # epsilon for numerical stability
        eps = 1e-15

        edge_mask = edge_mask.sigmoid()
        # Edge mask sparsity regularization
        loss = loss + self.alpha1 * torch.sum(edge_mask)
        # Edge mask entropy regularization
        ent = -edge_mask * torch.log(edge_mask + eps) - (
                1 - edge_mask
        ) * torch.log(1 - edge_mask + eps)
        loss = loss + self.alpha2 * ent.mean()

        feat_mask = feat_mask.sigmoid()
        # Feature mask sparsity regularization
        loss = loss + self.beta1 * torch.mean(feat_mask)
        # Feature mask entropy regularization
        ent = -feat_mask * torch.log(feat_mask + eps) - (
                1 - feat_mask
        ) * torch.log(1 - feat_mask + eps)
        loss = loss + self.beta2 * ent.mean()

        return loss

    def explain_node(self, node_id, graph, feat, **kwargs):
        self.model = self.model.to(graph.device)
        self.model.eval()
        num_nodes = graph.num_nodes()
        num_edges = graph.num_edges()

        # Extract node-centered k-hop subgraph and
        # its associated node and edge features.
        sg, inverse_indices = khop_in_subgraph(graph, node_id, self.num_hops)
        sg_nodes = sg.ndata[NID].long()
        sg_edges = sg.edata[EID].long()
        feat = feat[sg_nodes]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[sg_nodes]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[sg_edges]
            kwargs[key] = item

        # Get the initial prediction.
        with torch.no_grad():
            logits = self.model(graph=sg, feat=feat)
            pred_label = logits.argmax(dim=-1)

        feat_mask, edge_mask = self._init_masks(sg, feat)

        params = [feat_mask, edge_mask]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        if self.log:
            pbar = tqdm(total=self.num_epochs)
            pbar.set_description(f"Explain node {node_id}")

        for _ in range(self.num_epochs):
            optimizer.zero_grad()
            h = feat * feat_mask.sigmoid()
            logits = self.model(graph=sg, feat=h)
            log_probs = logits.log_softmax(dim=-1)
            loss = -log_probs[inverse_indices, pred_label[inverse_indices]]
            loss = self._loss_regularize(loss, feat_mask, edge_mask)
            loss.backward()
            optimizer.step()

            if self.log:
                pbar.update(1)

        if self.log:
            pbar.close()

        feat_mask = feat_mask.detach().sigmoid().squeeze()
        edge_mask = edge_mask.detach().sigmoid()

        return inverse_indices, sg, feat_mask, edge_mask

    def explain_graph(self, graph, feat, **kwargs):
        self.model = self.model.to(graph.device)
        self.model.eval()

        # Get the initial prediction.
        with torch.no_grad():
            logits = self.model(graph=graph, feat=feat)
            pred_label = logits.argmax(dim=-1)

        feat_mask, edge_mask = self._init_masks(graph, feat)

        params = [feat_mask, edge_mask]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        if self.log:
            pbar = tqdm(total=self.num_epochs)
            pbar.set_description("Explain graph")

        for _ in range(self.num_epochs):
            optimizer.zero_grad()
            h = feat * feat_mask.sigmoid()
            logits = self.model(graph=graph, feat=h)
            log_probs = logits.log_softmax(dim=-1)
            loss = -log_probs[0, pred_label[0]]
            loss = self._loss_regularize(loss, feat_mask, edge_mask)
            loss.backward()
            optimizer.step()

            if self.log:
                pbar.update(1)

        if self.log:
            pbar.close()

        feat_mask = feat_mask.detach().sigmoid().squeeze()
        edge_mask = edge_mask.detach().sigmoid()

        return feat_mask, edge_mask


class HeteroGNNExplainer(nn.Module):
    def __init__(
            self,
            model,
            num_hops,
            lr=0.01,
            num_epochs=100,
            *,
            alpha1=0.005,
            alpha2=1.0,
            beta1=1.0,
            beta2=0.1,
            log=True,
    ):
        super(HeteroGNNExplainer, self).__init__()
        self.model = model
        self.num_hops = num_hops
        self.lr = lr
        self.num_epochs = num_epochs
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2
        self.log = log

    def _init_masks(self, graph, feat):
        device = graph.device
        feat_masks = {}
        std = 0.1
        for node_type, feature in feat.items():
            _, feat_size = feature.size()
            feat_masks[node_type] = nn.Parameter(
                torch.randn(1, feat_size, device=device) * std
            )

        edge_masks = {}
        for canonical_etype in graph.canonical_etypes:
            src_num_nodes = graph.num_nodes(canonical_etype[0])
            dst_num_nodes = graph.num_nodes(canonical_etype[-1])
            num_nodes_sum = src_num_nodes + dst_num_nodes
            num_edges = graph.num_edges(canonical_etype)
            std = nn.init.calculate_gain("relu")
            if num_nodes_sum > 0:
                std *= sqrt(2.0 / num_nodes_sum)
            edge_masks[canonical_etype] = nn.Parameter(
                torch.randn(num_edges, device=device) * std
            )

        return feat_masks, edge_masks

    def _loss_regularize(self, loss, feat_masks, edge_masks):
        # epsilon for numerical stability
        eps = 1e-15

        for edge_mask in edge_masks.values():
            edge_mask = edge_mask.sigmoid()
            # Edge mask sparsity regularization
            loss = loss + self.alpha1 * torch.sum(edge_mask)
            # Edge mask entropy regularization
            ent = -edge_mask * torch.log(edge_mask + eps) - (
                    1 - edge_mask
            ) * torch.log(1 - edge_mask + eps)
            loss = loss + self.alpha2 * ent.mean()

        for feat_mask in feat_masks.values():
            feat_mask = feat_mask.sigmoid()
            # Feature mask sparsity regularization
            loss = loss + self.beta1 * torch.mean(feat_mask)
            # Feature mask entropy regularization
            ent = -feat_mask * torch.log(feat_mask + eps) - (
                    1 - feat_mask
            ) * torch.log(1 - feat_mask + eps)
            loss = loss + self.beta2 * ent.mean()

        return loss

    def explain_node(self, ntype, node_id, graph, feat, **kwargs):
        self.model = self.model.to(graph.device)
        self.model.eval()

        # Extract node-centered k-hop subgraph and
        # its associated node and edge features.
        sg, inverse_indices = khop_in_subgraph(
            graph, {ntype: node_id}, self.num_hops
        )
        inverse_indices = inverse_indices[ntype]
        sg_nodes = sg.ndata[NID]
        sg_feat = {}

        for node_type in sg_nodes.keys():
            sg_feat[node_type] = feat[node_type][sg_nodes[node_type].long()]

        # Get the initial prediction.
        with torch.no_grad():
            logits = self.model(graph=sg, feat=sg_feat)[ntype]
            pred_label = logits.argmax(dim=-1)

        feat_mask, edge_mask = self._init_masks(sg, sg_feat)

        params = [*feat_mask.values(), *edge_mask.values()]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        if self.log:
            pbar = tqdm(total=self.num_epochs)
            pbar.set_description(f"Explain node {node_id} with type {ntype}")

        for _ in range(self.num_epochs):
            optimizer.zero_grad()
            h = {}
            for node_type, sg_node_feat in sg_feat.items():
                h[node_type] = sg_node_feat * feat_mask[node_type].sigmoid()
            eweight = {}
            for canonical_etype, canonical_etype_mask in edge_mask.items():
                eweight[canonical_etype] = canonical_etype_mask.sigmoid()
            logits = self.model(graph=sg, feat=h, eweight=eweight, **kwargs)[
                ntype
            ]
            log_probs = logits.log_softmax(dim=-1)
            loss = -log_probs[inverse_indices, pred_label[inverse_indices]]
            loss = self._loss_regularize(loss, feat_mask, edge_mask)
            loss.backward()
            optimizer.step()

            if self.log:
                pbar.update(1)

        if self.log:
            pbar.close()

        for node_type in feat_mask:
            feat_mask[node_type] = (
                feat_mask[node_type].detach().sigmoid().squeeze()
            )

        for canonical_etype in edge_mask:
            edge_mask[canonical_etype] = (
                edge_mask[canonical_etype].detach().sigmoid()
            )

        return inverse_indices, sg, feat_mask, edge_mask

    def explain_graph(self, graph, feat, **kwargs):
        self.model = self.model.to(graph.device)
        self.model.eval()

        # Get the initial prediction.
        with torch.no_grad():
            logits = self.model(graph=graph, feat=feat, **kwargs)
            pred_label = logits.argmax(dim=-1)

        feat_mask, edge_mask = self._init_masks(graph, feat)

        params = [*feat_mask.values(), *edge_mask.values()]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        if self.log:
            pbar = tqdm(total=self.num_epochs)
            pbar.set_description("Explain graph")

        for _ in range(self.num_epochs):
            optimizer.zero_grad()
            h = {}
            for node_type, node_feat in feat.items():
                h[node_type] = node_feat * feat_mask[node_type].sigmoid()
            eweight = {}
            for canonical_etype, canonical_etype_mask in edge_mask.items():
                eweight[canonical_etype] = canonical_etype_mask.sigmoid()
            logits = self.model(graph=graph, feat=h, eweight=eweight, **kwargs)
            log_probs = logits.log_softmax(dim=-1)
            loss = -log_probs[0, pred_label[0]]
            loss = self._loss_regularize(loss, feat_mask, edge_mask)
            loss.backward()
            optimizer.step()

            if self.log:
                pbar.update(1)

        if self.log:
            pbar.close()

        for node_type in feat_mask:
            feat_mask[node_type] = (
                feat_mask[node_type].detach().sigmoid().squeeze()
            )

        for canonical_etype in edge_mask:
            edge_mask[canonical_etype] = (
                edge_mask[canonical_etype].detach().sigmoid()
            )

        return feat_mask, edge_mask
