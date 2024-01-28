import numpy as np
import yaml
import argparse
from models import *
from utils import *
from load_dataset import load_dataset
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

mpl.use('TkAgg')

dataset = 'amazon_photo'
# dataset = 'amazon_cs'
# dataset = 'coauthor_phy'
# dataset = 'coauthor_cs'
# dataset = 'pubmed'
# dataset = 'ogbn-arxiv'

exp = 'GAT'
with open(f"../{exp}/optimized_hyperparameter_configurations/FGAI_{dataset}.yml", 'r') as file:
    args = yaml.safe_load(file)
args = argparse.Namespace(**args)
args.device = 'cpu'

g, adj, feats, label, train_idx, valid_idx, test_idx, num_classes = load_dataset(args)
in_feats = feats.shape[1]
g = dgl.from_scipy(adj).to(args.device)

vanilla = GATNodeClassifier(
    feats_size=in_feats,
    hidden_size=args.hid_dim,
    out_size=num_classes,
    n_layers=args.n_layers,
    n_heads=args.n_heads,
    feat_drop=args.feat_drop,
    attn_drop=args.attn_drop
).to(args.device)
FGAI = GATNodeClassifier(
    feats_size=in_feats,
    hidden_size=args.hid_dim,
    out_size=num_classes,
    n_layers=args.n_layers,
    n_heads=args.n_heads,
    feat_drop=args.feat_drop,
    attn_drop=args.attn_drop
).to(args.device)

tim1 = '_12-07'
tim2 = '_01-23_12-43'

vanilla.load_state_dict(torch.load(f'../{exp}/vanilla_checkpoints/{dataset}{tim1}/model_parameters.pth'))
FGAI.load_state_dict(torch.load(f'../{exp}/FGAI_checkpoints/{dataset}{tim2}/FGAI_parameters.pth'))

orig_outputs, _, orig_att = evaluate_node_level(vanilla, feats, adj, label, test_idx, num_classes == 2)
pred = torch.argmax(orig_outputs[test_idx], dim=1)
accuracy = accuracy_score(label[test_idx].cpu(), pred.cpu())
print(f"vanilla accuracy: {accuracy:.4f}")
FGAI_outputs, _, FGAI_att = evaluate_node_level(FGAI, feats, adj, label, test_idx, num_classes == 2)
pred = torch.argmax(FGAI_outputs[test_idx], dim=1)
accuracy = accuracy_score(label[test_idx].cpu(), pred.cpu())
print(f"FGAI accuracy: {accuracy:.4f}")

adj_perturbed = sp.load_npz(f'../{exp}/vanilla_checkpoints/{args.dataset}{tim1}/adj_delta.npz')
feats_perturbed = torch.load(f'../{exp}/vanilla_checkpoints/{args.dataset}{tim1}/feats_delta.pth').to(args.device)

vanilla.eval()
new_outputs, _, new_att = vanilla(torch.cat((feats, feats_perturbed), dim=0), adj_perturbed)
new_outputs, new_att = new_outputs[:FGAI_outputs.shape[0]], new_att[:FGAI_att.shape[0]]
pred = torch.argmax(new_outputs[test_idx], dim=1)
accuracy = accuracy_score(label[test_idx].cpu(), pred.cpu())
print(f"Vanilla accuracy after attack: {accuracy:.4f}")

FGAI.eval()
new_FGAI_outputs, _, new_FGAI_att = FGAI(torch.cat((feats, feats_perturbed), dim=0), adj_perturbed)
new_FGAI_outputs, new_FGAI_att = new_FGAI_outputs[:FGAI_outputs.shape[0]], new_FGAI_att[:FGAI_att.shape[0]]
FGAI_pred = torch.argmax(new_FGAI_outputs[test_idx], dim=1)
accuracy = accuracy_score(label[test_idx].cpu(), FGAI_pred.cpu())
print(f"FGAI accuracy after attack: {accuracy:.4f}")

src, dst = g.edges()
att_list, att_list_new = [], []
att_list_FGAI, att_list_new_FGAI = [], []
neighbor_list = []
indices_list = []
# count = 0
# for node_id in g_dgl.nodes():
#     neighbors = g_dgl.successors(node_id)
#     if len(neighbors) <= 50:
#         continue
#     count += 1
#     if count > 3:
#         break
#     neighbor_list.append(neighbors.numpy())
#     indices = np.where(src == node_id)[0]
#     indices_list.append(indices)
#     att_list.append(orig_att[indices])
#     att_list_new.append(new_att[indices].detach())
#     att_list_FGAI.append(FGAI_att[indices])
#     att_list_new_FGAI.append(new_FGAI_att[indices].detach())

# for node_ID in g_dgl.nodes():
#     neighbor_list = []
#     neighbors = g_dgl.successors(node_ID)
#     if len(neighbors) == 0:
#         continue
#     for node_id in neighbors:
#         neighbors = g_dgl.successors(node_id)
#         neighbor_list.append(neighbors.numpy())
#     neighbor_ids = np.concatenate(neighbor_list)
#     if 50 < len(neighbor_ids) < 80:
#         print(node_ID)
# exit()

# neighbors = g.successors(192)
# print(neighbors)
# print(len(neighbors))
#
# for node_id in neighbors:
#     neighbors = g.successors(node_id)
#     neighbor_list.append(neighbors.numpy())
#     indices = np.where(src == node_id)[0]
#     indices_list.append(indices)
#     att_list.append(orig_att[indices])
#     att_list_new.append(new_att[indices].detach())
#     att_list_FGAI.append(FGAI_att[indices])
#     att_list_new_FGAI.append(new_FGAI_att[indices].detach())
#     if len(att_list) >= 0:
#         break
#
# neighbor_ids = np.concatenate(neighbor_list)
# att_color = np.concatenate(att_list)
# att_color_new = np.concatenate(att_list_new)
# att_color_FGAI = np.concatenate(att_list_FGAI)
# att_color_new_FGAI = np.concatenate(att_list_new_FGAI)
# select_nodes = np.unique(neighbor_ids)
# select_edges = np.concatenate(indices_list)
#
# print(len(select_nodes))
# print(len(select_edges))
# # if len(select_nodes) > 80:
# #     exit()

top_values, top_indices = torch.topk(orig_att, k=20)
# select_edges = np.arange(50)
select_edges = top_indices.detach().cpu().numpy()
att_color = orig_att[select_edges].detach().cpu().numpy()
att_color_new = new_att[select_edges].detach().cpu().numpy()

# 创建图形和子图
fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])
# edge_cmap = plt.get_cmap("coolwarm")
# edge_cmap = edge_cmap.reversed()
# edge_cmap = plt.cm.Blues
edge_cmap = plt.cm.Reds
# edge_cmap = plt.cm.Greens
power_rate = 1 / 3

sub_g = dgl.edge_subgraph(g, select_edges)
sub_g = dgl.to_networkx(sub_g)
# pos = nx.random_layout(sub_g)
# pos = nx.circular_layout(sub_g)
# pos = nx.shell_layout(sub_g)
pos = nx.spring_layout(sub_g)
# pos = nx.spectral_layout(sub_g)
# pos = nx.kamada_kawai_layout(sub_g)
# pos = nx.fruchterman_reingold_layout(sub_g)
# pos = nx.arf_layout(sub_g)
# pos = nx.planar_layout(sub_g)

ax1 = plt.subplot(gs[0])
colors = att_color / att_color.max()
# colors = np.power(colors, power_rate)
edges = nx.draw_networkx_edges(sub_g, pos=pos, edge_color=colors,
                               width=1.5, edge_cmap=edge_cmap, edge_vmin=0, alpha=0.9, ax=ax1)
nx.draw_networkx_nodes(sub_g, pos, nodelist=sub_g.nodes(), node_color='#5e86c1', alpha=0.95, node_size=100, ax=ax1)
# nx.draw_networkx_labels(g, pos, labels=node_labels, font_color='blue')
ax1.set_title("Original")

ax2 = plt.subplot(gs[1])
colors = att_color_new / att_color_new.max()
colors = np.power(colors, power_rate)
edges = nx.draw_networkx_edges(sub_g, pos=pos, edge_color='red',
                               width=1.5, edge_cmap=edge_cmap, edge_vmin=0, alpha=0.9, ax=ax2, label='1112')
sub_g = dgl.edge_subgraph(g, np.arange(20))
sub_g = dgl.to_networkx(sub_g)
edges = nx.draw_networkx_edges(sub_g, pos=pos, edge_color='green',
                               width=1.5, edge_cmap=None, edge_vmin=0, alpha=0.9, ax=ax2, label='111')
nx.draw_networkx_nodes(sub_g, pos, nodelist=sub_g.nodes(), node_color='#5e86c1', alpha=0.95, node_size=100, ax=ax2)
ax2.set_title("Perturbed")

cax = plt.subplot(gs[2])
pc = mpl.collections.PatchCollection([], cmap=edge_cmap)
pc.set_array(colors)
plt.colorbar(pc, cax=cax)

plt.tight_layout()
plt.savefig(f"./{exp}_{dataset}_vanilla.pdf", format="pdf")
plt.show()
