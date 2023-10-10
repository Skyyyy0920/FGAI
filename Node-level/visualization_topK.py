import dgl
import yaml
import argparse
from models import GATNodeClassifier
from utils import *
from load_dataset import load_dataset
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec


def ranking_value(orig_att):
    K = int(len(orig_att) * 0.4)
    sorted_indices = torch.argsort(orig_att, descending=True)
    ranks = torch.arange(1, len(orig_att) + 1, dtype=torch.float32)
    discrete_values = (len(orig_att) - ranks + 1) / len(orig_att)
    discrete_values[ranks > K] = 0
    discrete_values[ranks <= K] = 1
    sorted_discrete_values = discrete_values[sorted_indices]

    return sorted_discrete_values


dataset = 'amazon_photo'
# dataset = 'amazon_cs'
# dataset='coauthor_cs'
# dataset = 'coauthor_phy'

exp = 'GAT'
with open(f"./{exp}/optimized_hyperparameter_configurations/FGAI_{dataset}.yml", 'r') as file:
    args = yaml.safe_load(file)
args = argparse.Namespace(**args)
args.device = 'cpu'

adj, feats, label, train_idx, valid_idx, test_idx, num_classes = load_dataset(args)
in_feats = feats.shape[1]
g_dgl = dgl.from_scipy(adj).to(args.device)

vanilla = GATNodeClassifier(in_feats=in_feats,
                            hid_dim=args.hid_dim,
                            n_classes=num_classes,
                            n_layers=args.n_layers,
                            n_heads=args.n_heads,
                            feat_drop=args.feat_drop,
                            attn_drop=args.attn_drop).to(args.device)
FGAI = GATNodeClassifier(in_feats=in_feats,
                         hid_dim=args.hid_dim,
                         n_classes=num_classes,
                         n_layers=args.n_layers,
                         n_heads=args.n_heads,
                         feat_drop=args.feat_drop,
                         attn_drop=args.attn_drop).to(args.device)

tim1 = '_10-54'
tim2 = '_10-06_11-18'

vanilla.load_state_dict(torch.load(f'./{exp}/GAT_checkpoints/{dataset}{tim1}/model_parameters.pth'))
FGAI.load_state_dict(torch.load(f'./{exp}/FGAI_checkpoints/{dataset}{tim2}/FGAI_parameters.pth'))

orig_outputs, _, orig_att = evaluate_node_level(vanilla, feats, adj, label, test_idx, num_classes == 2)
pred = torch.argmax(orig_outputs[test_idx], dim=1)
accuracy = accuracy_score(label[test_idx].cpu(), pred.cpu())
print(f"vanilla accuracy: {accuracy:.4f}")
FGAI_outputs, _, FGAI_att = evaluate_node_level(FGAI, feats, adj, label, test_idx, num_classes == 2)
pred = torch.argmax(FGAI_outputs[test_idx], dim=1)
accuracy = accuracy_score(label[test_idx].cpu(), pred.cpu())
print(f"FGAI accuracy: {accuracy:.4f}")

adj_perturbed = sp.load_npz(f'./{exp}/GAT_checkpoints/{args.dataset}{tim1}/adj_delta.npz')
feats_perturbed = torch.load(f'./{exp}/GAT_checkpoints/{args.dataset}{tim1}/feats_delta.pth').to(args.device)

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

topk_values_orig, topk_indices_orig = torch.topk(orig_att, k=100, largest=True)
topk_values_new, topk_indices_new = torch.topk(new_att, k=100, largest=True)
topk_values_FGAI, topk_indices_FGAI = torch.topk(FGAI_att, k=100, largest=True)
topk_values_FGAI_new, topk_indices_FGAI_new = torch.topk(new_FGAI_att, k=100, largest=True)

src, dst = g_dgl.edges()
att_list, att_list_new = [], []
att_list_FGAI, att_list_new_FGAI = [], []
neighbor_list_orig = []
indices_list = []

neighbor_list_orig.append(src[topk_indices_orig])
neighbor_list_orig.append(dst[topk_indices_orig])

node_id = 6666
neighbors = g_dgl.successors(node_id)
neighbor_list_orig.append(neighbors.numpy())
indices = np.where(src == node_id)[0]
indices_list.append(indices)
att_list.append(ranking_value(orig_att[indices]))
att_list_new.append(ranking_value(new_att[indices].detach()))
att_list_FGAI.append(ranking_value(FGAI_att[indices]))
att_list_new_FGAI.append(ranking_value(new_FGAI_att[indices].detach()))

neighbor_ids_orig = np.concatenate(neighbor_list_orig)
att_color = np.concatenate(att_list)
att_color_new = np.concatenate(att_list_new)
att_color_FGAI = np.concatenate(att_list_FGAI)
att_color_new_FGAI = np.concatenate(att_list_new_FGAI)
select_nodes = np.unique(neighbor_ids_orig)
select_edges = np.concatenate(indices_list)

fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
# edge_cmap = plt.get_cmap("coolwarm")
# edge_cmap = edge_cmap.reversed()
# edge_cmap = plt.cm.Blues
edge_cmap = plt.cm.Reds
# edge_cmap = plt.cm.Greens
power_rate = 1

sub_g = dgl.edge_subgraph(g_dgl, select_edges)
sub_g = dgl.to_networkx(sub_g)
pos = nx.random_layout(sub_g)

ax1 = plt.subplot(gs[0])
colors = att_color
edges = nx.draw_networkx_edges(sub_g, pos=pos, edge_color=colors,
                               width=1.5, edge_cmap=edge_cmap, edge_vmin=0, alpha=0.9, ax=ax1)
nx.draw_networkx_nodes(sub_g, pos, nodelist=sub_g.nodes(), node_color='#5e86c1', alpha=0.95, node_size=125, ax=ax1)
ax1.set_title("Original")

ax2 = plt.subplot(gs[1])
colors = att_color_new
edges = nx.draw_networkx_edges(sub_g, pos=pos, edge_color=colors,
                               width=1.5, edge_cmap=edge_cmap, edge_vmin=0, alpha=0.9, ax=ax2)
nx.draw_networkx_nodes(sub_g, pos, nodelist=sub_g.nodes(), node_color='#5e86c1', alpha=0.95, node_size=125, ax=ax2)
ax2.set_title("Perturbed")

plt.tight_layout()
plt.savefig(f"./visualization_results/{exp}_{dataset}_vanilla_topK.pdf", format="pdf")
plt.show()

fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

ax1 = plt.subplot(gs[0])
colors = att_color_FGAI
edges = nx.draw_networkx_edges(sub_g, pos=pos, edge_color=colors,
                               width=1.5, edge_cmap=edge_cmap, edge_vmin=0, alpha=0.9, ax=ax1)
nx.draw_networkx_nodes(sub_g, pos, nodelist=sub_g.nodes(), node_color='#5e86c1', alpha=0.95, node_size=125, ax=ax1)
ax1.set_title("Original")

ax2 = plt.subplot(gs[1])
colors = att_color_new_FGAI
edges = nx.draw_networkx_edges(sub_g, pos=pos, edge_color=colors,
                               width=1.5, edge_cmap=edge_cmap, edge_vmin=0, alpha=0.9, ax=ax2)
nx.draw_networkx_nodes(sub_g, pos, nodelist=sub_g.nodes(), node_color='#5e86c1', alpha=0.95, node_size=125, ax=ax2)
ax2.set_title("Perturbed")

plt.tight_layout()
plt.savefig(f"./visualization_results/{exp}_{dataset}_FGAI_topK.pdf", format="pdf")
plt.show()
