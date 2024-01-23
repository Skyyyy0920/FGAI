import yaml
import argparse
from models import *
from utils import *
from explainer import *
from load_dataset import load_dataset
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

# dataset = 'amazon_photo'
# dataset = 'amazon_cs'
dataset = 'pubmed'
# dataset = 'coauthor_phy'
# dataset='coauthor_cs'

exp = 'GAT'
with open(f"./{exp}/optimized_hyperparameter_configurations/FGAI_{dataset}.yml", 'r') as file:
    args = yaml.safe_load(file)
args = argparse.Namespace(**args)
args.device = 'cpu'

g, adj, features, label, train_idx, valid_idx, test_idx, num_classes = load_dataset(args)
in_feats = features.shape[1]
g_dgl = dgl.from_scipy(adj).to(args.device)

vanilla = GATNodeClassifier(feats_size=in_feats,
                            hidden_size=args.hid_dim,
                            out_size=num_classes,
                            n_layers=args.n_layers,
                            n_heads=args.n_heads,
                            feat_drop=args.feat_drop,
                            attn_drop=args.attn_drop).to(args.device)
FGAI = GATNodeClassifier(feats_size=in_feats,
                         hidden_size=args.hid_dim,
                         out_size=num_classes,
                         n_layers=args.n_layers,
                         n_heads=args.n_heads,
                         feat_drop=args.feat_drop,
                         attn_drop=args.attn_drop).to(args.device)

tim1 = '_10-38'
tim2 = '_10-06_10-41'

vanilla.load_state_dict(torch.load(f'./{exp}/GAT_checkpoints/{dataset}{tim1}/model_parameters.pth'))
FGAI.load_state_dict(torch.load(f'./{exp}/FGAI_checkpoints/{dataset}{tim2}/FGAI_parameters.pth'))

adj_perturbed = sp.load_npz(f'./{exp}/GAT_checkpoints/{args.dataset}{tim1}/adj_delta.npz')
feats_perturbed = torch.load(f'./{exp}/GAT_checkpoints/{args.dataset}{tim1}/feats_delta.pth').to(args.device)

model_for_explain = ModelForExplain(vanilla)

GNN_expl = GNNExplainer(model_for_explain, num_hops=1)
new_center, sg, feat_mask, edge_mask = GNN_expl.explain_node(0, g, features)
print(new_center, sg, feat_mask, edge_mask)

# # Initialize the explainer
# PGexpl = PGExplainer(model_for_explain, num_classes, num_hops=2, explain_graph=False)
#
# # Train the explainer
# # Define explainer temperature parameter
# init_tmp, final_tmp = 5.0, 1.0
# optimizer_exp = torch.optim.Adam(PGexpl.parameters(), lr=0.01)
# epochs = 10
# for epoch in range(epochs):
#     tmp = float(init_tmp * np.power(final_tmp / init_tmp, epoch / epochs))
#     loss = PGexpl.train_step_node(g.nodes(), g, features, tmp)
#     optimizer_exp.zero_grad()
#     loss.backward()
#     optimizer_exp.step()
#
# # Explain the prediction for graph 0
# probs, edge_weight, bg, inverse_indices = PGexpl.explain_node(0, g, features)
# print(probs)




exit()



# for node_ID in g_dgl.nodes():
#     neighbor_list = []
#     neighbors = g_dgl.successors(node_ID)
#     if 30 < len(neighbors) < 100:
#         print(node_ID)
# exit()

src, dst = g_dgl.edges()
att_list, att_list_new = [], []
att_list_FGAI, att_list_new_FGAI = [], []
neighbor_list = []

# node_id = 18811
# node_id = 16462
node_id = 735
neighbors = g_dgl.successors(node_id)
neighbor_list.append(neighbors.numpy())
indices = np.where(src == node_id)[0]

kkk = 25
topk_values_orig, topk_indices_orig = torch.topk(orig_att[indices], k=kkk, largest=True)
topk_values_new, topk_indices_new = torch.topk(new_att[indices], k=kkk, largest=True)
topk_values_FGAI, topk_indices_FGAI = torch.topk(FGAI_att[indices], k=kkk, largest=True)
topk_values_FGAI_new, topk_indices_FGAI_new = torch.topk(new_FGAI_att[indices], k=kkk, largest=True)

result = np.zeros(kkk)
for i in range(len(topk_indices_new)):
    if topk_indices_new[i] in topk_indices_orig:
        result[i] = 1
att_list.append(np.ones(kkk))
att_list_new.append(result)

result = np.zeros(kkk)
for i in range(len(topk_indices_FGAI_new)):
    if topk_indices_FGAI_new[i] in topk_indices_FGAI:
        result[i] = 1
att_list_FGAI.append(np.ones(kkk))
att_list_new_FGAI.append(result)
# att_list.append(ranking_value(orig_att[indices]))
# att_list_new.append(ranking_value(new_att[indices].detach()))
# att_list_FGAI.append(ranking_value(FGAI_att[indices]))
# att_list_new_FGAI.append(ranking_value(new_FGAI_att[indices].detach()))

neighbor_ids = np.concatenate(neighbor_list)
att_color = np.concatenate(att_list)
att_color_new = np.concatenate(att_list_new)
att_color_FGAI = np.concatenate(att_list_FGAI)
att_color_new_FGAI = np.concatenate(att_list_new_FGAI)

fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
# edge_cmap = plt.get_cmap("coolwarm")
# edge_cmap = edge_cmap.reversed()
# edge_cmap = plt.cm.Blues
edge_cmap = plt.cm.Reds
# edge_cmap = plt.cm.Greens
power_rate = 1

select_nodes = np.unique(neighbor_ids)
indices_list = []
indices_list.append(topk_indices_orig)
select_edges = np.concatenate(indices_list)

sub_g = dgl.edge_subgraph(g_dgl, select_edges)
sub_g = dgl.to_networkx(sub_g)
# pos = nx.random_layout(sub_g)
pos = nx.circular_layout(sub_g)

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

indices_list = []
indices_list.append(topk_indices_FGAI)
select_edges = np.concatenate(indices_list)

sub_g = dgl.edge_subgraph(g_dgl, select_edges)
sub_g = dgl.to_networkx(sub_g)
# pos = nx.random_layout(sub_g)
pos = nx.circular_layout(sub_g)

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
