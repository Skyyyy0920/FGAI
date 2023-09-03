from model import *
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# u = torch.tensor([0, 0, 0, 0, 0])
# v = torch.tensor([1, 2, 3, 4, 5])
# g1 = dgl.DGLGraph((u, v))
# print(g1.adjacency_matrix())
# print(g1.adjacency_matrix().to_dense())
#
# # Example adjacency matrix
# adj = np.array([[0, 1, 1, 1, 0, 0, 0],
#                 [1, 0, 1, 1, 1, 0, 0],
#                 [1, 1, 0, 1, 0, 0, 0],
#                 [1, 1, 1, 0, 1, 0, 0],
#                 [0, 1, 0, 1, 0, 1, 0],
#                 [0, 0, 0, 0, 1, 0, 1],
#                 [0, 0, 0, 0, 0, 1, 0]])
#
# k_values = k_shell_algorithm(adj)
# print("Node K-values:", k_values)

# dataset = dgl.data.TUDataset('DD')
# dataset = dgl.data.GINDataset('MUTAG', self_loop=False, degree_as_nlabel=True)
dataset = dgl.data.GINDataset('MUTAG', False)
# train_dataset, train_labels = zip(*[dataset[i] for i in range(split)])
train_dataloader = GraphDataLoader(dataset, batch_size=1, drop_last=False, shuffle=True)
test_dataloader = GraphDataLoader(dataset, batch_size=1, drop_last=False, shuffle=False)

# 参数设置
in_feats = dataset[0][0].ndata['attr'].size(1)
hidden_feats = 32
num_classes = dataset.num_classes
num_heads = 1
learning_rate = 0.001
num_epochs = 50  # 500
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(device)

# model = GATGraphClassifier(in_feats, hidden_feats, num_classes, 0.3, 0.1, num_heads, 'top-k').to(device)
model = GATGraphClassifier(in_feats, hidden_feats, num_classes, 0.3, 0.1, num_heads, 'mean').to(device)
# model = GraphClassifierExample(7, 20, 5)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
for epoch in range(num_epochs):
    loss_list = []
    for batched_graph, labels in train_dataloader:
        feats = batched_graph.ndata['attr']
        # logits = model(batched_graph, feats)
        logits = model(feats, batched_graph.adjacency_matrix().to_dense())
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {np.mean(loss_list):.4f}")

model.eval()
with torch.no_grad():
    pred_list, label_list = [], []
    for batched_graph, labels in test_dataloader:
        feats = batched_graph.ndata['attr']
        logits = model(feats, batched_graph.adjacency_matrix().to_dense())
        predicted = logits.argmax(dim=1)
        accuracy = (predicted == labels).float().mean().item()
        print(predicted, labels)
        pred_list = pred_list + predicted.tolist()
        label_list = label_list + labels.tolist()

    accuracy = accuracy_score(label_list, pred_list)
    precision = precision_score(label_list, pred_list)
    recall = recall_score(label_list, pred_list)
    f1 = f1_score(label_list, pred_list)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
