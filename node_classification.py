from model import *
import torch
import torch.nn.functional as F

dataset = dgl.data.CoraGraphDataset()
graph = dataset[0]

# 参数设置
in_feats = graph.ndata['feat'].size(1)
hidden_feats = 32
num_classes = dataset.num_classes
num_heads = 1
learning_rate = 0.001
num_epochs = 500
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(device)

# 创建模型和优化器
# model = GATModel(in_feats, hidden_feats, num_classes, num_heads).to(device)
model = GATNodeClassifier(in_feats, hidden_feats, num_classes, 0.3, 0.1, num_heads).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 数据准备
g = graph.to(device)
features = g.ndata['feat']
labels = g.ndata['label']


def train(model, g, features, labels, optimizer):
    model.train()
    optimizer.zero_grad()
    logits = model(features, g.adjacency_matrix().to_dense())
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


for epoch in range(num_epochs):
    loss = train(model, g, features, labels, optimizer)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

model.eval()
with torch.no_grad():
    logits = model(features, g.adjacency_matrix().to_dense())
    predicted_labels = logits.argmax(dim=1)
    accuracy = (predicted_labels == labels).float().mean().item()
    print(f"Test Accuracy: {accuracy:.4f}")
