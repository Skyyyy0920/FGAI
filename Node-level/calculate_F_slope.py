import os
import numpy as np
import pandas as pd

# dataset = 'ogbn-arxiv'
# dataset = 'amazon-photo'
# dataset = 'amazon-cs'
# dataset = 'coauthor-phy'
dataset = 'pubmed'
print(dataset)
file_names = ["GAT.txt", "GATv2.txt", "GAT_FGAI.txt", "GATv2_FGAI.txt"] + \
             ["GAT+AT.txt", "GAT+LN.txt", "GAT+AT_FGAI.txt", "GAT+LN_FGAI.txt"]
labels = ["GAT", "GATv2", "GAT_FGAI", "GATv2_FGAI"] + ["GAT+AT", "GAT+LN", "GAT+AT_FGAI", "GAT+LN_FGAI"]

data = []
for file_name in file_names:
    file_path = os.path.join(f"./F/{dataset}", file_name)
    # file_path = os.path.join(f"./for_F/{dataset}", file_name)
    df = pd.read_csv(file_path)
    data.append(df)

x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
pos_list, neg_list = [], []
for i in range(len(data)):
    pos_list.append(data[i]['fidelity_pos'][0:7])
    neg_list.append(data[i]['fidelity_neg'][0:7])

slopes_pos = []
slopes_neg = []
for i in range(len(data)):
    # 获取x和y数据
    x = np.array(x)
    y_pos = np.array(pos_list[i])
    y_neg = np.array(neg_list[i])

    # 使用最小二乘法进行拟合
    slope_pos = np.polyfit(x, y_pos, 1)
    slope_neg = np.polyfit(x, y_neg, 1)

    # 将斜率添加到相应的列表中
    slopes_pos.append(slope_pos)
    slopes_neg.append(slope_neg)

# 打印或使用斜率值
for i in range(len(labels)):
    print(f"Label: {labels[i]}")
    print(f"Positive Slope: {slopes_pos[i]}")
    print(f"Negative Slope: {slopes_neg[i]}")
