import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# dataset = 'pubmed'
# dataset = 'ogbn-arxiv'
# dataset = 'amazon-photo'
dataset = 'amazon-cs'
# dataset = 'coauthor-phy'

figure = plt.figure(figsize=(20, 10))  # 调整总体图的大小
sns.set(style='whitegrid', color_codes=True)
plt.gcf().subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.98, wspace=0.15)  # 调整子图位置和间距

file_names = ["GAT.txt", "GATv2.txt", "GAT_FGAI.txt", "GATv2_FGAI.txt"]
labels = ["GAT", "GATv2", "GAT_FGAI", "GATv2_FGAI"]

data = []
for file_name in file_names:
    file_path = os.path.join(f"./for_F/{dataset}", file_name)
    df = pd.read_csv(file_path)
    data.append(df)

pos_list, neg_list = [], []
x = ['10%', '20%', '30%', '40%', '50%', '60%', '70%']
y = ["GAT", "GATv2", "GAT+FGAI", "GATv2+FGAI"]
color_palette = {'GAT': '#1E90FF', 'GATv2': '#6A5ACD', 'GAT+FGAI': '#FF4500', 'GATv2+FGAI': '#FF8C00'}
for i in range(len(data)):
    pos_list.append(data[i]['fidelity_pos'][1:8])
    neg_list.append(data[i]['fidelity_neg'][1:8])
pos_list = np.transpose(np.array(pos_list))
neg_list = np.transpose(np.array(neg_list))

plt.subplot(121)
wide_df = pd.DataFrame(pos_list, x, y)
ax2 = sns.lineplot(data=wide_df, markers=True, markersize=15, palette=color_palette)

plt.xlabel('(a) negative perturbation', fontdict={'family': 'Times New Roman', 'size': 43})
ax2.set_ylabel('')
plt.yticks(fontproperties='Times New Roman', size=30)
plt.xticks(fontproperties='Times New Roman', size=30)
plt.legend(ncol=2, loc="lower left", columnspacing=0.5, prop={'family': 'Times New Roman', 'size': 16})

plt.subplot(122)
wide_df = pd.DataFrame(neg_list, x, y)
ax2 = sns.lineplot(data=wide_df, markers=True, markersize=15, palette=color_palette)

plt.xlabel('(b) positive perturbation', fontdict={'family': 'Times New Roman', 'size': 43})
ax2.set_ylabel('')
plt.yticks(fontproperties='Times New Roman', size=30)
plt.xticks(fontproperties='Times New Roman', size=30)
plt.legend(ncol=2, loc="lower left", columnspacing=0.5, prop={'family': 'Times New Roman', 'size': 16})

figure.savefig(f'./line_chart_{dataset}_1.pdf')
# plt.show()

figure = plt.figure(figsize=(20, 10))  # 调整总体图的大小
sns.set(style='whitegrid', color_codes=True)
plt.gcf().subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.98, wspace=0.15)  # 调整子图位置和间距

file_names = ["GAT+AT.txt", "GAT+LN.txt", "GAT+AT_FGAI.txt", "GAT+LN_FGAI.txt"]
labels = ["GAT+AT", "GAT+LN", "GAT+AT_FGAI", "GAT+LN_FGAI"]

data = []
for file_name in file_names:
    file_path = os.path.join(f"./for_F/{dataset}", file_name)
    df = pd.read_csv(file_path)
    data.append(df)

pos_list, neg_list = [], []
x = ['10%', '20%', '30%', '40%', '50%', '60%', '70%']
y = ["GAT+AT", "GAT+LN", "GAT+AT+FGAI", "GAT+LN+FGAI"]
color_palette = {'GAT+AT': '#1E90FF', 'GAT+LN': '#6A5ACD', 'GAT+AT+FGAI': '#FF4500', 'GAT+LN+FGAI': '#FF8C00'}
for i in range(len(data)):
    pos_list.append(data[i]['fidelity_pos'][1:8])
    neg_list.append(data[i]['fidelity_neg'][1:8])
pos_list = np.transpose(np.array(pos_list))
neg_list = np.transpose(np.array(neg_list))

plt.subplot(121)
wide_df = pd.DataFrame(pos_list, x, y)
ax2 = sns.lineplot(data=wide_df, markers=True, markersize=15, palette=color_palette)

plt.xlabel('(a) negative perturbation', fontdict={'family': 'Times New Roman', 'size': 43})
ax2.set_ylabel('')
plt.yticks(fontproperties='Times New Roman', size=30)
plt.xticks(fontproperties='Times New Roman', size=30)
plt.legend(ncol=2, loc="lower left", columnspacing=0.5, prop={'family': 'Times New Roman', 'size': 16})

plt.subplot(122)
wide_df = pd.DataFrame(neg_list, x, y)
ax2 = sns.lineplot(data=wide_df, markers=True, markersize=15, palette=color_palette)

plt.xlabel('(b) positive perturbation', fontdict={'family': 'Times New Roman', 'size': 43})
ax2.set_ylabel('')
plt.yticks(fontproperties='Times New Roman', size=30)
plt.xticks(fontproperties='Times New Roman', size=30)
plt.legend(ncol=2, loc="lower left", columnspacing=0.5, prop={'family': 'Times New Roman', 'size': 16})

figure.savefig(f'./line_chart_{dataset}_2.pdf')
# plt.show()
