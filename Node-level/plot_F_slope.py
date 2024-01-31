import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# dataset = 'amazon-photo'
# dataset = 'amazon-cs'
# dataset = 'coauthor-cs'
dataset = 'coauthor-phy'
print(dataset)

file_names = ["GAT.txt", "GATv2.txt", "GT.txt"] + \
             ["GAT+AT.txt", "GATv2+AT.txt", "GT+AT.txt"] + \
             ["GAT+FGAI.txt", "GATv2+FGAI.txt", "GT+FGAI.txt"]
labels = ["GAT", "GATv2", "GT"] + \
         ["GAT+AT", "GATv2+AT", "GT+AT"] + \
         ["GAT+FGAI", "GATv2+FGAI", "GT+FGAI"]

data = []
for file_name in file_names:
    file_path = os.path.join(f"./F/{dataset}", file_name)
    df = pd.read_csv(file_path)
    data.append(df)

x = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
pos_list, neg_list = [], []
for i in range(len(data)):
    pos_list.append(data[i]['fidelity_pos'][0:6])
    neg_list.append(data[i]['fidelity_neg'][0:6])

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
    print(labels[i])
    print(f"Positive Slope: {slopes_pos[i]}, Negative Slope: {slopes_neg[i]}")

figure = plt.figure(figsize=(20, 10))  # 调整总体图的大小
sns.set(style='whitegrid', color_codes=True)
plt.gcf().subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.98, wspace=0.15)  # 调整子图位置和间距

x = ['0%', '10%', '20%', '30%', '40%', '50%']
color_palette = {'GAT': '#FF1F5B', 'GATv2': '#00CD6C', 'GT': '#009ADE',
                 'GAT+AT': '#A0B1BA', 'GATv2+AT': '#A6761D', 'GT+AT': 'black',
                 'GAT+FGAI': '#AF58BA', 'GATv2+FGAI': '#FFC61E', 'GT+FGAI': '#F28522'}

pos_list = np.transpose(np.array(pos_list))
neg_list = np.transpose(np.array(neg_list))

plt.subplot(121)
wide_df = pd.DataFrame(pos_list, x, labels)
ax2 = sns.lineplot(data=wide_df, markers=True, markersize=15, palette=color_palette, linewidth=3)

plt.xlabel('(a) negative perturbation', fontdict={'family': 'Times New Roman', 'size': 43})
ax2.set_ylabel('')
plt.yticks(fontproperties='Times New Roman', size=30)
plt.xticks(fontproperties='Times New Roman', size=30)
plt.legend(ncol=3, loc="lower left", columnspacing=1.5, prop={'family': 'Times New Roman', 'size': 16})

plt.subplot(122)
wide_df = pd.DataFrame(neg_list, x, labels)
ax2 = sns.lineplot(data=wide_df, markers=True, markersize=15, palette=color_palette, linewidth=3)

plt.xlabel('(b) positive perturbation', fontdict={'family': 'Times New Roman', 'size': 43})
ax2.set_ylabel('')
plt.yticks(fontproperties='Times New Roman', size=30)
plt.xticks(fontproperties='Times New Roman', size=30)
plt.legend(ncol=3, loc="lower left", columnspacing=1.5, prop={'family': 'Times New Roman', 'size': 16})

plt.tight_layout()
figure.savefig(f'./F_{dataset}.pdf')
