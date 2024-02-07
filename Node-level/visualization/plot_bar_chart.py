import numpy as np
import pylab as pl
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# 全局设置
sns.set(style='whitegrid', color_codes=True)  # 设置绘图风格和颜色
figure = plt.figure(figsize=(18, 6.5))
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内

# 子图设置
# plt.subplot(131)

data = [[0.7700, 0.7770, 0.7820, 0.7742],
        [0.5900, 0.7140, 0.7180, 0.3020],
        [0.7790, 0.7830, 0.7920, 0.7782],
        [0.7600, 0.7280, 0.7240, 0.7550]]

# plt.xlabel('Pubmed', fontdict={'family': 'Times New Roman', 'size': 20})
plt.yticks(fontproperties='Times New Roman', size=15)
plt.xticks(fontproperties='Times New Roman', size=15)

X = np.array([0.9, 0.9 * 2, 0.9 * 3, 0.9 * 4])  # X是1,2,3,4,5,6,7,8,柱的个数
plt.bar(X + 0.55, data[0], alpha=0.7, width=0.13, facecolor='#C43302', edgecolor='black', label='F1 w/o FGAI', lw=1)
plt.bar(X + 0.68, data[1], alpha=0.7, width=0.13, hatch='//', facecolor='#EDAA25', edgecolor='black',
        label='$\widetilde{F}1$ w/o FGAI', lw=1)
plt.bar(X + 0.81, data[2], alpha=0.7, width=0.13, facecolor='#B7BF99', edgecolor='black', label='F1 w/ FGAI ', lw=1)
plt.bar(X + 0.94, data[3], alpha=0.7, width=0.13, hatch='//', facecolor='#0A7373', edgecolor='black',
        label='$\widetilde{F}1$ w/ FGAI', lw=1)
pos = X + 0.9 - 0.08
plt.xticks(pos.tolist(), ['GAT', 'GAT+LN', 'GAT+AT', 'GATv2'])

# plt.legend(ncol=1, loc="upper left", columnspacing=0.2, prop={'family': 'Times New Roman', 'size': 15})
plt.tick_params(axis='x', length=0)
plt.savefig('bar_chart_Pubmed.pdf')

data = [[0.7819, 0.4349, 0.5516, 0.7977],
        [0.3158, 0.4255, 0.4910, 0.3443],
        [0.7668, 0.4322, 0.6301, 0.8116],
        [0.7649, 0.4300, 0.6279, 0.8118]]

# plt.xlabel('Amazon-photo', fontdict={'family': 'Times New Roman', 'size': 20})
plt.yticks(fontproperties='Times New Roman', size=15)
plt.xticks(fontproperties='Times New Roman', size=15)

X = np.array([0.9, 0.9 * 2, 0.9 * 3, 0.9 * 4])  # X是1,2,3,4,5,6,7,8,柱的个数
plt.bar(X + 0.55, data[0], alpha=0.7, width=0.13, facecolor='#C43302', edgecolor='black', label='F1 w/o FGAI', lw=1)
plt.bar(X + 0.68, data[1], alpha=0.7, width=0.13, hatch='//', facecolor='#EDAA25', edgecolor='black',
        label='$\widetilde{F}1$ w/o FGAI', lw=1)
plt.bar(X + 0.81, data[2], alpha=0.7, width=0.13, facecolor='#B7BF99', edgecolor='black', label='F1 w/ FGAI ', lw=1)
plt.bar(X + 0.94, data[3], alpha=0.7, width=0.13, hatch='//', facecolor='#0A7373', edgecolor='black',
        label='$\widetilde{F}1$ w/ FGAI', lw=1)
pos = X + 0.9 - 0.08
plt.xticks(pos.tolist(), ['GAT', 'GAT+LN', 'GAT+AT', 'GATv2'])

plt.legend(ncol=1, loc="upper center", columnspacing=0.2, prop={'family': 'Times New Roman', 'size': 12})
plt.tick_params(axis='x', length=0)
plt.savefig('bar_chart_Amazon-photo.pdf')

data = [[0.7888, 0.5475, 0.8321, 0.7300],
        [0.2605, 0.5296, 0.8202, 0.3457],
        [0.7772, 0.5470, 0.8153, 0.7205],
        [0.7773, 0.5459, 0.8154, 0.7206]]

# plt.xlabel('Amazon-cs', fontdict={'family': 'Times New Roman', 'size': 20})
plt.yticks(fontproperties='Times New Roman', size=15)
plt.xticks(fontproperties='Times New Roman', size=15)

X = np.array([0.9, 0.9 * 2, 0.9 * 3, 0.9 * 4])  # X是1,2,3,4,5,6,7,8,柱的个数
plt.bar(X + 0.55, data[0], alpha=0.7, width=0.13, facecolor='#C43302', edgecolor='black', label='F1 w/o FGAI', lw=1)
plt.bar(X + 0.68, data[1], alpha=0.7, width=0.13, hatch='//', facecolor='#EDAA25', edgecolor='black',
        label='$\widetilde{F}1$ w/o FGAI', lw=1)
plt.bar(X + 0.81, data[2], alpha=0.7, width=0.13, facecolor='#B7BF99', edgecolor='black', label='F1 w/ FGAI ', lw=1)
plt.bar(X + 0.94, data[3], alpha=0.7, width=0.13, hatch='//', facecolor='#0A7373', edgecolor='black',
        label='$\widetilde{F}1$ w/ FGAI', lw=1)
pos = X + 0.9 - 0.08
plt.xticks(pos.tolist(), ['GAT', 'GAT+LN', 'GAT+AT', 'GATv2'])

# plt.legend(ncol=1, loc="upper left", columnspacing=0.2, prop={'family': 'Times New Roman', 'size': 15})
plt.tick_params(axis='x', length=0)
plt.savefig('bar_chart_Amazon-cs.pdf')

data = [[0.9477, 0.9072, 0.9355, 0.9467],
        [0.7325, 0.9038, 0.9172, 0.7972],
        [0.9478, 0.8974, 0.9399, 0.9460],
        [0.9447, 0.8942, 0.9293, 0.9456]]

# plt.xlabel('Coauthor', fontdict={'family': 'Times New Roman', 'size': 20})
plt.yticks(fontproperties='Times New Roman', size=15)
plt.xticks(fontproperties='Times New Roman', size=15)

X = np.array([0.9, 0.9 * 2, 0.9 * 3, 0.9 * 4])  # X是1,2,3,4,5,6,7,8,柱的个数
plt.bar(X + 0.55, data[0], alpha=0.7, width=0.13, facecolor='#C43302', edgecolor='black', label='F1 w/o FGAI', lw=1)
plt.bar(X + 0.68, data[1], alpha=0.7, width=0.13, hatch='//', facecolor='#EDAA25', edgecolor='black',
        label='$\widetilde{F}1$ w/o FGAI', lw=1)
plt.bar(X + 0.81, data[2], alpha=0.7, width=0.13, facecolor='#B7BF99', edgecolor='black', label='F1 w/ FGAI ', lw=1)
plt.bar(X + 0.94, data[3], alpha=0.7, width=0.13, hatch='//', facecolor='#0A7373', edgecolor='black',
        label='$\widetilde{F}1$ w/ FGAI', lw=1)
pos = X + 0.9 - 0.08
plt.xticks(pos.tolist(), ['GAT', 'GAT+LN', 'GAT+AT', 'GATv2'])

# plt.legend(ncol=1, loc="upper left", columnspacing=0.2, prop={'family': 'Times New Roman', 'size': 15})
plt.tick_params(axis='x', length=0)
plt.savefig('bar_chart_Coauthor.pdf')
