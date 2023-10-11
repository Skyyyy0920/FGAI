import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

figure = plt.figure(figsize=(25, 10))
sns.set(style='whitegrid', color_codes=True)
plt.gcf().subplots_adjust(bottom=0.28, wspace=0.3)

file_names = ["GAT.txt", "GAT+AT.txt", "GAT+LN.txt", "GATv2.txt",
              "GAT_FGAI.txt", "GAT+AT_FGAI.txt", "GAT+LN_FGAI.txt", "GATv2_FGAI.txt"]
labels = ["GAT", "GAT+AT", "GAT+LN", "GATv2", "GAT_FGAI", "GAT+AT_FGAI", "GAT+LN_FGAI", "GATv2_FGAI"]

dataset = 'pubmed'
data = []
for file_name in file_names:
    file_path = os.path.join(f"./for_F/{dataset}", file_name)
    df = pd.read_csv(file_path)
    data.append(df)

pos_list, neg_list = [], []
x = ['10%', '20%', '30%', '40%', '50%', '60%', '70%']
y = ["GAT", "GAT+AT", "GAT+LN", "GATv2", "GAT+FGAI", "GAT+AT+FGAI", "GAT+LN+FGAI", "GATv2+FGAI"]
for i in range(len(data)):
    pos_list.append(data[i]['fidelity_pos'][1:8])
    neg_list.append(data[i]['fidelity_neg'][1:8])
pos_list = np.transpose(np.array(pos_list))
neg_list = np.transpose(np.array(neg_list))

plt.subplot(121)
wide_df = pd.DataFrame(pos_list, x, y)
ax2 = sns.lineplot(data=wide_df, markers=True, markersize=15, palette="icefire")

plt.xlabel('(a) $F_{slope}^+$', fontdict={'family': 'Times New Roman', 'size': 43})
ax2.set_ylabel('')
plt.yticks(fontproperties='Times New Roman', size=30)
plt.xticks(fontproperties='Times New Roman', size=30)
plt.legend(ncol=2, loc="lower left", columnspacing=0.5, prop={'family': 'Times New Roman', 'size': 16})

plt.subplot(122)
wide_df = pd.DataFrame(neg_list, x, y)
ax2 = sns.lineplot(data=wide_df, markers=True, markersize=15, palette="icefire")

plt.xlabel('(b) $F_{slope}^-$', fontdict={'family': 'Times New Roman', 'size': 43})
ax2.set_ylabel('')
plt.yticks(fontproperties='Times New Roman', size=30)
plt.xticks(fontproperties='Times New Roman', size=30)
plt.legend(ncol=2, loc="lower left", columnspacing=0.5, prop={'family': 'Times New Roman', 'size': 16})

figure.savefig(f'./line_chart_{dataset}.pdf')
plt.show()
