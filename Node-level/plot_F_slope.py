import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

figure = plt.figure(figsize=(18, 8))
sns.set(style='whitegrid', color_codes=True)
plt.gcf().subplots_adjust(bottom=0.28, wspace=0.3)

plt.subplot(121)
x = ['Acc@1', 'Acc@5', 'Acc@10', 'Acc@20']
data = [[0.14115545, 0.30077427, 0.373436569, 0.44192972],
        [0.146515783, 0.337701013, 0.412745682, 0.509231686],
        [0.154258487, 0.346634902, 0.427039905, 0.502680167],
        [0.144133413, 0.337701013, 0.419297201, 0.500297796],
        [0.147706968, 0.338296605, 0.412745682, 0.499702204]]
data = np.array(data)
data = np.transpose(data)
wide_df = pd.DataFrame(data, x, ["1-ary", "2-ary", "3-ary", "4-ary", "5-ary"])
ax2 = sns.lineplot(data=wide_df, markers=True, markersize=15, palette="icefire")
plt.xlabel('(a) Gowalla', fontdict={'family': 'Times New Roman', 'size': 43})
ax2.set_ylabel('')
plt.yticks(fontproperties='Times New Roman', size=30)
plt.xticks(fontproperties='Times New Roman', size=30)
plt.legend(ncol=2, loc="upper left", columnspacing=0.1,
           prop={'family': 'Times New Roman', 'size': 28})  # label的位置在左上，没有这句会找不到label去哪了
# plt.ylim(0.015, 0.07)

plt.subplot(122)
x = ['Acc@1', 'Acc@5', 'Acc@10', 'Acc@20']
data = [[0.14115545, 0.30077427, 0.373436569, 0.44192972],
        [0.146515783, 0.337701013, 0.412745682, 0.509231686],
        [0.154258487, 0.346634902, 0.427039905, 0.502680167],
        [0.144133413, 0.337701013, 0.419297201, 0.500297796],
        [0.147706968, 0.338296605, 0.412745682, 0.499702204]]
data = np.array(data)
data = np.transpose(data)
wide_df = pd.DataFrame(data, x, ["1-ary", "2-ary", "3-ary", "4-ary", "5-ary"])
ax1 = sns.lineplot(data=wide_df, markers=True, markersize=15, palette="icefire")
ax1.set_ylabel('')
plt.xlabel('(b) TKY', fontdict={'family': 'Times New Roman', 'size': 43})
plt.yticks(fontproperties='Times New Roman', size=30)
plt.xticks(fontproperties='Times New Roman', size=30)
plt.legend(ncol=2, loc="upper left", columnspacing=0.1,
           prop={'family': 'Times New Roman', 'size': 28})  # label的位置在左上，没有这句会找不到label去哪了
# plt.ylim(0, 0.068)


plt.show()
figure.savefig('line_chart.pdf')
