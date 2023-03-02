import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False   # 使坐标轴刻度显示负号值
plt.rcParams['font.size'] = 10 #设置字体大小
import pandas as pd

data = pd.read_csv('./info.csv')
data = data.values[0]

def draw_statistic(data):
    plt.xlabel("星期")
    plt.ylabel("违规次数")
    plt.plot([1,2,3,4,5,6,7], data)
    plt.savefig('./save_pic/statistic.png')
    plt.show()

draw_statistic(data)