import numpy as np
import pandas as pd

data = pd.read_csv("missing_data.csv", header=None, sep=',')
print("lagrange插值前（False为缺失值所在位置）",'\n',data.notnull())
whethernull=data[data.isnull().values == True]
print("插值前有空值的行：\n", whethernull)
print("\n此时各列中的空值总量：\n", data.isnull().sum())
# 拉格朗日插值法
from scipy.interpolate import lagrange
for i in range(0,3):
    #"训练"lagrange模型
    la=lagrange(data.loc[:,i].dropna().index,data.loc[:,i].dropna().values)
    #list_d用于记录当前列缺失值所在的行（记录缺失值下标）
    list_d=list(set(np.arange(0,21)).difference(set(data.loc[:,i].dropna().index)))
    #将缺失值list_d带入训练好的模型，并填入对应的位置
    data.loc[list_d,i]=la(list_d)
    print("第%d列缺失值的个数为:%d"%(i,data.loc[:,i].isnull().sum()))
print("lagrange插值后（False为缺失值所在位置）","\n",data.notnull())
print(data)
# 保存
data.to_csv('missLagrange.csv', header=False, index=False)
print("=============================================")

# 合并线损、用电量趋势与线路警告数据
loss = pd.read_csv("ele_loss.csv",encoding="GBK")
alarm = pd.read_csv("alarm.csv",encoding="GBK")

# 形状--先看形状在合并
print("线损表形状：", loss.shape)# (49, 4)
print("\n线路警告表形状：", alarm.shape)#(25, 3)

# 合并ID Date 列 内链接 内连接相同项合并连接
merge= pd.merge(left=loss, right=alarm, how="inner", left_on=['ID', 'date'], right_on=['ID', 'date'])
print("合并后的表形状为：",merge.shape)
print("合并后的数据表：\n", merge)
print("=============================================")

# 标准化建模专家样本数据
model = pd.read_csv("model.csv", encoding='GBK')
def Standard(data):
    data=(data-data.mean())/data.std()
    return data
S=Standard(model)
print("标准化后的数据为：",'\n',S.head())

#离差标准化函数
def MinMaxScale(data):
    data=(data-data.min())/(data.max()-data.min())
    return data
M=MinMaxScale(model)
print("离差标准化后的数据为：",'\n',S.head())

#小数定标差标准化函数
def DecimalScaler(data):
    data=data/10**np.ceil(np.log10(data.abs().max()))
    return data
D=DecimalScaler(model)
print("小数定标差标准化的数据为：",'\n',D.head())