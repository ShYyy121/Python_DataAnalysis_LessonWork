import numpy as np
import matplotlib.pyplot as plt
data=np.load("数据分析1数据-populations.npz",allow_pickle=True)
# 设置中文字体
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False
label3=['1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']
keys = data.files
values = data[keys[0]][-3::-1, :]  # 将数据改为年份升序，并去除无效项
print(data[keys[0]])
name=data['feature_names']
print(name)
# 创建网格
p1 = plt.figure(figsize = (15, 9))
ax1 = p1.add_subplot(1, 2, 1)
plt.title('1996-2015年人口数据特征间的关系散点图')
plt.xlabel('年份')
plt.ylabel('人口数（万人）')
plt.xticks(range(0, 20), values[:, 0], rotation = 45)
# maker形状 c颜色
plt.scatter(values[:, 0], values[:, 1], marker = 'o', c = '#0d0d73')
plt.scatter(values[:, 0], values[:, 2], marker = 'o', c = '#bf854d')
plt.scatter(values[:, 0], values[:, 3], marker = 'o', c = '#4798b3')
plt.scatter(values[:, 0], values[:, 4], marker = 'o', c = '#500024')
plt.scatter(values[:, 0], values[:, 5], marker = 'o', c = '#23482c')
plt.legend(['年末总人口','男性人口','女性人口','城镇人口','乡村人口'])

ax2 = p1.add_subplot(1, 2, 2)
plt.title('1996-2015年人口数据特征间的关系折线图')
plt.xlabel('年份')
plt.ylabel('人口数（万人）')
plt.xticks(range(0,20), values[:,0], rotation=45)
plt.plot(values[:,0], values[:,1], 'ko-',
         values[:,0], values[:,2], 'mo-.',
         values[:,0], values[:,3], 'yo--',
         values[:,0], values[:,4], 'ro:',
         values[:,0], values[:,5], 'go-.')
plt.legend(['年末总人口','男性人口','女性人口','城镇人口','乡村人口'])
# 保存
plt.savefig('1996~2015年人口数据特征间关系散点图和折线图.png')

p2=plt.figure(figsize=(12,12))
a1=p2.add_subplot(2,2,1)
plt.bar(range(2),values[0,2:4],width=0.3,color='#e9967a')
plt.xlabel('性别')
plt.ylabel('人口（万人）')
# 限制y范围
plt.ylim(0,90000)
plt.xticks(range(2),['男性','女性'])
plt.title('1996年男、女人口数直方图')

b1=p2.add_subplot(2,2,2)
plt.bar(range(2),values[19,2:4],width=0.3,color='#66ff00')
plt.xlabel('性别')
plt.ylabel('人口（万人）')
plt.ylim(0,90000)
plt.xticks(range(2),['男性','女性'])
plt.title('2015年男、女人口数直方图')


#子图3
c1=p2.add_subplot(2,2,3)
plt.bar(range(2),values[0,4:6],width=0.3,color='#e9967a')
plt.xlabel('类别')
plt.ylabel('人口（万人）')
plt.ylim(0,90000)
plt.xticks(range(2),['城镇','乡村'])
plt.title('1996年城、乡人口数直方图')

#子图4
d1=p2.add_subplot(2,2,4)
plt.bar(range(2),values[19,4:6],width=0.3,color='#66ff00')
plt.xlabel('类别')
plt.ylabel('人口（万人）')
plt.ylim(0,90000)
plt.xticks(range(2),['城镇','乡村'])
plt.title('2015年城、乡人口数直方图')
plt.savefig('1996年和2015年各类人口直方图.png')

#柱状图
p3=plt.figure(figsize=(21,9))
#子图1
a1=p3.add_subplot(1,2,1)
p = range(20)
plt.bar(x=p,height=values[0:20,2],width=0.3,color='#ffd700',label='男性人口')
plt.bar(x=[i+0.3 for i in p],height=values[0:20,3],width=0.4,color='#4798b3',label='女性人口')
plt.ylabel('人口（万人）')
plt.ylim(0,80000)#
plt.xticks(range(20),label3)
plt.legend()
plt.title('1996~2015年男、女人口数对比柱状图')

#子图2
a2=p3.add_subplot(1,2,2)
p = range(20)
plt.bar(x=p,height=values[0:20,4],width=0.3,color='#ffd700',label='乡村人口')
plt.bar(x=[i+0.3 for i in p],height=values[0:20,5],width=0.4,color='#4798b3',label='城镇人口')
plt.ylabel('人口（万人）')
plt.ylim(0,90000)#
plt.xticks(range(20),label3)
plt.legend()
plt.title('1996~2015年城、乡人口数对比柱状图')
plt.savefig('1996~2015年人口数对比柱状图.png')
# 饼图
p3=plt.figure(figsize=(10,10))
#子图1
a2=p3.add_subplot(2,2,1)
# autoopct 保留一位小数并带有一个%
# explode 设置突出
plt.pie(values[0,2:4],explode=[0.015, 0.015],labels=['男性','女性'],colors=['#ff4d40','#ccff00'],autopct='%1.1f%%')
plt.title('1996年男、女人口数饼图')

#子图2
b2=p3.add_subplot(2,2,2)
plt.pie(values[19,2:4],explode=[0.015, 0.015],labels=['男性','女性'],colors=['#b399ff','#1bead9'],autopct='%1.1f%%')
plt.title('2015年男、女人口数饼图')

#子图3
c2=p3.add_subplot(2,2,3)
plt.pie(values[0,4:6],explode=[0.015, 0.015],labels=['城镇','乡村'],colors=['#ff4d40','#ccff00'],autopct='%1.1f%%')
plt.title('1996年城、乡人口数饼图')

#子图4
d2=p3.add_subplot(2,2,4)
plt.pie(values[19,4:6],explode=[0.015, 0.015],labels=['城镇','乡村'],colors=['#b399ff','#1bead9'],autopct='%1.1f%%')
plt.title('2015年城、乡人口数饼图')
plt.savefig('1996年和2015年各类人口饼图.png')

# 箱线图
p4 = plt.figure(figsize=(10,10))
# notch 凹凸形式展现箱型图
#meanline 线的形式表示均值
plt.boxplot(values[0:20,1:6],notch=False,labels=['年末','男性','女性','城镇','乡村'],meanline=True)
plt.xlabel('类别')
plt.ylabel('人口（万人）')
plt.title('1996-2015年各特征人口箱线图')
plt.savefig('1996-2015年各特征人口箱线图.png')

plt.show()