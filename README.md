# Python数据分析课程作业
## 一．实验目的
在基础实验中，能熟练运用Python语言的基本语法、控制语句、函数、面向对象、GUI等开发一个应用类游戏。为后续Python学习打下扎实基础。
在数据分析实验中，能熟练运用Python列表、字典、集合、数组等数据结构以及索引和切片等查询操作解决实际问题，熟练掌握Python读写数据文件的方法，熟练使用Matplotlib数据可视化工具，熟练使用pandas数据分析库函数，基本掌握使用scikit-learn库多种机器学习算法进行数据分析的过程，了解Python程序的调试方法，运用Python编写程序解决实际应用问题。
## 二．实验内容
1. 基础实验
设计一个带有图形用户界面的人机对战井字棋游戏。
游戏在九宫方格内进行，如果一方抢先于某方向（横、竖、斜）连成3子，则获取胜利。游戏界面，首先询问哪个玩家先走，然后根据玩家落子位置，显示棋盘状态。最终根据游戏规则，评判哪个玩家获胜。
2. 数据分析实验
分析1996- -2015年人口数据特征间的关系以及各个特征的分布与分散状况
插补用户用电量数据缺失值，合并线损、用电量趋势与线路告警数据，标准化建模专家样本数据
使用sklearn处理wine和wine_quality 数据集，构建基于wine数据集的K-Means聚类模型、SVM分类模型、回归模型
三．实验环境 Experiment Environment
操作系统：Windows 10 
编译软件： Intellij IDEA 2022.1.3 
Python 3.10

## 四．实验过程与分析
4.1 基础实验
1. 问题描述及分析
设计一个带有图形用户界面的人机对战井字棋游戏。游戏在九宫方格内进行，如果一方抢先于 
某方向（横、竖、斜）连成 3 子，则获取胜利。游戏界面，首先询问哪个玩家先走，然后根据玩家 
落子位置，显示棋盘状态。最终根据游戏规则，评判哪个玩家获胜。 
该问题的重点与难点在于：如何在图形界面中实现下棋功能，如何判断当前棋盘中是否有赢家， 
以及电脑如何选择下棋位置。玩家可以通过在每一次下棋之后，判断某一路线中的三个下棋点是否为同一玩家的棋子，若相同，则该玩家获胜，游戏结束；若不存在这样的情况，则此时没有玩家获胜，游戏继续。电脑可以通过自己的算法，来帮助判断当前的下棋 、位置，判断的优先级依次为：赢得游戏、不能输（平局）、最佳空位置下棋。
同时，双方每次只能落一枚棋子，在落下字后删除下棋的位置，避免重复下棋。在游戏开始前的界面，首先应选择是玩家先走还是计算机先走，然后根据玩家和计算机落子位置，显示棋盘状态。最终根据游戏规则，评判获胜方，或是和棋平局。
2. 关键算法设计
关键点1：利用tkinter实现下棋功能：
利用九宫格，定义棋盘每个点的位置
0	1	2
3	4	5
6	7	8
利用按钮实现下棋的方式，根据行列循环先将棋盘九宫格填充按钮，并且定义监听按下后实现的函数内容，当按下按钮，棋盘文本被修改为‘X’或‘O’，并且删除按下按钮的位置。

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/71db366b-c789-468d-a6e4-d24f3ce2bae2)


在胜利条件判断的时候，将所有可能全部列出来，判断是否满足，如果满足就返回true，反之则返回false

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/22462aef-116c-4a4a-82f0-0b040116b8c4)

关键点2：电脑下棋算法的实现：
电脑的每一步落子都包含在以下这三种情况中：
①如果有一步棋可以让计算机在本轮获胜，就选那一步走。
②否则，如果有一步棋可以让玩家在本轮获胜，就选那一步走。
在1.2中，利用循环遍历剩下的位置，同时复制棋盘，模拟下棋，遍历所有仍未下字的位置，当有一个地方可以使自己赢得胜利的时候，就下那一步使自己获得胜利，如果不是使自己获得胜利，而是使对手获得胜利，也同样下那一步阻止对手胜利。
③否则，计算机应该选择最佳空位置来走。最佳位置即为棋盘正中间的空位，其次是四个角，最后是剩下的四个位置。在程序中定义一个元组best储存最佳方格的位置：[4,0,2,6,8,1,3,5,7]。上述规则简单来说，就是电脑优先考虑己方获胜，其次是阻止玩家获胜，在前两种情况都不符合的情况下，选择最佳位置落子。按这样的规则设计程序，可实现电脑下棋的智能性。

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/186a4cad-e832-46e1-90ad-cfbd46a7e7bd)


3.实现结果
  程序运行后先选择开始游戏（如左上图所示），按下开始游戏后，提示是否选择先手下棋（如右上图所示）。之后玩家与电脑轮流下棋，直到已经下棋的位置不能再次下棋，同时实现了电脑下棋的智能性。如果最终玩家与计算机均为取得胜利则提示Match Tied（平局）（如下两图所示）。

        ![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/b058cc30-9de9-465f-8ce8-784625d553ad)
![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/ee4ed56c-3582-457b-9a69-f2c1365cfc77)
![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/52018425-5b67-4961-b889-20e1439bea5c)
![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/3f13a2a0-dc90-4ad1-b739-acba7bbb94b1)


## 4.2 数据分析实验
4.2.1 分析数据特征间的关系以及各个特征的分布与分散状况
（1）问题理解
    通过Python实现数据可视化，通过统计图对人口数据进行分析和预测。通过绘制散点图、折线图、柱状图和饼图等来分析和预测人口结构的变化，分析预测未来男女人口比例、城乡人口变化的方向，也可以通过箱线图发现不同特征增长或者减少的速率有怎样的变化。完成本次实验需要使用使用numpy库读取人口数据，需要掌握matplotlib.pyplot的基础语法和子图的绘制方法。
所绘制的统计图需要便于查看，各种图形、字体和图例等大小需要合适，以便后续进行分析。
在 populations.npz 数据集中存储了 1996 年~2015 年的人口数据，总共拥有 6 个特征，分别为年末总人口、男性人口、女性人口、城镇人口、乡村人口和年份。可以利用 numpy 来分析各个特征随着时间推移发生的变化情况，从而分析出未来男女人口比例、城乡人口变化的方向。
散点图和折线图是数据分析最常用的两种图形。这两种图形都能够分析不同数值型特征间的关系。其中，散点图主要用于分析特征间的相关关系，折线图则用于分析自变量特征和因变量特征之间的趋势关系。
直方图、饼图和箱线图是另外三种数据分析常用的图形，主要用于分析数据内部的分布状态和
分散状态。直方图主要用于查看个分组数据的数量分布，以及各个分组数据之间的数量比较。饼图
倾向于查看各分组数据在总数据中的占比。箱线图的主要作用是发现整体数据的分布分散情况。
基于以上统计图的特征，并结合分析得到结果，我们可以利用 matplotlib 可以绘制出各年份男/女/人口数目及城乡人口数目的直方图、散点图、折线图；男女人口比例及城乡人口比例的饼图；每个特征的箱线图。通过这些不同种类的可视化图，可以直观地发现不同特征增长或减少的速率变化情况，也可以发现人口结构的变化，帮助有关方面人员做出决策及管理措施。
（2）实验过程
·数据读取 
首先，通过 numpy.load 读取 populations.npz 文件到 data 变量，并利用 data.files 获取所有标 
签值。提取其中的feature_names数组，视为数据的标签，再提取其中的data数据，作为数据的存在位置。再根据需要创建所需的标签以便后续使用。然后通过将数据的标签和数值分离，分别赋予给 name 和 values 变量，其中 values 利用切片方法将数据改为年份升序，并去除无效项。 

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/73a26a7d-cd5a-4591-ac80-3e92268d33b3)

·模型构建 
分别绘制散点图和折线图、柱状图、饼图和箱线图。散点图和折线图用于显示1996年到2015年各类人口数量的变化，包括年末总人口、男性人口、女性人口、城镇总人口乡村人口，不同类别的人口用不同颜色的线和不同的符号来表示。绘制时，横轴取 values 的第 0 列（年份），纵轴取 values 的第 1 列（人 口数），采用特定点和线的图像，设定好图像标题，分别构建统计图。由于构建完成后，发现 
横轴的标签由于字符较长，水平放置会出现重叠现象，于是在 xticks 的参数中设置了 rotation=45 
这一参数，将标签逆时针旋转了 45 度，解决了重叠问题。 
柱状图用于对比1996年到2015年男、女人口数的情况，以及1996年到2015年城、乡人口数的情况，为了方便对比，这里采用对比柱状图的形式，对比的两者用不同的颜色来区分。同时，发现取连续年份得到的数据差异性不大，于是又取了 1996 年和 2015 年的数据作为两个比较对象，这样可以让区别更加明显。

饼图用于显示部分与部分、部分与整体之间的比例关系，最开始选择绘制1996年和2015年男、女人口和城、乡人口的饼状图，后来经过实际考虑认为仅凭第一年和最后一年的饼图难以分析变化情况，便增加了2002年和2009年的饼状图，在控制饼图数量既不会太少也不会太多的前提下，查看和分析人口占比的变化情况。
最后绘制1996年到2015年人口数据的箱线图。箱线图可以直观地识别数据集中的异常值(查看离群点)，也可以帮助我们判断数据集的数据离散程度和偏向。
·模型验证与评价 
通过所有的统计图分析 1996~2015 年人口数据，可以发现人口结构的变化，在初步完成绘制后，根据实验需求和要求，以及绘制图片的实际观感，对部分参数进行调整和完善，对错误和不合理之处进行完善。最终生成出的图像数据清晰直观，符合数据本身的变化趋势，证明该模型是有效的
（3）实验结果分析
本次实验共绘制了1996~2015年人口数据特征间关系散点图和折线图、1996~2015年人口数对比柱状图、1996年和2015年各类人口直方图、1996年和2015年各类人口饼图和1996-2015年各特征人口箱线图共五张统计图。

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/16265005-3147-4148-887b-1bc889ee3a87)

图一：1996~2015年人口数据特征间关系散点图和折线图
从图 一 中可以看出： 
1、1996 年-2015 年间总人口呈现平稳上升趋势，由 12 亿缓慢上升至 14 亿，增长速度逐渐放缓。 
这说明我国从 1982 年起开始实行的计划生育政策有明显成效，人口得到有效控制，增长速度放缓， 
但人口基数大这一问题仍有待改善。 
2、在性别比例分布方面，男性和女性的总人口数总体缓慢上升，男性总人口数略大于女性总人 
口数，二者差距先增大后减小。这说明我国传统的“重男轻女”现象有所改观。虽然截至 2015 年男女性别比例仍未达到理想的 1.01：1 的情况，依此趋势，未来有望达到理想比例。 
3、在城乡人口比例分布方面，城镇人口快速增长，2015 年的城镇人口数量超过 1996 年的城镇 
人口数量的 200%，而乡村人口数量也有明显下降，降幅超过 26%。1996 年我国城镇化率仅为 33% 
不到，经过 16 年的发展，城镇化率已经提高到了将近 60%。这说明我国在这些年间人口大量从乡村涌入城市，城市化进程因此大大加快，城镇化率大幅增加，居民平均生活条件有很大改善。

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/5bdbf3a4-f143-40bf-9539-a958abfe33f3)

图二：1996年和2015年各类人口直方图

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/006d5333-2f92-4a1f-b9a2-5545d4b67c4d)

图三：1996~2015年人口数对比柱状图

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/2dd24a13-2409-4300-aa96-2c1fc3d54809)

图四：1996年和2015年各类人口饼图
从图二、图三、图四中可以看出：
通过分析柱状图可以看出，从1996年到2015年，总人口数呈缓慢下降趋势，男、女人口数量随总人口数下降也在缓慢下降，男性人口始终略多于女性人口，但从2012年开始两者的数量在不断接近，男女比例开始逐渐趋近于1:1，，在性别比例分布方面，男性总人口数略大于女性总人口数，但二者差距增大，性别比从 1996 年的 1.032 上升到 2015 年的 1.049。截至 2015 年男女性别比例仍未达到 
理想的 1.01 的比例。在城镇人口和乡村人口方面，1996年到2015年，城镇人口数量不断增加，同时乡村人口数量不断减少，说明我国的城镇化正在稳步推进，在城乡人口比例分布方面，城镇人口快速增长，城镇人口数由 1996 年的 40000 万人不到上升到 2015 年的 80000 万人，乡村人口数量也有明显下降，降幅为 25.6%。1996 年我国城镇化率仅为 30.5%，经过 16 年的发展，城镇化率已经提高到了 56.1%。这说明我国在这些年间人口大量从乡村涌入城市，城市化进程因此大大加快，城镇化率大幅增加，居民平均生活条件有很大改善。


 
![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/16435738-9fd4-4fb6-b48e-e2e56e0b0243)


图五：1996-2015年各特征人口箱线图
从图五可以看出：
1、1996 年-2015 年间总人口数是呈现上升趋势的，由 12 亿缓慢上升至 14 亿，增长速度逐渐放 
缓。人口得到有效控制。 
2、在性别比例分布方面，男性和女性的总人口数总体缓慢上升，男性总人口数略大于女性总人 
口数。其中，男性人口在 1996 年-2015 年间增长速度降低，女性增长速度提高。 
3、在城乡人口比例分布方面，城镇人口快速增长，乡村人口数量也有明显下降。这说明我国在 
这些年间人口大量从乡村涌入城市，城市化进程因此大大加快，城镇化率大幅增加，居民平均生活 
条件有很大改善
4、在中位数上，男性人口总体略多于女性人口，城镇人口总体多于乡村人口。人口总数量以及男女人口数量较为稳定，而城镇人口数量和乡村人口数量有较大的变化，特别是乡村人口的数量，变化非常大。
4.2.2 数据预处理
（1）问题理解
使用python读取alarm.csv、ele_loss.csv、missing_data.csv以及model.csv中的数据，并按照需求和一定步骤对数据进行操作和分析。需要使用缺失值识别方法，对缺失值数据处理的方法、主键合并的几种方法、多个键值的主键合并的方法，同时需要使用到数据标准化的原理并应用数据标准化的方法。
用户用电量数据呈现一定的周期性关系，missing_data.csv表中存放了用户A、用户B和用户C的用电量数据，其中存在缺失值，需要进行缺失值插补才能进行下一步分析。线路线损数据、线路用电量趋势下降数据和线路告警数据是识别用户窃电漏电与否的3个重要特征，需要对由线路编号(ID)和时间(date)两个键值构成的主键进行合并。
另一方面，由于不同特征之间往往具有不同的量纲，由此所造成的数值间的差异可能很
大，再设计空间距离计算或者梯度下降法等情况时，不对其进行处理会影响到数据分析结果
的准确性。为了消除特征之间量纲和取值范围差异可能会造成的影响，需要对数据进行标准
化处理。因此，要对线路线损特征、线路用电量趋势下降特征、线路告警特征进行标准化。
（2）实验过程
·数据读取
在读取数据之前，首先需要插补用户用电量数据的缺失值。先将 missing_data.csv中的数据利用 read_csv 方法读取到 data 变量，然后利用 isnull 方法判断缺失值所在位置。接下来，利用 Scipy 库中的 interpolate 模块中的 Lagrange 对数据进行拉格朗日插值

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/4392c7a2-0253-435c-818a-a4e0f46dd849)

·模型构建
读取完数据后，接下来是对由线路编号(ID)和时间(date)两个键值构成的主键进行
合并。先将 ele_loss.csv 和 alarm.csv 中的数据分别保存到 loss 和 alert 变量中。然后使用pandas 里的 merge 函数对’ID’和’date’两个主键进行合并，连接方式为 inner。
最后是对数据进行标准化，本实验中采用了标准差标准化数据方法，转化公式为： 

其中，X 为原始数据的均值（通过 data.mean()获取），δ为原始数据的标准差。经过此方法处理之后的数据其均值为 0，标准差为 1。
（3）实验结果分析
通过对比插值前后的数据，插值后的数据不存在空值，说明插值成功

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/d3d1f255-1f61-42a8-9d2a-9201f757d316)
![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/9a72fd83-f5a8-48af-80ad-dc9e711b4792)
![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/fff33202-d6a1-435c-8c35-fb098206e864)

插值前后数据对比

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/6f48de99-f9b5-4858-b399-db8d370764c9)


date：日期，ele：线路用电量趋势下降数据，loss：线路线损数据，alarm：线路告警数据。
通过分析合并后的数据表，可以看出9月16日和17日的警告较多。ele和alarm的关系不明显，但ele随着时间在不断变大。loss和ele相似，也在随着时间逐渐变大。可以推断出随着使用时间增长，线路的用电量趋势下降数据和线损数据会不断变大，故障的概率也会越来越大。

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/ed0be479-3e1c-4c5d-a0a8-4c9e760c7f9b)
![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/7f88dcca-9959-46b5-bb07-63c400632b08)


标准化前的数据

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/136beb07-e7dd-4f88-82ee-fea3abc8bad5)
![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/43570c81-a9f2-45d5-8d23-b0c0df064725)


通过观察结果可以发现，可以看出当警告类指标接近0或更高时，用户窃漏电的可能性更大。线损指标和电量趋势下降指标这两者和用户窃漏电也有一定关系，在多数情况下，线损指标越大，或电量趋势下降指标越接近1，用户窃漏电的可能性越大。并且标准差标准化后的值区间不局限于[0, 1]，并且存在负值。同 时也不难发现，标准差标准化和理查标准化一样不会改变数据的分布情况。
4.2.3 数据分析
（1）问题理解
wine.数据集和wine_quality数据集是两份和酒有关的数据集。wine数据集包含3种不同起源的葡萄酒的记录，共178条。其中，每个特征对应葡萄酒的每种化学成分，并且都属于连续型数据。通过化学分析可以推断葡萄酒的起源。wine_quality数据集共有4898个观察值、11个输入特征和一个标签。其中，不同类的观察值数量不等，所有特征为连续型数据。通过酒的各类化学成分，预测该葡萄酒的评分。
将样本分成独立的 3 部分：训练集、验证集和测试集。其中，训练集用于估计模型，验证集用于确定网络结构或者控制模型复杂程度的参数，而测试集用于检验最优模型的性能。使用训练集训练 SVM 分类模型，并使用训练完成的模型预测测试集的葡萄酒类别归属。由于分类模型对测试集进行预测而得出的准确率并不能很好的反映模型的性能，为了有效判断一个预测模型的性能表现，需要结合真实值计算出精确率、召回率、F1 值和 Cohen’s Kappa 系数等指标来衡量。
wine数据集的葡萄酒总共分为3种，通过将wine数据集的数据进行聚类，聚集为3个簇，能够实现葡萄酒的类别划分。将wine数据集划分为训练集和测试集，使用训练集训练SVM分类模型，并使用训练完成的模型预测测试集的葡萄酒类别归属。wine_quality数据集的葡萄酒评分在1-10之间，构建线性回归模型与梯度提升回归模型，训练wine_quality数据集的训练集数据，训练完成后预测测试集的葡萄酒评分。结合真实评分，评价构建的两个回归模型的好坏。
（2）实验过程
·数据读取
首先，使用 pandas 库中的 read_csv 方法分别读取 wine 数据集和 wine_quality 数据
集，然后通过 wine[“Class”].values 得到 wine 的所有类名；通过 iloc 切片获取所有数据；
通过 wine.columns 获得所有标签，即可将两个数据集的数据和标签拆分开。数划分为训练集和测试集。其中，训练集用于估计模型，验证集标准化wine数据集和wine_qualiy数据集。之后对两个数据集进行PCA降维用于 确定网络结构或者控制模型复杂程度的参数，而测试集则用于检验最优模型的性能。
·模型构建
在聚类数目为2-10类时，确定最优聚类数目。求取模型的轮廓系数，绘制轮廓系数折线图，确定最优聚类数目。求取Calinski-Harabasz指数，确定最优聚类数目
读取wine数据集，区分标签和数据。将wine数据集划分为训练集和测试集。使用离差标准化方法标准化wine数据集。构建SVM模型，并预测测试集结果。打印建立的SVM模型和出分类报告，评价分类模型性能
根据 wine_quality 数据集处理的结果，构建线性回归模型、梯度提升回归模型，结合真实评分和预测评分，计算均方误差，中值绝对误差，可解释方差值，根据得分，判定模型的性能优劣
（3）实验结果分析

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/6febc05d-0f83-4720-aa4c-6adef0f1fa34)
![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/e8352798-959e-4287-924b-da30954f1edd)


FMI评分运行结果和绘制的轮廓系数折线图
通过程序运行结果可以看出，由wine数据集构建聚类数目为3的K-Means模型，求得的FMI为0.924119。在聚类数目为2-10类时，iris数据聚3类的FMI评分值最高，约为0.93，所以iris数据聚类中聚3类为最优聚类。seeds数据聚10类的calinski-harabasz指数相对其它9类最高，约为1475.99，所以seeds数据聚类中聚10类为最优聚类。另外，通过轮廓系数折线图也可以看出，聚3类的畸变程度相对其它数据聚最大，也可以得出iris数据聚类中聚3类为最优聚类的结论。

不同聚类数目下 wine 数据集的 Calinski-Harabasz 指数的结果图
由前三点可知，最优聚类结果为 3 类，所以将 wine 数据聚集为 3 类的可视化。

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/9d894e24-d1d2-4e83-b079-ef52d579326b)

所以综上所述将 wine 数据聚集为 3类的结果是最好的。

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/791ebeab-5309-4313-8f74-b255a01b82da)

由于 ROC 曲线横纵坐标范围为[0, 1]，通常情况下，ROC 曲线与 x 轴形成的面积越
大，与 y 轴靠得越近，表示模型性能越好。因此，从图中可以看出，SVM 的预测 wine
数据是有效的

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/e4094cb0-8c62-4a0f-aa1e-bdb5c49d81da)
![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/e096b2ca-dbd4-4900-9fa8-b125bff4ae7a)


从两个模型预测图中中可以看出，除了部分预测值和原值相差较大以外，绝大多数拟合效果良好，与实际偏差不大。

![image](https://github.com/ShYyy121/Python_DataAnalysis_LessonWork/assets/145829122/6e509cf5-ad25-4cb3-b58c-bca4a72da74c)

通过对比两种线性回归模型和梯度提升回归模型的结果，可以看出两者的平均绝对误差相差无几，而线性回归模型的均方误差和中值绝对误差高于梯度提升回归模型，梯度提升回归模型的可解释方差和R方值明显高于线性回归模型。综合几种结果的对比可以看出，在对当前数据集的分析中，梯度提升回归模型的平均绝对误差、均方误差、中值绝对误差更接近最优值0.0，可解释方差值和R方值更接近最优值1.0。所以梯度提升回归模型优于线性回归模型，可靠性更高。
五．实验总结
完成四个实验的内容后，我认为我有以下不足：

井字棋：对于tkinter的时候仍不够熟练，对界面的设计需要学习，在开始界面的时候，可以将开始游戏的按钮固定在一个固定位置，但是这样布局会产生变化，会导致生成九宫格按钮的代码也需要修改，所以对于tkinter的使用仍需进一步熟练，对于算法的设计，个人认为可以利用机器学习的方式，让其自己学习而不是自己设定一套固定的算法，从而完成真正的人机对战。

数据分析实验一：制图还不够美观和合理。对于折线图和散点图，个人认为过于丑陋，可能使眼色过于丰富和鲜艳，同时长度过于长，可以适当加大宽度，增加美观性。同时，相关参数还可以继续优化，使绘制的统计图可以更好地体现出变化和趋势。

数据分析实验二：是通过拉格朗日插值法对原数据中的缺失值进行补充，但从结果图像中可以
看出，该插值方法并不是那么满足需求，而且很容易产生噪点，影响整个数据的分布情况，是因为当插值点比较多的时候，拉格朗日插值多项式的次数可能会很高，因此具有数值不稳定的特点，也就是说尽管在已知的几个点取到给定的数值，但在附近却会和“实际上”的值之间有很大的偏差，解决可以是分段用较低次数的插值多项式。另外，也可以通过绘制一些图表实现数据可视化，使数据更为直观，方便得出更为合理的结论。

数据分析实验三：得到的回归模型评分仍有较大改进空间，之后可以研究其他模型，提高正确率。同时对于sklearn的使用仍需要进一步熟练和掌握，对于其内部的原理和算法仍需要进一步的学习。对实验结果的分析也不够透彻，不能够根据代码运行结果和实验结果较为客观地得出一些结论，或对模型进行评价，需要在以后多加使用和练习。
