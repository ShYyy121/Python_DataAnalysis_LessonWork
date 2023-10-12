import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import fowlkes_mallows_score
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
# 读取
wine=pd.read_csv("wine.csv")
# 数据中有；分开
winequality=pd.read_csv("winequality.csv",sep=";");

# 标签数据分开
winedata=wine.iloc[:,1:]
wine0=wine['Class']

# wine0=wine["Class"].values
# winedata=wine.iloc[:,1:].values
winefeature=wine.columns[1:]
winequalitydata=winequality.iloc[:,:-1]
winequality0=winequality['quality']
# winequality0=winequality["quality"].values
# winequalitydata=winequality.iloc[:,:11].values
winequalityfeature=winequality.columns[:11]

# 划分训练集和测试集
wine_train,wine_test,wine0_train,wine0_test=train_test_split(winedata,wine0,test_size=0.1,random_state=6)

winequality_train,winequality_test,winequality0_train,winequality0_test=train_test_split(winequalitydata,winequality0,test_size=0.1,random_state=6)

# 数据标准化
scaler  =  StandardScaler().fit(wine_train)
wine_trainScaler=scaler.transform(wine_train)
wine_testScaler=scaler.transform(wine_test)
wine_dataScale = scaler.transform(winedata)
scaler  =  StandardScaler().fit(winequality_train)
winequality_trainScaler=scaler.transform(winequality_train)
winequality_testSCaler=scaler.transform(winequality_test)

# PCA降维
pca = PCA(n_components="mle").fit(wine_trainScaler)
wine_trainPca=pca.transform(wine_trainScaler)
wine_testPca=pca.transform(wine_testScaler)
# print("wine数据集降维前训练集数据的形状为：", wine_trainScaler.shape)
# print("wine数据集降维后训练集数据的形状为：", wine_trainPca.shape)
# print("wine数据集降维后测试集数据的形状为：", wine_testPca.shape)
pca=PCA(n_components="mle").fit(winequality_trainScaler)
winequality_trainPca=pca.transform(winequality_trainScaler)
winequality_testPca=pca.transform(winequality_testSCaler)

# 根据 wine 数据集处理的结果，构建聚类数目为 3 的 K-Means 模型
scale = StandardScaler().fit(winedata)

kmeans = KMeans(n_clusters=3, random_state=6).fit(wine_dataScale)
print('构建的KMeans模型为：\n',kmeans)
score=fowlkes_mallows_score(wine0,kmeans.labels_)
print("wine数据集的FMI:%f"%(score))
# 在聚类数目为 2-10 类时，确定最优聚类数目。
bestNo = 0
bestScore = 0.0
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=33).fit(wine_dataScale)
    score = fowlkes_mallows_score(wine0, kmeans.labels_)
    print("wine数据集聚为%d类FMI评价分值为：%f" % (i, score))
    # 在聚类数目为 2-10 类时，确定最优聚类数目
    if score > bestScore:
        bestNo = i
        bestScore = score

plt.figure(figsize=(10,10))
kmeans = KMeans(n_clusters=bestNo, random_state=343).fit(wine_dataScale)
tsne1 = TSNE(n_components=2, init="random", random_state=232).fit(wine_dataScale)
dataframe = pd.DataFrame(tsne1.embedding_)
dataframe["label"] = kmeans.labels_
data1 = dataframe[dataframe["label"] == 0]
data2 = dataframe[dataframe["label"] == 1]
data3 = dataframe[dataframe["label"] == 2]
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.title("最优聚类结果：wine数据集聚为" + str(bestNo) + "类")
plt.plot(data1[0], data1[1], "bo", data2[0], data2[1], "ro", data3[0], data3[1], "go",alpha=0.5)
plt.show()
# 求取模型的轮廓系数，绘制轮廓系数
silhouetteScore = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=3).fit(wine_dataScale)
    score = silhouette_score(winedata, kmeans.labels_)
    silhouetteScore.append(score)

p = plt.figure(figsize=(10, 21))
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.title("轮廓系数折线图", fontsize=12)
plt.tight_layout(pad=6)
plt.plot(range(2, 11), silhouetteScore, linewidth=1.5, linestyle="-")
plt.savefig("轮廓系数折线图.png")
plt.show()

# 求取Calinski-Harabasz指数，确定最优聚类数目
bestNo = 0
bestScore = 0.0
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=123).fit(wine_dataScale)
    score = calinski_harabasz_score(wine_dataScale, kmeans.labels_)
    print("wine数据聚%d类calinski_harabaz指数为：%f" % (i, score))
    if score > bestScore:
        bestNo = i
        bestScore = score

# 构建 SVM 模型，并预测测试集结果

# 使用离差标准化方法标准化wine数据集
Scaler = MinMaxScaler().fit(wine_train)
wine_trainScaler = Scaler.transform(wine_train)
wine_testScaler = Scaler.transform(wine_test)
# 构建SVM模型，并预测测试集结果
svm = SVC().fit(wine_trainScaler, wine0_train)
wine0_predict = svm.predict(wine_testScaler)

# 打印出分类报告，评价分类模型性能
report = classification_report(wine0_test, wine0_predict)
print("使用SVM预测wine数据的分类报告为：\n", report)

# ROC曲线
X = winedata
Y = wine0
Y = label_binarize(Y, classes=[1, 2, 3])
n_classes = Y.shape[1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=322)
test = OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=344))
Y_test_svcpred = test.fit(X_train, Y_train).decision_function(X_test)
fpr = dict()
tpr = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_test_svcpred[:, i])
plt.figure(figsize=(10, 6))
plt.plot(fpr[0], tpr[0], color="#ff0000")
plt.plot(fpr[1], tpr[1], color="#7fff00")
plt.plot(fpr[2], tpr[2], color="#0000a0")
plt.legend(["class=1", "class=2", "class=3"])
plt.title("ROC曲线")
plt.xlim(0, 1)
plt.ylim(0, 1.1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig("ROC曲线.png")
plt.show()

# 根据wine_quality数据集处理的结果，构建线性回归模型
LR = LinearRegression().fit(winequality_train, winequality0_train)
wine_quality_test_LinearPredict = LR.predict(winequality_test)
print("建立的线性回归模型为：\n", LR)
print("==================分割线==================")

# 根据wine_quality数据集处理的结果，构建梯度提升回归模型
GBR = GradientBoostingRegressor().fit(winequality_train, winequality0_train)
wine_quality_test_GradientPredict = GBR.predict(winequality_test)
print("建立的梯度提升回归树模型为：\n", GBR)
# print("==================分割线==================")

# 结合真实评分和预测评分，计算均方误差，中值绝对误差，可解释方差值
print("wine_quality数据集线性回归模型的平均绝对误差为：\n",
      mean_absolute_error(winequality0_test, wine_quality_test_LinearPredict))
print("wine_quality数据集线性回归模型的均方误差为：\n",
      mean_squared_error(winequality0_test, wine_quality_test_LinearPredict))
print("wine_quality数据集线性回归模型中值绝对误差为：\n",
      median_absolute_error(winequality0_test, wine_quality_test_LinearPredict))
print("wine_quality数据集线性回归模型的可解释方差值为：\n",
      explained_variance_score(winequality0_test, wine_quality_test_LinearPredict))
print("wine_quality数据集线性回归模型的R²值为：\n",
      r2_score(winequality0_test, wine_quality_test_LinearPredict))

# print("==================分割线==================")

print("wine_quality数据集梯度提升回归树模型的平均绝对误差为：\n",
      mean_absolute_error(winequality0_test, wine_quality_test_GradientPredict))
print("wine_quality数据集梯度提升回归树模型的均方误差为：\n",
      mean_squared_error(winequality0_test, wine_quality_test_GradientPredict))
print("wine_quality数据集梯度提升回归树模型中值绝对误差为：\n",
      median_absolute_error(winequality0_test, wine_quality_test_GradientPredict))
print("wine_quality数据集梯度提升回归树模型的可解释方差值为：\n",
      explained_variance_score(winequality0_test, wine_quality_test_GradientPredict))
print("wine_quality数据集梯度提升回归树模型的R²方值为：\n",
      r2_score(winequality0_test, wine_quality_test_GradientPredict))

# 线性回归模型可视化
p1 = plt.figure(figsize=(15, 6))
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.plot(range(winequality0_test.size), winequality0_test, color="#7fff00")
plt.plot(range(winequality0_test.size), wine_quality_test_LinearPredict, color="#ff0000")
plt.legend(["真实值", "预测值"])
plt.title("线性回归模型预测结果")
plt.savefig("线性回归模型预测结果.png")
plt.show()

# 梯度提升回归树模型可视化
p2 = plt.figure(figsize=(15, 6))
plt.plot(range(winequality0_test.size), winequality0_test, color="#7fff00")
plt.plot(range(winequality0_test.size), wine_quality_test_GradientPredict, color="#ff0000")
plt.legend(["真实值", "预测值"])
plt.title("梯度提升回归树模型预测结果")
plt.savefig("梯度提升回归树模型预测结果.png")
plt.show()