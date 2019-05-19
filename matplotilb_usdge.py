#!/usr/bin/python
#coding=utf-8
print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause
import matplotlib.pyplot as plt  #绘图
import numpy as np

# #---------------------------------------------------
# #1.figure
# #---------------------------------------------------
# x = np.linspace(-1,1,50) #从-1到1之间分配50个点
# y1 = 2*x+1
# y2 = x**2
# plt.figure(2)
# # plt.plot(x,y1,color='red',linewidth=3,linestyle='--')
# # plt.plot(x,y2,color='b',linewidth=3,linestyle='--')
# #---------------------------------------------------
# #2.axis of coordinates
# #---------------------------------------------------
# plt.xlim((-1,2))
# plt.ylim((-2,3))
# plt.xlabel('I am x')
# plt.ylabel('I am y')
# plt.title('axis of coordinates')
# new_ticks = np.linspace(-1,2,5)
# print(new_ticks)
# plt.xticks(new_ticks) #坐标的单位
# plt.yticks(new_ticks,[r'$really\ bad$',r'$bad\ \alpha$',r'$c$',r'$d$',r'$e$'])
# #---------------------------------------------------
# #3.get current axis
# #---------------------------------------------------
# ax = plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom') #下面的轴
# ax.yaxis.set_ticks_position('left') #下面的轴
# ax.spines['bottom'].set_position(('data',0))
# ax.spines['left'].set_position(('data',0))
# #---------------------------------------------------
# #4.legend 图例
# #---------------------------------------------------
# l1, = plt.plot(x,y1,color='red',linewidth=3,linestyle='-',label ='up')
# l2,  = plt.plot(x,y2,color='b',linewidth=3,linestyle='-',label ='down')
# plt.legend(handles = [l1,l2],labels =['aaa','bbb'],loc= 'best')
# #---------------------------------------------------
# #4.Annotation 标注
# #---------------------------------------------------
# x0 = 1
# y0 = 2*x0 + 1
# plt.scatter(x0,y0,s=50,color='b')
# plt.plot([x0,x0],[y0,0],'k--',lw=2.5) #画辅助线
# plt.annotate(r'$2x+1=%s$'%y0,xy=(x0,y0),xycoords='data',xytext=(+30,-30))
#
# #---------------------------------------------------
# #4.scatter
# #---------------------------------------------------
# n = 1024
# X = np.random.normal(0,1,n) # 平均数0,方差１,ｎ个数字
# Y = np.random.normal(0,1,n) # 平均数0,方差１,ｎ个数字
# T = np.arctan2(Y,X)#for color
# plt.figure(4)
# plt.scatter(X,Y,s=75,cmap = plt.cm.cool,edgecolors='k',alpha=0.5)#alpha = 0.5透明度50％
# plt.xlim((-1.5,1.5))
# plt.ylim((-1.5,1.5))
# #---------------------------------------------------
# #5.function
# #---------------------------------------------------
plt.clf()#清空图形窗口
plt.gca()
# #---------------------------------------------------
# #6.根据点的不同标签来给对应的color
# #---------------------------------------------------
labels =[1, 2, 3, 5, 1, 5, 2, 0, 1, 1, 3, 0, 4, 5, 3, 4, 1, 4, 1 ,2, 2 ,0, 3, 1 ,5 ,4, 0, 4 ,4 ,4, 2 ,2 ,4 ,0, 5 ,2 ,3
 ,1 ,0 ,4 ,3 ,2 ,3 ,0 ,0 ,5 ,0 ,4 ,5 ,5 ,3 ,0 ,1 ,2 ,2 ,0 ,5 ,4 ,1 ,1 ,2 ,3 ,1 ,3 ,3, 0 ,0, 2, 4 ,3, 0 ,2 ,5 ,1,
 5 ,3 ,4 ,3 ,2 ,5, 1 ,4 ,3 ,1, 3, 1 ,2, 1 ,1 ,4 ,2, 2, 0, 0, 0, 0, 2, 4, 0, 0]
from itertools import cycle
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(np.arange(6), colors):
    my_members = labels == k
    print '========================'
    print k
    print my_members
    plt.plot(X_blobs[my_members, 0], X_blobs[my_members, 1], col + '.')
plt.show()
# #---------------------------------------------------



from matplotlib.colors import ListedColormap
from sklearn.svm import SVR,SVC
from sklearn import datasets
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
#random_state是随机数的种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, train_size=0.6)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scalered = scaler.transform(X_train)
X_test_scalered = scaler.transform(X_test)
# kernel='linear'时，为线性核，C越大分类效果越好，但有可能会过拟合（defaul C=1）。
# kernel='rbf'时（default），为高斯核，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。
# decision_function_shape='ovr'时，为one v rest，即一个类别与其他类别进行划分，
# decision_function_shape='ovo'时，为one v one，即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。
clf = SVC(C=60, kernel='rbf', gamma=0.2, decision_function_shape='ovr')
clf.fit(X_train_scalered, y_train)
print '训练集',clf.score(X_train_scalered, y_train)
print '测试集',clf.score(X_test_scalered, y_test)
def boundary(model,X):
    h = .02  # step size in the mesh
#1.确定坐标轴范围，x，y轴分别表示两个特征
    x_min, x_max = X_train_scalered[:, 0].min() - 0.5, X_train_scalered[:, 0].max() + 0.5 # 第0列的范围
    y_min, y_max = X_train_scalered[:, 1].min() - 0.5, X_train_scalered[:, 1].max() + 0.5 # 第1列的范围
#2. np.meshgrid  生成网格采样点
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
#3. 生成测试点
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    cmap_light = ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
#4. pcolormesh(x,y,z,cmap)这里参数代入x1, x2, model.predict, cmap=cm_light绘制的是背景。
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
boundary(clf,X)
#5. scatter中edgecolors是指描绘点的边缘色彩，s指描绘点的大小，cmap指点的颜色。
plt.scatter(X_train_scalered[:,0],X_train_scalered[:,1],c = y_train,cmap = plt.cm.cool,edgecolors='k')
plt.scatter(X_test_scalered[:,0],X_test_scalered[:,1],c = y_test,cmap = plt.cm.cool,edgecolors='k')
plt.show()