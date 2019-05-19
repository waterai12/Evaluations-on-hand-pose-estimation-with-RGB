#!/usr/bin/python
#coding=utf-8
"""
===============================================================
sklearn
该文件主要是通过自己理解了的程序来编译实现教程里面的每个例子
===============================================================
"""

print(__doc__)
from sklearn import linear_model, datasets
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC, Lasso
from sklearn.preprocessing import PolynomialFeatures #多项式
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline    #预处理可以通 过 Pipeline 工具进行简化
import time
import numpy as np
import matplotlib.pyplot as plt  #绘图
from sklearn.metrics import r2_score
#--------------------------------------------------------------
#Step1.采集数据集
#--------------------------------------------------------------
#1. 线性回归数据集
#1.1 通过散点来求得出回归方程
# X = np.array([[0],[1],[2]])
# y = np.arange(0,3)
#1.2 建立2元线性模型通过矩阵αβ=γ来求得β的最优解
# X = np.array([[0, 0], [1, 1], [2, 2]])
# X_test = X
# y = np.arange(0,3)
# y_test = y
#1.3 通过糖尿病数据集来建立一元线性多项式数据模型 y=f(x)
# diabetes = datasets.load_diabetes()          #diabetes.data.shape=(442,10)
# diabetes_X = diabetes.data[:, np.newaxis, 2] #只去其中第三列数据使用  diabetes_X.shape=(442,1)
# X = diabetes_X[:-20] #数组切片,无代表最大,(0,sum-20)
# X_test = diabetes_X[-20:]  #数组切片,即(sum-20,sum)
# y = diabetes.target[:-20]
# y_test = diabetes.target[-20:]

#2 Ridge回归数据集
# X = 1. / (np.arange(1, 6) + np.arange(0, 5)[:, np.newaxis])
# X_test = 1. / (np.arange(1, 6) + np.arange(0, 5)[:, np.newaxis])
# y = np.ones(5)
# y_test = np.ones(5)

#2 lasso回归数据集
# X = np.array([[0, 0], [1, 1]])
# X_test = X
# y = np.arange(0,2)
# y_test = y
#2.2 lasso稀疏样本 (用来分析lasso的最佳参数的数据集)
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
rng = np.random.RandomState(42)
X = np.c_[X, rng.randn(X.shape[0], 14)]  # add some bad features
X /= np.sqrt(np.sum(X ** 2, axis=0))  # normalize data as done by Lars to allow for comparison
#2.3 手动生成一些随机稀疏数据(弹性网络和lasso的对比)
# print np.random.seed(42)
# n_samples, n_features = 50, 200
# X = np.random.randn(n_samples, n_features)
# coef = 3 * np.random.randn(n_features)  # 这个就是实际的参数
# inds = np.arange(n_features)
# np.random.shuffle(inds)  # 打乱
# coef[inds[10:]] = 0  # 生成稀疏数据
# y = np.dot(X, coef)  # 参数与本地点乘
# # 来点噪音
# y += 0.01 * np.random.normal((n_samples,))
#
# X_train, y_train = X[:n_samples / 2], y[:n_samples / 2]
# X_test, y_test = X[n_samples / 2:], y[n_samples / 2:]


#-------------------------------
#查看数据集的信息
#-------------------------------
print '数据集X:\n',X
print '数据集X的维度:',X.ndim
print '数据集X的shape',X.shape
print '数据集y:',y
print '数据集y的维度:',y.ndim
print '数据集y的shape',y.shape





#-----------------------------------------------------------------------------------------
#Step2.选择建模方法(X --> y的一个映射关系y=f(x)),fit()函数就是通过X,y的数据来确定y=f(x)模型里面的参数
       #建模方法将决定fit函数以何种方式去求出参数,such as 线性回归中采用的是梯度下降的方式去计算参数值的
#-----------------------------------------------------------------------------------------
# 1.linear_model.LinearRegression方法
# reg0 = linear_model.LinearRegression()
# reg0.fit(X, y)


# 2.linear_model.Ridge方法
#ridge = linear_model.Ridge(alpha=0.1, fit_intercept=False)
# ridge = linear_model.RidgeCV(alphas=[5.99484250e-09, 0.1,1,10,100], fit_intercept=False) #CV可以根据得分来选择提供的alphas
# ridge.fit(X, y)


#3.lasso回归 #可自动选择最佳alpha
lasso = linear_model.Lasso(alpha = 0.1)
lasso.fit(X,y)
#3.1 lassCV回归,#coordinate descent
t1 = time.time()
LassoCV = LassoCV(cv=20).fit(X,y)
t_lasso_cv = time.time() - t1
#3.2 LassoLarsCV回归 #least angle regression    速度更快一些,因为它的迭代次数仅仅只有一次
t1 = time.time()
LassoLarsCV = LassoLarsCV(cv=20).fit(X,y)
t_lasso_lars_cv = time.time() - t1


#4.polynomial linear回归
#4.1 预处理 degree=3 #最高次为3
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

model = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression(fit_intercept=False))])
model = model.fit(X, y)
print(model.named_steps['linear'].coef_)


X_text=np.linspace(-3, 3, 100).reshape(100, 1)
y_text = model.predict(X_text)
plt.figure(1)
plt.plot(X_text, y_text, "r-", linewidth=2, label="Predictions")
plt.plot(X, y, "b.")


#--------------------------------------------
#多种alpha的曲线函数
#--------------------------------------------
# n_alphas = 100
# alphas = np.logspace(-5, 1, n_alphas)
# coefs = []
# coef  = []
# coef1 = []
# score = []
# score1 = []
# for a in alphas:
#     ridge = linear_model.Ridge(alpha=a)
#     ridge.fit(X, y)
#     coefs.append(ridge.coef_)
#     lasso = linear_model.Lasso(alpha=a)
#     lasso.fit(X, y)
#     coef.append(lasso.coef_)
#     reg0 = linear_model.LinearRegression()
#     reg0.fit(X, y)
#     coef1.append(reg0.coef_)
#     score.append(np.array(r2_score(y_test, lasso.predict(X))))
#     score1.append(np.array(r2_score(y_test, ridge.predict(X))))

#-------------------------------
#查看模型的信息(predict,score,alphas)
#-------------------------------
# #线性回归,score
# Y_pred = np.array(reg0.predict(X_test))
# print '线性回归预测值:',Y_pred
# print '线性回归测试集得分',r2_score(y_test,Y_pred)
# print '线性回归系数',reg0.coef_

# # #岭回归,L2正则化参数
#Y_pred = np.array(ridge.predict(X_test))
# print '岭回归选取的alphas',ridge.alpha_
# print '岭回归预测值:',Y_pred
# print '岭回归测试集得分',r2_score(y_test,Y_pred)
#print '岭回归系数',ridge.coef_

# #lasso回归,L1正则化参数,可以用来执行特征选择
# Y_pred = np.array(lasso.predict(X))
# #print 'lasso回归选取的alphas',lasso.alpha_
# #print 'lasso回归预测值:',Y_pred
# print 'lasso回归测试集得分',r2_score(y,Y_pred)
# print 'lasso回归系数',lasso.coef_

# #polynomial linear


# 函数显示模型信息
# def model_infotmation(model,name,X,y):
#     Y_pred = np.array(model.predict(X))
#     model_score = r2_score(y, Y_pred)
#     #print '%s回归预测值:'%name,Y_pred
#     print '%s回归选取的alphas'%name,model.alpha_
#     print '%s回归测试集得分'%name, model_score
#     print '%s回归的系数'%name, model.coef_
#     return model_score
# LassoCV_score = model_infotmation(LassoCV,'LassoCV',X,y)
# LassoLarsCV_score = model_infotmation(LassoLarsCV,'LassoLarsCV',X,y)

#-----------------------------------------------------------------------------------------
#Step3.matplotlib显示
#-----------------------------------------------------------------------------------------
# z = np.linspace(0,0.1,50) #从-1到1之间分配50个点
# plt.scatter(X,y,s=8)   #绘制散点图
# plt.plot(z,reg0.predict(z.reshape(-1,1)),color='g',linewidth=5,linestyle='-')
# plt.plot(z,ridge.predict(z.reshape(-1,1)),color='b',linewidth=3,linestyle='-')
# plt.plot(z,lasso.predict(z.reshape(-1,1)),color='red',linewidth=3,linestyle='--')
# plt.show()
# ax = plt.gca()
# ax.set_color_cycle(['r'])
# ax.plot(alphas, coefs)
# ax.plot(alphas, coef,linewidth=1,linestyle='--')
# plt.show()
#---------------------------------------------------------
#显示多条线段,x,y的多维数组要对应好多条曲线
#---------------------------------------------------------
# ax = plt.gca()
# ax.plot(alphas, coefs,linewidth=2,linestyle='-',color='g',label='ridge.coef_')
# ax.plot(alphas, coef,linewidth=2,linestyle='-',color='y',label='lasso.coef_')
# ax.plot(alphas,coef1,linewidth=2,linestyle='-',color='r',label='linear.coef_')
# ax.plot(alphas,score,linewidth=1,linestyle='--',color='y',label='lasso.score')
# ax.plot(alphas,score1,linewidth=1,linestyle='--',color='g',label='ridge.score')\
#
# ax.set_xscale('log')
# ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
# plt.xlabel('alpha')
# plt.ylabel('weights')
# plt.title('Ridge coe fficients as a function of the regularization')
# plt.axis('tight')
# plt.show()
#-------------------------------------------------------------------

#---------------------------------------------------------
#显示多个图形(用来分析lasso的最佳参数的数据集)
#---------------------------------------------------------
# plt.figure(2)
# def plt_figure(model,name,alphas_,time,model_score):
#     model.cv_alphas_=alphas_
#     m_log_alphas = -np.log10(model.cv_alphas_)
#     plt.figure()#创建一个子图
#
#     xmin, xmax = 0, 3
#     ymin, ymax = 2300, 3800
#     plt.plot(m_log_alphas, model.mse_path_, ':')
#     plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',label=name+'score_: %.5f'%model_score, linewidth=2)
#     plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',label='lpha: CaV estimate')
#     plt.legend() #添加图例
#     plt.xlabel('-log(alpha:%.5f'%-np.log10(model.alpha_))
#     plt.ylabel('Mean square error')
#     plt.title('Mean square error on each fold''(train time: %.2fs)' % time)
#     plt.axis('tight')
#     plt.xlim(xmin, xmax)
#     plt.ylim(ymin, ymax)
# plt_figure(LassoCV,'LassoCV',LassoCV.alphas_,t_lasso_cv,LassoCV_score)
# plt_figure(LassoLarsCV,'LassoLarsCV',LassoLarsCV.cv_alphas_,t_lasso_lars_cv,LassoLarsCV_score)
plt.show()



















