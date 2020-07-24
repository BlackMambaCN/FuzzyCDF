import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

epsilon = 1e-16
# file = open('testdata.txt', mode='w')
# file.write('\n')
# file.write(' ')
# file.write("it's a test txt file.")
# file.write("whywhywhywhy")
b = np.random.random([6, 6])  # 创建随机的6x6矩阵
# print(np.random.random(1))  # np.random.random(size)生成size个0-1随机浮点数
c = np.log(b)
c[:, 0] = b[:, 0]
# print(c)
# print(b)
# print(c[:, 0])
# print(b[:, 0])
# print(2 > np.random.random(1))
# print(c[:, 0] - 1)
a = np.random.random([6, 6])
b = np.random.random([1, 6])
c = np.ones([6, 6])
d = np.array([1, 0, 0, 1, 0, 0])
e = float("-inf")
f = np.ones([1, 6])
x = (1 - b) * a + d * (1 - a)
g = np.copy(a)
print(a)
print(b)
g[:, 0] = np.copy(b)
print(g)
print(a)
# print(x, x.shape)
# print(a[:, 1])
# print(np.sum(a, axis=0))
# print(stats.norm.pdf(a[:, 1]))
# print(stats.norm.pdf(a[:, 2]))
# print(np.sum(a, axis=0) + stats.norm.pdf(a[:, 1]))
# P = np.sum(a, axis=0) + stats.norm.pdf(a[:, 1]) + stats.norm.pdf(a[:, 2])
# print(P)
that = np.log(0)
print(np.log(0))
print((0 - that) > 999)
# print(np.exp(float('-inf')))
# print(a)
# print(np.var(a))
A = stats.lognorm.rvs(s=1, loc=0, scale=np.exp(0), size=a.shape)
print(A[0][0])
print(type(np.linspace(0, 1, 10)))

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
rmse = np.loadtxt("..\\FuzzyRMSE.txt")
predictscore = np.loadtxt("..\\FuzzyX.txt")
score = np.loadtxt("..\\math2015\\FrcSub\\data.txt")
sumrmse = (predictscore-score) * (predictscore-score)
sumrmse = np.sqrt(np.sum(sumrmse)/(536 * 20))
print("SumRMSE:" , sumrmse)
x = [i for i in range(20)]
x = np.array(x)
x = x + 1
plt.xticks(x)
plt.plot(x, rmse, 'rs-', label='FuzzyCDF')
plt.xlabel("题目号")
plt.ylabel("RMSE")
plt.legend(loc='upper right')
plt.show()
'''n维矩阵减1维矩阵，会将n维矩阵的每一个维度都减去该一维矩阵'''
