import numpy as np
from sklearn.model_selection import KFold

a = np.arange(10)
b = np.ones([10, 10])
c = [0, 1]
print(a)
kfold = KFold(n_splits=10, shuffle=False)  # n_splits表示划分为几块
index = kfold.split(X=a)  # 返回分类后的数据集的索引
for train_index, test_index in index:
    print(train_index)
    print(test_index)
    b[test_index[0]][c] = 0
bb = b.copy()
bb[:][0] = 0
print(bb)
print(int(21*0.8))
# temp = []
# temp.append(1.56)
# temp.append(2.47)
# print(temp)
# print(min(temp))
# print(type(min(temp)))
# print(int(536 * 0.1))
