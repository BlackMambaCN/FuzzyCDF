import numpy as np
import matplotlib.pyplot as plt

from snack import getDESC

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
score = np.loadtxt("..\\math2015\\Math2\\data.txt")
q = np.loadtxt("..\\math2015\\Math2\\q.txt")
stuNum = len(score)
print(stuNum)
questionNum = len(score[0])
knowledgePoint = len(q[0])
N = np.loadtxt("..\\FuzzyN.txt")
alpha = np.loadtxt("..\\FuzzyAlpha.txt")
S = np.loadtxt("..\\FuzzyS.txt")
G = np.loadtxt("..\\FuzzyG.txt")
test_index = np.loadtxt("..\\FuzzyTest_Index.txt")
test_index = test_index.astype(np.int)
# N = np.zeros([stuNum, questionNum])
desc, subqueIndex, objqueIndex, subqusNum, objqusNum = getDESC.getdesc("..\\math2015\\Math2\\problemdesc.txt")
leeRMSE = np.array(
    [0.49, 0.406, 0.463, 0.46, 0.54, 0.457, 0.44, 0.505, 0.6, 0.42, 0.37, 0.50, 0.40, 0.44, 0.44, 0.53, 0.41, 0.44,
     0.41, 0.38])
# for i in range(stuNum):
#     for j in range(questionNum):
#         temp = []  # 将每道题考察的知识点的qjk加进来取最大/最小作为学生的对该题目的认知状态
#         for k in range(knowledgePoint):
#             if q[j][k] == 1:
#                 temp.append(alpha[i][k])
#         # print(temp)
#         if desc[j] == 'Obj':  # 客观题取最小值
#             N[i][j] = min(temp)
#         elif desc[j] == 'Sub':  # 主观题取最大值
#             N[i][j] = max(temp)

predictscore = (1 - S) * N + G * (1 - N)
rmse = (score[test_index] - predictscore[test_index]) * (score[test_index] - predictscore[test_index])
rmse = np.sqrt(np.sum(rmse, axis=0) / len(test_index))
sumRmse = np.sqrt(np.sum((score[test_index] - predictscore[test_index])
                         * (score[test_index] - predictscore[test_index])) / (len(test_index) * questionNum))
mark = predictscore >= 0.5
predictscore233 = np.copy(predictscore)
predictscore233[mark] = 1
mark = predictscore < 0.5
predictscore233[mark] = 0
if len(subqueIndex) > 0:
    predictscore233[:, subqueIndex] = predictscore[:, subqueIndex]
rmse2 = (score[test_index] - predictscore233[test_index]) * (score[test_index] - predictscore233[test_index])
print(np.sum(rmse2, axis=0).shape)
rmse2 = np.sqrt(np.sum(rmse2, axis=0) / len(test_index))
sumRmse2 = np.sqrt(np.sum((score[test_index] - predictscore233[test_index])
                          * (score[test_index] - predictscore233[test_index])) / (len(test_index) * questionNum))
rmse3 = (score - predictscore233) * (score - predictscore233)
rmse3 = np.sqrt(np.sum(rmse3, axis=0) / stuNum)
sumRmse3 = np.sqrt(np.sum((score - predictscore233) * (score - predictscore233)) / (stuNum * questionNum))
print("RMSE(无阈值）:", rmse)
print("RMSE(阈值，测试集):", rmse2)
print("averageRMSE（无阈值）", sumRmse)
print("averageRMSE2(阈值，测试集）", sumRmse2)
print(np.sum(rmse2) / questionNum)
# np.savetxt('predictscore.txt', predictscore, fmt='%0.2f')
# np.savetxt('predictscore233.txt', predictscore233, fmt='%0.2f')
# print(np.sqrt(np.sum((score - predictscore) * (score - predictscore))/stuNum))
x = [i for i in range(questionNum)]
x = np.array(x)
x = x + 1
plt.xticks(x)
plt.plot(x, rmse3, 'ro-', label='测试集+训练集')
plt.plot(x, rmse2, 'bs-', label='测试集')
# plt.plot(x, leeRMSE, 'g-', label='交叉验证')
plt.xlabel("题目号")
plt.ylabel("RMSE")
plt.legend(loc='upper right')
plt.show()
