import numpy as np
import matplotlib.pyplot as plt

from snack import getDESC

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
score = np.loadtxt("..\\math2015\\Math1\\data.txt")
q = np.loadtxt("..\\math2015\\Math1\\q.txt")
stuNum = len(score)
print(stuNum)
questionNum = len(score[0])
knowledgePoint = len(q[0])
N = np.loadtxt("..\\FuzzyN.txt")
alpha = np.loadtxt("..\\FuzzyAlpha.txt")
S = np.loadtxt("..\\FuzzyS.txt")
G = np.loadtxt("..\\FuzzyG.txt")
# N = np.zeros([stuNum, questionNum])
desc, subqueIndex, objqueIndex, subqusNum, objqusNum = getDESC.getdesc("..\\math2015\\Math1\\problemdesc.txt")

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
rmse = (score - predictscore) * (score - predictscore)
rmse = np.sqrt(np.sum(rmse, axis=0) / stuNum)
sumRmse = np.sqrt(np.sum((score - predictscore) * (score - predictscore)) / (stuNum * questionNum))
mark = predictscore >= 0.5
predictscore233 = np.copy(predictscore)
predictscore233[mark] = 1
mark = predictscore < 0.5
predictscore233[mark] = 0
predictscore233[:, subqueIndex] = predictscore[:, subqueIndex]

rmse2 = (score - predictscore233) * (score - predictscore233)
rmse2 = np.sqrt(np.sum(rmse2, axis=0) / stuNum)
sumRmse2 = np.sqrt(np.sum((score - predictscore233) * (score - predictscore233)) / (stuNum * questionNum))
print("RMSE:", rmse)
print("RMSE(Wu):", rmse2)
print("averageRMSE", sumRmse)
print("averageRMSE2(Wu)", sumRmse2)
# np.savetxt('predictscore.txt', predictscore, fmt='%0.2f')
# np.savetxt('predictscore233.txt', predictscore233, fmt='%0.2f')
# print(np.sqrt(np.sum((score - predictscore) * (score - predictscore))/stuNum))
x = [i for i in range(questionNum)]
x = np.array(x)
x = x + 1
plt.xticks(x)
plt.plot(x, rmse, 'ro-', label='不设置阈值')
plt.plot(x, rmse2, 'bs-', label='阈值0.5')
plt.xlabel("题目号")
plt.ylabel("RMSE")
plt.legend(loc='upper right')
plt.show()
