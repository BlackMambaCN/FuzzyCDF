import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
score = np.loadtxt("..\\math2015\\FrcSub\\data.txt")
q = np.loadtxt("..\\math2015\\FrcSub\\q.txt")
stuNum = len(score)
print(stuNum)
questionNum = len(score[0])
knowledgePoint = len(q[0])
N = np.loadtxt("..\\FuzzyN.txt")
alpha = np.loadtxt("..\\FuzzyAlpha.txt")
S = np.loadtxt("..\\FuzzyS.txt")
G = np.loadtxt("..\\FuzzyG.txt")
# N = np.zeros([stuNum, questionNum])
file = open("../math2015/FrcSub/problemdesc.txt")
fileStr = file.readlines()  # string类型的list
desc = []
for i in fileStr:
    i = i[:-1]  # 去掉换行符
    i = i.strip()  # 去掉字符串两端的空白字符
    i = i.split("\t")  # 以\t为分隔符分隔每个字符串
    desc.append(i)
    # print(i)
file.close()
desc = np.array(desc)
desc = np.delete(desc, 0, axis=0)
desc = np.delete(desc, 0, axis=1)
desc = np.delete(desc, 1, axis=1)
desc = desc.reshape(-1)

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

predictscore = np.copy((1 - S) * N + G * (1 - N))
rmse = (score - predictscore) * (score - predictscore)
rmse = np.sqrt(np.sum(rmse, axis=0) / stuNum)
sumRmse = np.sqrt(np.sum((score - predictscore) * (score - predictscore)) / (stuNum * questionNum))
for i in range(len(predictscore)):
    for j in range(len(predictscore[i])):
        if desc[j] == 'Obj':
            if predictscore[i][j] > 0.5:
                predictscore[i][j] = 1
            else:
                predictscore[i][j] = 0
rmse2 = (score - predictscore) * (score - predictscore)
rmse2 = np.sqrt(np.sum(rmse2, axis=0) / stuNum)
sumRmse2 = np.sqrt(np.sum((score - predictscore) * (score - predictscore)) / (stuNum * questionNum))
print("RMSE:", rmse)
print("RMSE(Wu):", rmse2)
print("sumRMSE", sumRmse)
print("sumRMSE2(Wu)", sumRmse2)
# print(np.sqrt(np.sum((score - predictscore) * (score - predictscore))/stuNum))
x = [i for i in range(questionNum)]
x = np.array(x)
x = x + 1
plt.xticks(x)
plt.plot(x, rmse, 'ro-', label='R-Fuzzy')
plt.plot(x, rmse2, 'bs-', label='FuzzyCDF')
plt.xlabel("题目号")
plt.ylabel("RMSE")
plt.legend(loc='upper right')
plt.show()
