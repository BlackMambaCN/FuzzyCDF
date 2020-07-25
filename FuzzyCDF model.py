import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import style
from snack import fuzzyGetlog
from sklearn.model_selection import KFold

''' numpy里面的等号为引用（先创建numpy对象），深拷贝为np.copy'''
epsilon = 1e-16
'''计算主观题和客观题的数目，根据problemdesc.txt'''
# score = np.loadtxt("math2015\\FrcSub\\data.txt")
# q = np.loadtxt("math2015\\FrcSub\\q.txt")  # 知识点矩阵
# file = open("math2015\\FrcSub\\problemdesc.txt")

score = np.loadtxt("math2015\\Math1\\data.txt")
q = np.loadtxt("math2015\\Math1\\q.txt")  # 知识点矩阵
file = open("math2015\\Math1\\problemdesc.txt")
subqusNum = 0  # 主观题数目
objqusNum = 0  # 客观题数目
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
print(desc)
for i in desc:
    if i == 'Obj':
        objqusNum += 1
    elif i == 'Sub':
        subqusNum += 1

trainscore = score  # 删除列
knowledgePoint = len(q[0])  # 题目考察的知识点
stuNum = len(trainscore)  # 学生数目
questionNum = len(score[0])  # 考试题目数

print("训练集试题数：", questionNum)
print("客观题数：", objqusNum)
print("主观题数:", subqusNum)
print("学生数：", stuNum)
print("知识点数：", knowledgePoint)

'''定义FuzzyCDF所需的变量'''
theta = np.zeros([stuNum, 1])  # 学生潜力
updateTheta = np.zeros(stuNum)
A = np.zeros([stuNum, knowledgePoint])  # aik：知识点k对学生i的区分度
updateA = np.zeros([stuNum, knowledgePoint])  # 更新时暂存矩阵
B = np.zeros([stuNum, knowledgePoint])  # aik：知识点k对学生i的难度
updateB = np.zeros([stuNum, knowledgePoint])
alpha = np.zeros([stuNum, knowledgePoint])  # Alpha ij:学生对知识点的掌握程度
updateAlpha = np.zeros([stuNum, knowledgePoint])
S = np.zeros(questionNum)  # 失误率
G = np.zeros(questionNum)
updateS = np.zeros(questionNum)
updateG = np.zeros(questionNum)
variance = 0  # 主观题计算得分的方差
updateV = 0
N = np.zeros([stuNum, questionNum])
updateN = np.zeros([stuNum, questionNum])

'''初始化各参数的分布参数'''
mu_theta = 0
sig_theta = 1
max_theta = 4
min_theta = -4
mu_a = 0
sig_a = 1
mu_b = 0
sig_b = 1
min_s = 0
max_s = 0.6
min_g = 0
max_g = 0.6
delta = 1 * 1e-2
echo = 5000  # 迭代次数
burnin = 2500

'''初始化估计值'''
ea = np.zeros(A.shape)
eb = np.zeros(B.shape)
etheta = np.zeros(theta.shape)
evariance = 0
ealpha = np.zeros(alpha.shape)
ealpha2 = np.zeros(alpha.shape)
es = np.zeros(S.shape)
eg = np.zeros(G.shape)

'''初始化S，G'''

# tempS = stats.beta.rvs(size=S.shape, a=2, b=1)
tempS = stats.beta.rvs(size=S.shape, a=1, b=2)
tempG = stats.beta.rvs(size=G.shape, a=1, b=2)
S = tempS * (max_s - min_s)
# S = 1 - (min_s + (max_s - min_s) * tempS)
G = tempG * (max_g - min_g)

updateS = np.copy(S)
updateG = np.copy(G)
# print(S, "S")
# print(G, "G")
'''初始化学生潜力矩阵'''
# theta = np.linspace(min_theta, max_theta, stuNum)
theta = stats.norm.rvs(size=theta.shape, loc=mu_theta, scale=sig_theta)
# theta = mu_theta + sig_theta * np.random.random(theta.shape)
updateTheta = np.copy(theta)
# print(theta[1])

'''初始化难度矩阵（知识点k对学生i的难度）'''

# B = mu_b + sig_b * np.random.random(B.shape)
B = stats.norm.rvs(size=B.shape, loc=mu_b, scale=sig_b)
updateB = np.copy(B)
# print(B[1])
'''初始化区分度矩阵：知识点k对学生i的区分度
   所以要得到一般意义上符合对数正态分布的随机变量X（即，logX服从n(mu,sigma^2)），
   需要令lognorm中的参数s=sigma,loc=0,scale=exp(mu)。'''
A = stats.lognorm.rvs(s=sig_a, loc=0, scale=np.exp(mu_a), size=A.shape)
updateA = np.copy(A)
# print(A[1])
# print(stats.lognorm.pdf(x=A[:, 0], loc=0, scale=np.exp(0), s=1))
'''初始化主观题的方差矩阵'''

# 要求一个形态参数a。注意到β的设置可以通过设置scale关键字为1/β进行，文献上
# 的第二个参数应该是β，一般设定β=1/λ。
variance = 1 / stats.gamma.rvs(a=4, size=1, scale=1 / 6)
updateV = np.copy(variance)

'''2020.7.24'''
'''初始化标记矩阵'''
percent = 0.2  # 测试集所占百分比
stuIndex = np.arange(stuNum)
'''2020.7.25
   测试学生数 * 测试题目数 = 学生数 * 题目数 * percent
   测试学生数 = 学生数 * 题目数 * percent / 测试题目数
   kfoldNum = 学生数 / 测试学生数 = 测试题目数 / （题目数 * percent）'''
testQueNum = 8
kfoldNum = int(testQueNum / (questionNum * percent))
kfold = KFold(n_splits=kfoldNum, shuffle=True)  # n_splits表示划分为几块
index = kfold.split(X=stuIndex)  # 返回分类后的数据集的索引

'''这里的kfold将【0-535】536个数字分为了5种不同组合形式的训练集+测试集，
   train_index里的数字就是作为训练集的学生的‘编号‘
   test_index里的数字就是作为测试集的学生的‘编号‘ 
   因为有5种不同的，所以要用一个For循环一次一次取出来进行测试'''

for train_index, test_index in index:
    indicator = np.ones([stuNum, questionNum])  # 训练数据标注，1表示训练，0表示测试
    testQuestionIndex = np.random.randint(0, questionNum - 1, size=testQueNum)  # 测试题号
    while len(set(testQuestionIndex)) != len(testQuestionIndex):  # 如果取随机数的时候出现了重复数，重新取
        testQuestionIndex = np.random.randint(0, questionNum - 1, size=testQueNum)
    print("测试题号：", testQuestionIndex)
    for i in test_index:
        indicator[i][testQuestionIndex] = 0  # 测试题号的X%的学生成绩作为测试集
    '''开始迭代'''
    for w in range(echo):
        '''计算学生对知识点的认知状态'''
        alpha = 1 / (1 + np.exp(-1.7 * A * (theta - B)))
        # for i in range(stuNum):
        #     for j in range(knowledgePoint):
        #         alpha[i][j] = 1 / (1 + np.exp(-1.7 * A[i][j] * (theta[i] - B[i][j])))
        # print(np.all(alpha233 == alpha))
        # print(Q)
        '''计算学生对每道题的认知状态'''
        for i in range(stuNum):
            for j in range(questionNum):
                temp = []  # 将每道题考察的知识点的qjk加进来取最大/最小作为学生的对该题目的认知状态
                for k in range(knowledgePoint):
                    if q[j][k] == 1:
                        temp.append(alpha[i][k])
                # print(temp)
                if desc[j] == 'Obj':  # 客观题取最小值
                    N[i][j] = min(temp)
                elif desc[j] == 'Sub':  # 主观题取最大值
                    N[i][j] = max(temp)
        # print(N)

        '''开始更新a,b'''
        '''In the standard form, the distribution is uniform on [0, 1]. 
        Using the parameters loc and scale, one obtains the uniform distribution on 
        [loc, loc + scale].'''
        '''更新a，b的过程如下：每一个知识点作为一次更新，更新一个知识点的a，b，然后检查每个学生对这个a，b的拟合度
           符合约束就更新，不符合就保留原来的。'''
        # updateA = np.copy(A + 0.3 * np.random.random(A.shape))
        # updateB = np.copy(B + 0.3 * np.random.random(B.shape))
        updateA = stats.uniform.rvs(size=A.shape, loc=A - 0.1 * np.random.random(A.shape),
                                    scale=0.2 * np.random.random(A.shape))
        updateB = stats.uniform.rvs(size=B.shape, loc=B - 0.1 * np.random.random(B.shape),
                                    scale=0.2 * np.random.random(B.shape))
        # updateB = stats.norm.rvs(size=B.shape, loc=B, scale=sig_b)
        # updateA = stats.norm.rvs(size=A.shape, loc=A, scale=sig_a)
        # updateB = stats.norm.rvs(size=B.shape, loc=mu_b, scale=sig_b)
        # updateA = stats.lognorm.rvs(s=sig_a, loc=0, scale=np.exp(mu_a), size=A.shape)
        countAB = 0
        for z in range(knowledgePoint):
            # updateA[i][j] = np.random.uniform(low=A[i][j] - delta, high=A[i][j] + delta, size=1)
            # updateB[i][j] = np.random.uniform(low=B[i][j] - delta, high=B[i][j] + delta, size=1)
            '''2020.7.10
            这里给numpy矩阵赋值时如果用'='就类似c++中的&，给变量取了一个别名，因此修改tempA也会修改A，导致各种问题。
            应改为np.copy()'''
            # tempA = A
            # tempB = B
            tempA = np.copy(A)
            tempB = np.copy(B)
            tempA[:, z] = np.copy(updateA[:, z])  # 更新a，b
            tempB[:, z] = np.copy(updateB[:, z])

            '''记录新的学生对知识点的认知状态'''
            updateAlpha = 1 / (1 + np.exp(-1.7 * tempA * (theta - tempB)))
            '''记录新的学生对每道题的认知状态'''
            for i in range(stuNum):
                for j in range(questionNum):
                    temp = []  # 将每道题考察的知识点的qjk加进来取最大/最小作为学生的对该题目的认知状态
                    for k in range(knowledgePoint):
                        if q[j][k] == 1:
                            temp.append(updateAlpha[i][k])
                    # print(temp)
                    if desc[j] == 'Obj':  # 客观题取最小值
                        updateN[i][j] = min(temp)
                    elif desc[j] == 'Sub':  # 主观题取最大值
                        updateN[i][j] = max(temp)
            # print(N, 'n')
            '''文章中的似然函数是连乘，那我们求其对数似然函数变为累加，计算方便(a,b)'''
            L = fuzzyGetlog.getLog(trainscore, q, desc, N, S, G, variance)
            updateL = fuzzyGetlog.getLog(trainscore, q, desc, updateN, S, G, variance)
            for i in test_index:  # 除去测试集数据影响
                L[i][testQuestionIndex] = 0
                updateL[i][testQuestionIndex] = 0

            logP0 = np.sum(L, axis=1) + np.log(stats.norm.pdf(x=B[:, z], loc=mu_b, scale=sig_b)) + np.log(
                stats.lognorm.pdf(x=A[:, z], loc=0, scale=np.exp(mu_a), s=sig_a))  # axis=1行相加 0为列相加
            # print(np.sum(L, axis=1))
            # print(stats.norm.pdf(x=B[:, z], loc=mu_b, scale=sig_b))
            # print(stats.lognorm.pdf(x=A[:, z], s=sig_a, loc=0, scale=np.exp(mu_a)))
            logP1 = np.sum(updateL, axis=1) + np.log(stats.norm.pdf(x=tempB[:, z], loc=mu_b, scale=sig_b)) + np.log(
                stats.lognorm.pdf(x=tempA[:, z], s=sig_a, loc=0, scale=np.exp(mu_a)))  # axis=1行相加 0为列相加
            transferP = np.exp(logP1 - logP0)
            # print(transferP)
            mask = np.random.random(1)
            for i in range(stuNum):
                if transferP[i] >= 1:
                    transferP[i] = 1
                if transferP[i] >= mask:
                    countAB += 1
                    A[i, z] = updateA[i, z]  # 更新a，b
                    B[i, z] = updateB[i, z]
        print(w, countAB / knowledgePoint, "A，B转移学生的平均数")

        '''更新a，b后需要更新alpha和N矩阵'''
        alpha = 1 / (1 + np.exp(-1.7 * A * (theta - B)))
        for i in range(stuNum):
            for j in range(questionNum):
                temp = []  # 将每道题考察的知识点的qjk加进来取最大/最小作为学生的对该题目的认知状态
                for k in range(knowledgePoint):
                    if q[j][k] == 1:
                        temp.append(alpha[i][k])
                # print(temp)
                if desc[j] == 'Obj':  # 客观题取最小值
                    N[i][j] = min(temp)
                elif desc[j] == 'Sub':  # 主观题取最大值
                    N[i][j] = max(temp)

        '''开始更新theta'''
        '''遇到问题：不收敛
        猜想原因，学习率过低导致t和t-1时刻的对数似然函数+对数概率差距过小，使得转移概率每次都接近1导致无法收敛'''
        # updateTheta = theta + 0.1 * np.random.random(theta.shape)
        deltaTheta = np.random.random(theta.shape)
        updateTheta = stats.uniform.rvs(size=theta.shape, loc=theta - 0.1 * deltaTheta, scale=0.2 * deltaTheta)
        # updateTheta = stats.norm.rvs(size=theta.shape, loc=theta, scale=sig_theta)
        countTheta = 0
        '''记录新的学生对知识点的认知状态'''
        updateAlpha = 1 / (1 + np.exp(-1.7 * A * (updateTheta - B)))
        '''记录新的学生对每道题的认知状态'''
        for i in range(stuNum):
            for j in range(questionNum):
                temp = []  # 将每道题考察的知识点的qjk加进来取最大/最小作为学生的对该题目的认知状态
                for k in range(knowledgePoint):
                    if q[j][k] == 1:
                        temp.append(updateAlpha[i][k])
                # print(temp)
                if desc[j] == 'Obj':  # 客观题取最小值
                    updateN[i][j] = min(temp)
                elif desc[j] == 'Sub':  # 主观题取最大值
                    updateN[i][j] = max(temp)

        '''文章中的似然函数是连乘，那我们求其对数似然函数变为累加，计算方便(theta)'''
        L = fuzzyGetlog.getLog(trainscore, q, desc, N, S, G, variance)
        updateL = fuzzyGetlog.getLog(trainscore, q, desc, updateN, S, G, variance)
        for i in test_index:  # 除去测试集数据影响
            L[i][testQuestionIndex] = 0
            updateL[i][testQuestionIndex] = 0
        logP0 = np.sum(L, axis=1) + np.log(stats.norm.pdf(x=theta, loc=mu_theta, scale=sig_theta))  # 1为行相加 0为列相加
        logP1 = np.sum(updateL, axis=1) + np.log(stats.norm.pdf(x=updateTheta, loc=mu_theta, scale=sig_theta))
        transferP = np.exp(logP1 - logP0)
        # print(transferP, "theta 转移概率矩阵")
        mask = np.random.random(1)
        for i in range(stuNum):
            if transferP[i] >= 1:
                transferP[i] = 1
            if transferP[i] >= mask:  # 该theta值适合学生i
                # print("学生", i, "的潜力转移了")
                countTheta += 1
                theta[i] = updateTheta[i]  # 更新学生潜力
        print(w, countTheta, "名学生潜力转移了")

        '''更新theta后需要更新alpha和N矩阵'''
        alpha = 1 / (1 + np.exp(-1.7 * A * (theta - B)))
        for i in range(stuNum):
            for j in range(questionNum):
                temp = []  # 将每道题考察的知识点的qjk加进来取最大/最小作为学生的对该题目的认知状态
                for k in range(knowledgePoint):
                    if q[j][k] == 1:
                        temp.append(alpha[i][k])
                # print(temp)
                if desc[j] == 'Obj':  # 客观题取最小值
                    N[i][j] = min(temp)
                elif desc[j] == 'Sub':  # 主观题取最大值
                    N[i][j] = max(temp)

        '''更新s，g。因为s，g只影响似然函数的值，知识点认知状态，题目的认知状态与其无关，直接用alpha和N矩阵即可
           更新方法（假想）：一道题一道题的改变其s，g，然后根据所有学生的似然函数值计算转移概率，之后进行转移or保留'''
        # updateS = abs(S + 0.2 * np.random.random(S.shape) - 0.1)
        # updateG = abs(G + 0.2 * np.random.random(G.shape) - 0.1)
        updateS = stats.uniform.rvs(size=S.shape, loc=S - 0.1 * np.random.random(S.shape),
                                    scale=0.2 * np.random.random(S.shape))
        updateG = stats.uniform.rvs(size=G.shape, loc=G - 0.1 * np.random.random(S.shape),
                                    scale=0.2 * np.random.random(G.shape))
        # for i in range(questionNum):
        #     if updateS[i] >= 1:
        #         updateS[i] = 0.6
        #     if updateG[i] >= 1:
        #         updateG[i] = 0.6
        countSG = 0
        '''文章中的似然函数是连乘，那我们求其对数似然函数变为累加，计算方便(s,g)'''
        L = fuzzyGetlog.getLog(trainscore, q, desc, N, S, G, variance)
        updateL = fuzzyGetlog.getLog(trainscore, q, desc, N, updateS, updateG, variance)
        for i in test_index:  # 除去测试集数据影响
            L[i][testQuestionIndex] = 0
            updateL[i][testQuestionIndex] = 0

        logP0 = np.sum(L, axis=0) + np.log(stats.beta.pdf(x=S / (max_s - min_s), a=1, b=2)) + np.log(
            stats.beta.pdf(x=G / (max_g - min_g), a=1, b=2))  # axis=1行相加 0为列相加
        # logP0 = np.sum(L, axis=0) + np.log(
        #     stats.beta.pdf(x=(1 - S - min_s) / (max_s - min_s), a=2, b=1) / (max_s - min_s)) + np.log(
        #     stats.beta.pdf(x=G / (max_g - min_g), a=1, b=2) / (max_g - min_g))  # axis=1行相加 0为列相加
        logP1 = np.sum(updateL, axis=0) + np.log(stats.beta.pdf(x=updateS / (max_s - min_s), a=1, b=2)) + np.log(
            stats.beta.pdf(x=updateG / (max_g - min_g), a=1, b=2))
        # logP1 = np.sum(L, axis=0) + np.log(
        #     stats.beta.pdf(x=(1 - updateS - min_s) / (max_s - min_s), a=2, b=1) / (max_s - min_s)) + np.log(
        #     stats.beta.pdf(x=updateG / (max_g - min_g), a=1, b=2) / (max_g - min_g))  # axis=1行相加 0为列相加

        transferP = np.exp(logP1 - logP0)
        # print(transferP)
        mask = np.random.random(1)
        for i in range(questionNum):
            if transferP[i] >= 1:
                transferP[i] = 1
            if transferP[i] >= mask:
                countSG += 1
                S[i] = updateS[i]
                G[i] = updateG[i]
        print(w, countSG, "道题转移了")
        '''2020.7.7'''
        predictscore = np.copy((1 - S) * N + G * (1 - N))
        for i in range(len(predictscore)):
            for j in range(len(predictscore[i])):
                if desc[j] == 'Obj':
                    if predictscore[i][j] > 0.5:
                        predictscore[i][j] = 1
                    else:
                        predictscore[i][j] = 0
        rmse = (score - predictscore) * (score - predictscore)
        rmse2 = np.sqrt(np.sum(np.copy(rmse)) / (stuNum * questionNum))
        rmse = np.sqrt(np.sum(rmse, axis=0) / stuNum)
        print("echo:", w, "RMSE:", rmse)
        print("SumRMSE:", rmse2)

        '''2020.6.13'''
        '''更新方差'''
        updateV = variance - 0.01 + 0.02 * np.random.random(1)

        '''计算对数似然函数'''
        # variance = 1 / stats.gamma.rvs(a=4, size=1, scale=1 / 6)
        P = np.log(stats.gamma.pdf(x=1 / (variance + 1e-9), a=4, scale=1 / 6))
        updateP = np.log(stats.gamma.pdf(x=1 / (updateV + 1e-9), a=4, scale=1 / 6))
        '''2020.7.24'''
        L = fuzzyGetlog.getLog(trainscore, q, desc, N, S, G, variance)
        updateL = fuzzyGetlog.getLog(trainscore, q, desc, N, S, G, updateV)
        for i in test_index:  # 除去测试集数据影响
            L[i][testQuestionIndex] = 0
            updateL[i][testQuestionIndex] = 0
        for i in range(len(desc)):
            if desc[i] == 'Obj':  # 去除客观题
                L[:][i] = 0
                updateL[:][i] = 0
        logP0 = np.sum(L) + P
        logP1 = np.sum(updateL) + updateP
        transferP = np.exp(logP1 - logP0)
        if transferP >= np.random.random(1):
            variance = updateV
        if w > burnin:
            ea += np.copy(A)
            eb += np.copy(B)
            etheta += np.copy(theta)
            ealpha += np.copy(alpha)
            es += np.copy(S)
            eg += np.copy(G)
            evariance += variance
    print(theta)
    print(1 - S)
    print(G)
    ea = ea / (echo - burnin)
    eb = eb / (echo - burnin)
    etheta = etheta / (echo - burnin)
    ealpha = ealpha / (echo - burnin)
    es = es / (echo - burnin)
    eg = eg / (echo - burnin)
    evariance = evariance / (echo - burnin)
    for i in range(stuNum):
        for j in range(questionNum):
            temp = []  # 将每道题考察的知识点的qjk加进来取最大/最小作为学生的对该题目的认知状态
            for k in range(knowledgePoint):
                if q[j][k] == 1:
                    temp.append(ealpha[i][k])
            # print(temp)
            if desc[j] == 'Obj':  # 客观题取最小值
                N[i][j] = min(temp)
            elif desc[j] == 'Sub':  # 主观题取最大值
                N[i][j] = max(temp)
    es = es / (echo - burnin)
    eg = eg / (echo - burnin)
    predictscore = (1 - es) * N + eg * (1 - N)
    # predictscore = (1 - S) * N + G * (1 - N)
    rmse = (score - predictscore) * (score - predictscore)
    rmse = np.sqrt(np.sum(rmse, axis=0) / stuNum)
    print("LastRMSE:", rmse)
    np.savetxt('FuzzyA.txt', ea, fmt='%0.2f')
    np.savetxt('FuzzyB.txt', eb, fmt='%0.2f')
    np.savetxt('FuzzyS.txt', es, fmt='%0.2f')
    np.savetxt('FuzzyG.txt', eg, fmt='%0.2f')
    np.savetxt('FuzzyTheta.txt', etheta, fmt='%0.2f')
    np.savetxt('FuzzyAlpha.txt', ealpha, fmt='%0.2f')
    np.savetxt('FuzzyN.txt', N, fmt='%0.2f')
    np.savetxt('FuzzyX.txt', predictscore, fmt='%0.2f')
    np.savetxt('FuzzyRMSE.txt', rmse, fmt='%0.3f')
    # np.savetxt('FuzzyB.txt', B, fmt='%0.2f')
    # np.savetxt('FuzzyN.txt', alpha, fmt='%0.2f')
    break
