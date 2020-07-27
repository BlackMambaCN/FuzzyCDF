import numpy as np
from scipy import stats


def getLog(score, q,  N, s, g, sigma, subqueIndex, objqueIndex):
    stuNum = len(score)
    questionNum = len(score[0])
    knowledgePoint = len(q[0])
    result = np.zeros([questionNum, stuNum])
    x = np.copy((1 - s) * N + g * (1 - N))  # *即点乘
    # print("x", x)
    result[objqueIndex] = (np.transpose(x)[objqueIndex] ** np.transpose(score)[objqueIndex]) * (
            (1 - np.transpose(x)[objqueIndex]) ** (1 - np.transpose(score)[objqueIndex]))
    if len(subqueIndex > 0):  # 存在主观题
        result[subqueIndex] = stats.norm.pdf(np.transpose(score)[subqueIndex], loc=np.transpose(x)[subqueIndex],
                                             scale=sigma)
    result = np.log(result)
    result[np.isnan(result)] = np.log(0)
    return np.transpose(result)
