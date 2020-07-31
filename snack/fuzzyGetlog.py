import numpy as np
from scipy import stats


def getLog(score, q, N, s, g, sigma, subqueIndex, objqueIndex):
    stuNum = len(score)
    questionNum = len(score[0])
    knowledgePoint = len(q[0])
    result = np.zeros([questionNum, stuNum])
    x = (1 - s) * N + g * (1 - N)  # *即点乘
    # print("x", x)
    result[objqueIndex] = (np.transpose(x)[objqueIndex] ** np.transpose(score)[objqueIndex]) * (
            (1 - np.transpose(x)[objqueIndex]) ** (1 - np.transpose(score)[objqueIndex]))
    if len(subqueIndex > 0):  # 存在主观题
        result[subqueIndex] = stats.norm.pdf(np.transpose(score)[subqueIndex], loc=np.transpose(x)[subqueIndex],
                                             scale=sigma)
    result = np.log(result)
    result[np.isnan(result)] = np.log(0)
    return np.transpose(result)


# def getLog2(score, q, desc, N, s, g, sigma):
#     stuNum = len(score)
#     questionNum = len(score[0])
#     knowledgePoint = len(q[0])
#     result = np.zeros([stuNum, questionNum])
#     x = np.copy((1 - s) * N + g * (1 - N))  # *即点乘
#     # print("x", x)
#     for i in range(stuNum):
#         for j in range(questionNum):
#             if desc[j] == 'Obj':
#                 if ((x[i][j] == 0) & (score[i][j] == 0)) | ((x[i][j] == 0) & (score[i][j] == 0)):
#                     result[i][j] = np.log(1)
#                 else:
#                     result[i][j] = (np.log(x[i][j]) * score[i][j]) + (np.log(1 - x[i][j]) * (1 - score[i][j]))
#                     if np.isnan(result[i][j]):
#                         result[i][j] = np.log(0)
#             elif desc[j] == 'Sub':
#                 result[i][j] = stats.norm.pdf(score[i][j], loc=x[i][j], scale=sigma)
#     return result
