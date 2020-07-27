import numpy as np


def getdesc(filepath):
    file = open(filepath)
    subqusNum = 0  # 主观题数目
    objqusNum = 0  # 客观题数目
    subqueIndex = []
    objqueIndex = []
    fileStr = file.readlines()  # string类型的list
    desc = []
    for i in fileStr:
        i = i[:-1]  # 去掉换行符
        i = i.strip()  # 去掉字符串两端的空白字符
        i = i.split("\t")  # 以\t为分隔符分隔每个字符串
        for j in i:
            desc.append(j)
        # print(i)
    file.close()
    desc = np.array(desc)
    if desc[len(desc) - 1] == '':
        desc = np.delete(desc, len(desc) - 1, axis=0)
    desc = desc.reshape([-1, 3])
    desc = np.delete(desc, 0, axis=0)
    desc = np.delete(desc, 0, axis=1)
    desc = np.delete(desc, 1, axis=1)
    desc = desc.reshape(-1)
    for i in range(len(desc)):
        if desc[i] == 'Obj':
            objqusNum += 1
            objqueIndex.append(i)
        elif desc[i] == 'Sub':
            subqusNum += 1
            subqueIndex.append(i)
    objqueIndex = np.array(objqueIndex)
    subqueIndex = np.array(subqueIndex)
    return desc, subqueIndex, objqueIndex, subqusNum, objqusNum
