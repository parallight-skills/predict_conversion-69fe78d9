"""SKILL #1 —— 特征工程(「如何处理数据」)。
把原始 6 个特征变换成「更可学」的特征。

这道题的转化信号几乎全藏在**特征的交互**(x_i·x_j)和**非线性**(x_i²)里,
纯线性模型在原始 6 列上抓不到。实验证实「平方+两两交互+立方」是甜点
(再加三三相乘会维度爆炸 → 过拟合):33 维,把非线性信号变线性可分后,
最简单的 logistic 反而泛化最好。
约束:只看特征本身做变换,绝不读 ml/_truth/。"""
from itertools import combinations


def engineer(X):
    out = []
    for row in X:
        feats = list(row)                                   # 原始 6 列
        feats += [v * v for v in row]                       # 平方 6
        feats += [a * b for a, b in combinations(row, 2)]   # 两两交互 15
        feats += [v ** 3 for v in row]                      # 立方 6
        out.append(feats)
    return out
