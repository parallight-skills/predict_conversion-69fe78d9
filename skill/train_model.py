"""SKILL #2 —— 模型与训练(「用什么模型 / 怎么训练」)。
在(已被 feature_engineer 处理过的)训练数据上拟合一个模型,给训练集和测试集都输出概率。

特征工程已把非线性信号变线性可分 → 最简单的 logistic 反而是最优(在小数据上泛化最好,
不像 RF/GBDT 容易把训练样本背下来)。标准化后训练,系数绝对值还可当特征重要性看。"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_predict(train_X, train_y, test_X):
    Xtr = np.asarray(train_X, float)
    Xte = np.asarray(test_X, float)
    y = np.asarray(train_y, int)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    clf = LogisticRegression(max_iter=5000).fit(Xtr_s, y)
    return {
        "train": [float(p[1]) for p in clf.predict_proba(Xtr_s)],
        "test": [float(p[1]) for p in clf.predict_proba(Xte_s)],
    }
