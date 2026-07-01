"""SKILL #3 —— 校准 / 预测(「如何 predict」)。
把模型输出的概率变成最终的 0/1 标签。

⚠️ 现在是固定 0.5 阈值。指挥 agent 改成**在训练集上挑一个让训练 accuracy 最高的阈值**
(threshold tuning)——类别不均衡时,这一步能再抠几分。这是价值最小、但也最便宜的一件 skill。"""


def to_labels(train_probs, train_y, test_probs):
    # 在训练集上扫一遍阈值,挑让训练 accuracy 最高的那个(类别不均衡时 0.5 不是最优切点)。
    best_thr, best_acc = 0.5, -1.0
    for thr in [i / 100 for i in range(101)]:
        acc = sum(
            1 for p, t in zip(train_probs, train_y)
            if (1 if p >= thr else 0) == int(t)
        ) / len(train_y)
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return [1 if p >= best_thr else 0 for p in test_probs]
