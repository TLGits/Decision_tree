# -*- coding: utf-8 -*-
from sklearn import datasets
import numpy as np
import math

class Decision_Node:
    def __init__(self, feature, value, true_branch, false_branch, entropy_Yes, entropy_No):
        """
        定义决策树类
        :param feature: 分类特征
        :param value: 分类值
        :param true_branch:第一类
        :param false_branch: 第二类
        :param entropy_Yes: 第一类的熵
        :param entropy_No: 第二类的熵
        """
        self.feature = feature
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.entropy_Yes = entropy_Yes
        self.entropy_No = entropy_No

class Leaf:
    def __init__(self, data_set):
        self.predictions = class_counts(data_set)

def class_counts(data_set):
    """
    计算数据集中的类别及其对应的数量
    :param data_set: 数据集
    :return: 字典label -> count
    """
    counts = {}  # dictionary of label -> count.
    for row in data_set:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def splitting(data_set, axis, value):
    """
    分割数据集
    :param data_set: 数据集
    :param axis: 列序号
    :param value: 列分解值(每一列值的大小)
    :return: 分割正确的数据，分割错误的数据
    """
    true_rows, false_rows = [], []
    for row in data_set:
        if row[axis] >= value:
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

def Entropy(data_set):
    """
    根据公式计算此时的熵
    :param data_set: 输入数据集
    :return: 熵
    """
    counts = class_counts(data_set)
    entropy = 0.0
    for label in counts:
        prob = counts[label] / float(len(data_set))
        entropy -= prob * math.log(prob, 2)
    return entropy

def find_best_split(data_set):
    """
    寻找最佳分类
    :param data_set: 输入数据集
    :return: 最佳信息增益，最佳特征，最佳分界值，第一类的熵，第二类的熵
    """
    best_gain = 0.0  # 最佳信息增益
    best_Feature = 0  # 最佳特征
    best_value = 0.0  # 最佳分界值
    Base_entropy = Entropy(data_set)
    entropy_Yes = 0.0  # 第一类的熵
    entropy_No = 0.0  # 第二类的熵
    n_features = len(data_set[0]) - 1
    for col in range(n_features):
        values = set([row[col] for row in data_set])  # 每一列不同值 的集合
        for val in values:
            true_rows, false_rows = splitting(data_set, col, val)  # 分割数据集
            if len(true_rows) == 0 or len(false_rows) == 0:  # 如果根据val值没有分割
                continue  # 跳出循环，尝试下一个值
            p = float(len(true_rows)) / (len(true_rows) + len(false_rows))  # 计算第一类的比例
            entropy1 = Entropy(true_rows)  # 计算第一类的熵
            entropy2 = Entropy(false_rows)  # 计算第二类的熵
            gain = Base_entropy - p * entropy1 - (1 - p) * entropy2  # 计算此时的信息增益(信息增益越大越好)
            if gain >= best_gain:  # 如果此时信息增益达到最大，更新最佳分类相关量
                best_gain, best_Feature, best_value = gain, col, val
                entropy_Yes, entropy_No = entropy1, entropy2
    return best_gain, best_Feature, best_value, entropy_Yes, entropy_No

def build_tree(data_set):
    """
    构建决策树
    :param data_set: 数据集
    :return: 决策树的结构体
    """
    gain, feature, value, entropy_Yes, entropy_No = find_best_split(data_set)
    if gain == 0:
        return Leaf(data_set)
    true_rows, false_rows = splitting(data_set, feature, value)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)
    return Decision_Node(feature, value, true_branch, false_branch, entropy_Yes, entropy_No)

def print_tree(node, spacing=""):
    """
    输出决策树
    :param node: 构建好的决策树类
    :param spacing: 空格
    :return: 输出决策树
    """
    if isinstance(node, Leaf):  # 当分到最下面的一类时，输出此时隶属于这个类别下面的样本数数量
        print(spacing + "∟Samples", node.predictions)
        return
    print(spacing + header[node.feature] + '>=' + str(node.value) + '?')
    print(spacing + '=> Yes:' + '(熵=', node.entropy_Yes, ')')
    print_tree(node.true_branch, spacing + "  ")
    print(spacing + '=> No:' + '(熵=', node.entropy_No, ')')
    print_tree(node.false_branch, spacing + "  ")

if __name__ == "__main__":
    # 加载鸢尾花数据集
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    training_data = np.c_[X, y]

    header = ["花萼长度", "花萼宽度", "花瓣长度", "花瓣宽度", "label"]  # 添加列名

    decision_tree = build_tree(training_data)  # 构建决策树
    print_tree(decision_tree)  # 输出决策树
