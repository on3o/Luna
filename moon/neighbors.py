# -*- coding: utf-8 -*-#
"""
@File    :   neighbors.py    
@Contact :   sheng_jun@yeah.net
@Author  :   Ace
@Description: 
@Modify Time      @Version    @Desciption
------------      --------    -----------
2021-07-25 0:06    1.0         None
"""
import numpy as np
from collections import Counter
from math import sqrt
from .metrics import accuracy_score


class KNeighborsClassifier:
    """
    通过KNeighbors创建一个kNN类，用于分类算法
    """

    def __init__(self, n_neighbors=5):
        assert n_neighbors > 1, "n_neighbors必须是有效的"
        self.n_neighbors = n_neighbors
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """数据拟合过程，在kNN中为了何其他机器学习格式统一，加入这个方法，进行训练数据赋值"""
        assert X_train.shape[0] == y_train.shape[0], "传入的训练特征数据必须与传入的训练结果样本数相同"
        assert self.n_neighbors <= X_train.shape[0], "n_neighbors传入值不能超过训练特征样本数"
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        """对输入的X_predict进行预测，预测之前必须进行数据拟合"""
        assert self._X_train is not None and self._y_train is not None, "predict之前必须先进行fit"
        assert self._X_train.shape[1] == X_predict.shape[1], "预测特征向量的维度必须等于测试特征向量的维度"
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        assert x.shape[0] == self._X_train.shape[1], "预测值的特征维度必须等于训练集的特征维度"
        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        nearest = np.argsort(distances)
        topK_y = self._y_train[nearest[:self.n_neighbors]]
        votes = Counter(topK_y)
        votes.most_common(1)
        return votes.most_common(1)[0][0]

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)
