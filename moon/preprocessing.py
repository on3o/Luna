# -*- coding: utf-8 -*-#
"""
@File    :   preprocessing.py    
@Contact :   sheng_jun@yeah.net
@Author  :   Ace
@Description: 
@Modify Time      @Version    @Desciption
------------      --------    -----------
2021-07-29 0:48    1.0         None
"""
import numpy as np


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """
        数据拟合
        Args:
            X: 需要拟合的数据
        Returns:
            将self.mean_和self.scale_赋值后返回
        """
        assert X.ndim == 2, "数据必须是二维的矩阵"
        self.mean_ = np.array([np.mean(X[:, i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:, i]) for i in range(X.shape[1])])
        return self

    def transform(self, X):
        """
        数据转换
        :param X:
        :return:
        """
        assert X.ndim == 2, "数据必须是二维矩阵"
        assert self.mean_ is not None and self.scale_ is not None, "必须先进行数据拟合之后才能完成transform"
        assert X.shape[1] == len(self.mean_), "特征维度必须与训练的特征维度相等"
        resultX = np.empty(shape=X.shape, dtype=np.float)
        for col in range(X.shape[1]):
            resultX[:, col] = (X[:, col] - self.mean_[col] / self.scale_[col])
        return resultX
