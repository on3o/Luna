# -*- coding: utf-8 -*-#
"""
@File    :   metrics.py    
@Contact :   sheng_jun@yeah.net
@Author  :   Ace
@Description: 
@Modify Time      @Version    @Desciption
------------      --------    -----------
2021-07-26 1:23    1.0         None
"""

import numpy as np
from math import sqrt


def accuracy_score(y_true, y_predict):
    """
    评价分类预测的准确率
    :param y_true: 正确的目标值
    :param y_predict: 预测的目标值
    :return: 0--1的值
    """
    assert y_true.shape[0] == y_predict.shape[0], "传入的目标数必须相等"
    return sum(y_true == y_predict) / len(y_true)


def mean_squared_error(y_true, y_predict):
    """
    均方误差
    :param y_true:正确的目标值
    :param y_predict: 预测得到的目标值
    :return:
    """
    assert len(y_true) == len(y_predict), "预测值的数据量必须等于测试值得数据量"
    return np.sum((y_true - y_predict) ** 2) / len(y_true)


def root_mean_squared_error(y_true, y_predict):
    """
    均方根误差
    :param y_true:
    :param y_predict:
    :return:
    """
    assert len(y_true) == len(y_predict), "预测值的数据量必须等于测试值得数据量"
    return sqrt(mean_squared_error(y_true, y_predict))


def mean_absolute_error(y_true, y_predict):
    """
    平均绝对误差
    :param y_true:
    :param y_predict:
    :return:
    """
    assert len(y_true) == len(y_predict), "预测值的数据量必须等于测试值得数据量"
    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)


def r2_score(y_true, y_predict):
    """
    R^2
    :param y_true:
    :param y_predict:
    :return:
    """
    return 1 - (mean_squared_error(y_true, y_predict) / np.var(y_true))
