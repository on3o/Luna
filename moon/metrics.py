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


def accuracy_score(y_true, y_predict):
    """
    评价预测准确率
    :param y_true: 正确的目标值
    :param y_predict: 预测的目标值
    :return: 0--1的值
    """
    assert y_true.shape[0] == y_predict.shape[0], "传入的目标数必须相等"
    return sum(y_true == y_predict) / len(y_true)
