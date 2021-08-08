# -*- coding: utf-8 -*-#
"""
@File    :   model_selection.py    
@Contact :   sheng_jun@yeah.net
@Author  :   Ace
@Description: 
@Modify Time      @Version    @Desciption
------------      --------    -----------
2021-07-26 0:00    1.0         None
"""

import numpy as np


def train_test_split(X_raw, y_raw, test_ratio=.2, seed=None):
    """
    切分训练集和测试集
    :param X_raw: 原始特征向量
    :param y_raw: 原始结果
    :param test_ratio: 测试数据集比例
    :param seed: 随机数种子
    :return: X_train, X_test, y_train, y_test
    """
    assert X_raw.shape[0] == y_raw.shape[0], "传入的数据样本特征数量必须与目标数量一致"
    assert 0.0 <= test_ratio < 1.0, "测试数据的比例必须是有效值"
    if seed:
        np.random.seed(seed)
    shuffled_indexes = np.random.permutation(len(X_raw))
    test_size = int(len(X_raw) * test_ratio)
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]
    X_train = X_raw[train_indexes]
    y_train = y_raw[train_indexes]
    X_test = X_raw[test_indexes]
    y_test = y_raw[test_indexes]
    return X_train, X_test, y_train, y_test
