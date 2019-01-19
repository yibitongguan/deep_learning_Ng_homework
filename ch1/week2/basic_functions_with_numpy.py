# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 08:16:57 2019

@author: Dcm
"""
"""1.1 - sigmoid function, np.exp()"""
import math
def basic_sigmoid(x):
    s = 1 / (1 + math.exp(-x))
    return s
#print(basic_sigmoid(3))

import numpy as np
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
#x = np.array([1, 2, 3])
#print(sigmoid(x))
"""1.2 - Sigmoid gradient"""
def sigmoid_derivative(x):
    s = 1 / (1 + np.exp(-x))
    ds = s * (1- s)
    return ds
#X = np.array([1, 2, 3])
#print("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))
"""1.3 - Reshaping arrays"""
def image2vector(image):
    v = image.reshape((image.shape[0] * image.shape[1] * image.shape[2], 1))
    return v
# =============================================================================
# image = np.array([[[ 0.67826139,  0.29380381],
#         [ 0.90714982,  0.52835647],
#         [ 0.4215251 ,  0.45017551]],
#        [[ 0.92814219,  0.96677647],
#         [ 0.85304703,  0.52351845],
#         [ 0.19981397,  0.27417313]],
#        [[ 0.60659855,  0.00533165],
#         [ 0.10820313,  0.49978937],
#         [ 0.34144279,  0.94630077]]])
# print ("image2vector(image) = " + str(image2vector(image)))
# =============================================================================
"""1.4 - Normalizing rows"""
def normalizeRows(x):
    """归一化"""
    # 计算每一行长度
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / x_norm
    return x
# ============================ =================================================
# x = np.array([
#     [0, 3, 4],
#     [1, 6, 4]])
# print("normalizeRows(x) = " + str(normalizeRows(x)))
# =============================================================================

"""1.5 - Broadcasting and the softmax function"""  
def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum
    return s
x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
print("softmax(x) = " + str(softmax(x)))
