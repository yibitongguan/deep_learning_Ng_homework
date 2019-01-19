# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 09:40:00 2019

@author: Dcm
"""
"""Vectorization"""
"""2.1 Implement the L1 and L2 loss functions"""
import numpy as np
def L1(yhat, y):
    loss = np.sum(np.abs(y - yhat))
    return loss

def L2(yhat, y):
    loss = np.sum(np.power((y - yhat), 2))
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))
print("L2 = " + str(L2(yhat,y)))
