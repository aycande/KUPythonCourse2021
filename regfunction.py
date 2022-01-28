#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 22:46:02 2022

@author: denizaycan
"""

import numpy as np

def regfunction(x, y):

    b = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    e = y - x.dot(b)
    etrp= np.transpose(e)
    
    
    row = x.shape[0]
    column = x.shape[1]
    varsq= (etrp @ e) / (row - column - 1)


    varb= np.diag(np.multiply(varsq, np.linalg.inv(x.T.dot(x))))
    errorstd= np.sqrt(varb).reshape(2,1)

    z=1.96
    cinterval=[b - z*errorstd, b + z*errorstd]

    return b, errorstd, cinterval
   