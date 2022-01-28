#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 06:56:56 2022

@author: denizaycan
"""

import numpy as np
import unittest
from regfunction import regfunction

class testregfunction(unittest.TestCase):
   

    def test_insufficiency(self):
        with self.assertRaises(AttributeError):
            regfunction(3, 4)

    def test_inconsistency(self):
        x = np.random.randint(10, size=(10, 3))
        y = np.random.randint(10, size=(15, 1))
        with self.assertRaises(ValueError):
            regfunction(x, y)

    def test_string(self):
        y = np.array([1,2,3,4,5,6,7,8,9,'word'])
        x = np.random.randint(10, size=(10, 2))
        with self.assertRaises(ValueError):
             regfunction(x, y)

            
if  __name__ == '__main__':
    unittest.main()
    

    #https://docs.python.org/3/library/unittest.html
    