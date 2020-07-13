# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 01:56:16 2020

@author: David Micouin--Jorda

"""

import numpy as np
p = np.loadtxt("valPlateau.txt")

for i in range(len(p)):
    if(p[i] == -12):
        p[i] = -6

np.savetxt("valPlateau.txt", p)