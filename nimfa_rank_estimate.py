# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 18:20:04 2022

@author: Richard
"""

print("This is the nimfa program")
import sys

sys.path.append("C:\\Users\\Richard\\Anaconda3\\Lib\\site-packages") #path to the nimfa package on my computer
import numpy as np

from nimfa import *
import csv


from numpy import genfromtxt
V = genfromtxt('input_matrix_sqrt_with_background.csv', delimiter=',')
print(V)
print(" ")
print(V.shape)
print(type(V))




#V = np.random.rand(40, 100)
ranks_array = [1,2,3,4,5,6,7,8,9,10]
params = V
Nmf_object = Nmf(params)
rank_estimation = Nmf_object.estimate_rank(rank_range=ranks_array)


#print(type(V))

print(len(rank_estimation))
print(type(rank_estimation[1]['residuals']))
print((rank_estimation[1]['residuals']))

#np.savetxt("TestResidualMatrix.csv", (rank_estimation[1]['residuals']), delimiter=",")
for i in ranks_array:
    print(i)
    csv_matrix =(rank_estimation[i]['residuals'])
    filename = "SqrtWithBackgroundResidualMatrix" + str(i) + ".csv"
    np.savetxt(filename, csv_matrix, delimiter=",")
    
