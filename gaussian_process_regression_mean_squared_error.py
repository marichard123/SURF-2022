# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 00:21:35 2022

@author: Richard
"""

print("This is the Gaussian process regression program")
from itertools import product
import numpy as np
import csv
from numpy import genfromtxt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from io import StringIO
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C,RationalQuadratic, WhiteKernel, ExpSineSquared, DotProduct
print("Imports successful")

#linspace_array_length = 50
linspace_array_length = 100
kernel = C(1.0, (1e-3, 1e3)) * RBF([5,5], (1e-2, 1e2))
#kernel = C()*ExpSineSquared(length_scale=24,periodicity=1)
#kernel = C()
#parameter_to_be_considered = "Center" #Amplitude, Center, Sigma, Fraction, FWHM, Height
parameter_to_be_considered = "Amplitude"
peak_to_be_considered = 1
peak_parameter_filename = "fitted_peak_parameters26.csv"
sample_coordinate_map_filename = "coordinate_sample_number_map.csv"
uncertainty_matrix = np.array(["Certainty Boundary size", "X","Y"])
uncertainty_matrix_filename = "STD_distances_from_mean.csv"


model_random_subset= True
scatterplot_only = False
random_subset_percentage = .3 #.8 means that 80% of the original data will be present in the subset



peak_parameter_matrix = genfromtxt(peak_parameter_filename, delimiter=',')
with open(peak_parameter_filename, newline='') as f:
    reader = csv.reader(f)
    peak_parameter_matrix_string = list(reader)
with open(sample_coordinate_map_filename, newline='') as f:
    reader = csv.reader(f)
    sample_coordinate_map_matrix = list(reader)

peak_parameter_matrix_sample_numbers = peak_parameter_matrix[:,0]
number_of_samples = int(peak_parameter_matrix_sample_numbers[len(peak_parameter_matrix_sample_numbers)-1] +1)
y = np.zeros((1,number_of_samples))
possible_parameters_array = ["Amplitude","Center","Sigma","Fraction","FWHM","Height"]
parameter_column = possible_parameters_array.index(parameter_to_be_considered) + 2
peak_name = "v" + str(peak_to_be_considered) + "_"
peak_parameter_matrix_length, peak_parameter_matrix_width = peak_parameter_matrix.shape

for i in range(peak_parameter_matrix_length):
    if peak_parameter_matrix_string[i][1] == peak_name:
        y[0,int(peak_parameter_matrix[i,0])] = peak_parameter_matrix[i,parameter_column]
y = y.T
#print(y)
#y values are loaded in by now

X = np.empty((number_of_samples,2),dtype = list)




for i in range(number_of_samples):
    X[i, 0] = float(sample_coordinate_map_matrix[i+1][0])
    X[i, 1] = float(sample_coordinate_map_matrix[i+1][1])
    
x_min =  X[:,0].min()
x_max = X[:,0].max()
y_min = X[:,1].min()
y_max = X[:,1].max()

#print(X.shape)
if model_random_subset:
    print("Modelling random subset")
    number_of_elements_to_delete = int(round((1-random_subset_percentage)*(y.size)))
    index_array = np.zeros((1,number_of_elements_to_delete), dtype = int)
    for iiii in range(number_of_elements_to_delete):
        #print(index_array[0,iiii])
        index = 0
        while index in index_array:
            index = int(random.randint(0, y.size-1))
        index_array[0,iiii] = index
    
    X = np.empty((number_of_samples-number_of_elements_to_delete,2),dtype = list)
    X_filling_index = 0
    for i in range(number_of_samples):
        if i not in index_array:
            #print(X_filling_index)
            X[X_filling_index, 0] = float(sample_coordinate_map_matrix[i+1][0])
            X[X_filling_index, 1] = float(sample_coordinate_map_matrix[i+1][1])
            X_filling_index+=1

    yplaceholder = np.zeros((1,number_of_samples-number_of_elements_to_delete))
    y_filling_index = 0
    for i in range(number_of_samples):
        if i not in index_array:
            yplaceholder[0,y_filling_index] = y[i,0]
            y_filling_index+=1
    yplaceholder = yplaceholder.T
    y = yplaceholder


x1 = np.linspace(x_min, x_max,linspace_array_length) #p
x2 = np.linspace(y_min, y_max,linspace_array_length) #q
x = (np.array([x1, x2])).T



#kernel = C(1.0, (1e-3, 1e3)) * RBF([5,5], (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)

gp.fit(X, y)

x1x2 = np.array(list(product(x1, x2)))
y_pred, MSE = gp.predict(x1x2, return_std=True)

X0p, X1p = x1x2[:,0].reshape(linspace_array_length,linspace_array_length), x1x2[:,1].reshape(linspace_array_length,linspace_array_length)
Zp = np.reshape(y_pred,(linspace_array_length,linspace_array_length))
uncertainty_range =  np.reshape(MSE,(linspace_array_length,linspace_array_length))

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax = fig.add_subplot(111, projection='3d')    
if not scatterplot_only:        
    surf = ax.plot_surface(X0p, X1p, Zp, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False)
    #surf = ax.plot_surface(X0p, X1p, uncertainty_range, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False)

#print(Zp.shape)
subset_mesh = Zp
#X0p, X1p = X[:,0].reshape(50,50), X[:,1].reshape(50,50)
#Zp = np.reshape(y,(50,50))
X0p, X1p = X[:,0], X[:,1]
ax.scatter(X0p,X1p, y)
ax.set_xlabel('X coordinates (mm)')
ax.set_ylabel('Y coordinates (mm)')
ax.set_zlabel(parameter_to_be_considered)


#plotting uncertainty graph
X0p, X1p = x1x2[:,0].reshape(linspace_array_length,linspace_array_length), x1x2[:,1].reshape(linspace_array_length,linspace_array_length)
Zp = np.reshape(y_pred,(linspace_array_length,linspace_array_length))
uncertainty_range =  np.reshape(MSE,(linspace_array_length,linspace_array_length))

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax = fig.add_subplot(111, projection='3d')    


surf = ax.plot_surface(X0p, X1p, uncertainty_range, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False)

for iy, ix in np.ndindex(X0p.shape):
    #insert if statement here to only save a certain subset of uncertainties or coordinates
    new_row = [uncertainty_range[iy, ix], X0p[iy, ix], X1p[iy, ix]]
    uncertainty_matrix = np.vstack([uncertainty_matrix, new_row])

np.savetxt(uncertainty_matrix_filename, uncertainty_matrix, delimiter=",", fmt="%s")
print("Uncertainties file saved as " + uncertainty_matrix_filename)
X0p, X1p = X[:,0], X[:,1]
ax.set_xlabel('X coordinates (mm)')
ax.set_ylabel('Y coordinates (mm)')
ax.set_zlabel("Distance of one standard deviation from mean")

#Proof of concept: Asigning uncertainty values to the discrete datapoints from the mesh












#Doing it again to re-calculate the original to compare to the subset
#You can mostly forget about the stuff under this line
peak_parameter_matrix = genfromtxt(peak_parameter_filename, delimiter=',')
with open(peak_parameter_filename, newline='') as f:
    reader = csv.reader(f)
    peak_parameter_matrix_string = list(reader)
with open(sample_coordinate_map_filename, newline='') as f:
    reader = csv.reader(f)
    sample_coordinate_map_matrix = list(reader)

peak_parameter_matrix_sample_numbers = peak_parameter_matrix[:,0]
number_of_samples = int(peak_parameter_matrix_sample_numbers[len(peak_parameter_matrix_sample_numbers)-1] +1)
y = np.zeros((1,number_of_samples))
possible_parameters_array = ["Amplitude","Center","Sigma","Fraction","FWHM","Height"]
parameter_column = possible_parameters_array.index(parameter_to_be_considered) + 2
peak_name = "v" + str(peak_to_be_considered) + "_"
peak_parameter_matrix_length, peak_parameter_matrix_width = peak_parameter_matrix.shape
#print(peak_parameter_matrix_length)

for i in range(peak_parameter_matrix_length):
    if peak_parameter_matrix_string[i][1] == peak_name:
        y[0,int(peak_parameter_matrix[i,0])] = peak_parameter_matrix[i,parameter_column]
y = y.T
#print(y)
#y values are loaded in by now


#print(sample_coordinate_map_matrix)
#print(type([2,4]))
X = np.empty((number_of_samples,2),dtype = list)




for i in range(number_of_samples):
    X[i, 0] = float(sample_coordinate_map_matrix[i+1][0])
    X[i, 1] = float(sample_coordinate_map_matrix[i+1][1])
    
x_min =  X[:,0].min()
x_max = X[:,0].max()
y_min = X[:,1].min()
y_max = X[:,1].max()

#print(X.shape)

x1 = np.linspace(x_min, x_max,linspace_array_length) #p
x2 = np.linspace(y_min, y_max,linspace_array_length) #q
x = (np.array([x1, x2])).T


#REMEMBER: YOU ARE CURRENTLY USING THE SAME KERNEL FOR BOTH EVALUATIONS
#kernel = C(1.0, (1e-3, 1e3)) * RBF([5,5], (1e-2, 1e2))


gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)

gp.fit(X, y)

x1x2 = np.array(list(product(x1, x2)))
y_pred, MSE = gp.predict(x1x2, return_std=True)

X0p, X1p = x1x2[:,0].reshape(linspace_array_length,linspace_array_length), x1x2[:,1].reshape(linspace_array_length,linspace_array_length)
Zp = np.reshape(y_pred,(linspace_array_length,linspace_array_length))

#fig = plt.figure(figsize=(10,8))
#ax = fig.add_subplot(111)
#ax = fig.add_subplot(111, projection='3d')    
#if not scatterplot_only:        
#    surf = ax.plot_surface(X0p, X1p, Zp, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False)

#print(Zp.shape)
original_mesh = Zp
#print(original_mesh.shape)
mse = (np.square(original_mesh - subset_mesh)).mean(axis=None)
#print(mse)
print("Mean squared error: " + str(mse))
print("Mean of original mesh: " + str(np.average(original_mesh)))
#plt.show()