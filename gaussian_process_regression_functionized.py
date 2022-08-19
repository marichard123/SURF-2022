# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 20:50:03 2022

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
import copy
from io import StringIO
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C,RationalQuadratic, WhiteKernel, ExpSineSquared, DotProduct
print("Imports successful")
kernel = C(1.0, (1e-3, 1e3)) * RBF([5,5], (1e-2, 1e2))
#kernel = C(1.0, (1e-3, 1e3)) * RBF([5,5], (1e-2, 1e2))

#parameter_to_be_considered = "Center" #Amplitude, Center, Sigma, Fraction, FWHM, Height
parameter_to_be_considered = "Amplitude"
#peak_to_be_considered = 1
peak_to_be_considered = 3
random_subset_percentage = .1
uncertainty_matrix = np.array(["Certainty Boundary size", "X","Y"])
uncertainty_matrix_filename = "STD_distances_from_mean.csv"
scatterplot_only = False
def get_index_positions(list_of_elems, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    index_pos_list = []
    index_pos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            index_pos = list_of_elems.index(element, index_pos)
            # Add the index position in list
            index_pos_list.append(index_pos)
            index_pos += 1
        except ValueError as e:
            break
    return index_pos_list
def calculate_fit(linspace_array_length = 100,kernel = C(1.0, (1e-3, 1e3)) * RBF([5,5], (1e-2, 1e2)), parameter_to_be_considered = "Amplitude", peak_to_be_considered = 1, peak_parameter_filename = "fitted_peak_parameters26.csv", sample_coordinate_map_filename = "coordinate_sample_number_map.csv",model_random_subset= True, random_subset_percentage = .3):
    
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
        print("Modeling random subset")
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
    return Zp, uncertainty_range, X0p, X1p, X, y, parameter_to_be_considered
def calculate_fit_X_input(X, original_discrete_y,discrete_coordinates, linspace_array_length = 100,kernel = C(1.0, (1e-3, 1e3)) * RBF([5,5], (1e-2, 1e2)), parameter_to_be_considered = "Amplitude", peak_to_be_considered = 1, peak_parameter_filename = "fitted_peak_parameters26.csv", sample_coordinate_map_filename = "coordinate_sample_number_map.csv",model_random_subset= False, random_subset_percentage = .3):
    #rebuild y from input X
    X_length, X_width = X.shape
    y = np.zeros((1,X_length))
    for iii in range (X_length):
        x_array = discrete_coordinates[:,0]
        x_array = x_array.tolist()
        y_array = discrete_coordinates[:,1]
        y_array = y_array.tolist()
        x_indices = get_index_positions(x_array, X[iii,0])
        y_indices = get_index_positions(y_array, X[iii,1])
        matching_index = 0
        for iiiii in x_indices:
            if iiiii in y_indices:
                matching_index = iiiii
                break
        #print(matching_index)
        #print(original_discrete_y.shape)
        #print(x_indices)
        #print(y_indices)
        #print(matching_index)
        y[0,iii] = original_discrete_y[matching_index, 0]
        
    y = y.T
    #print(y)
    #print(y.shape)
    #raise error
    x_min =  X[:,0].min()
    x_max = X[:,0].max()
    y_min = X[:,1].min()
    y_max = X[:,1].max()

    


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
    return Zp, uncertainty_range, X0p, X1p, X, y, parameter_to_be_considered

#Calculating mean squared error- creating the original mesh to compare against
Zp, uncertainty_range, X0p, X1p, X,y,parameter_to_be_considered = calculate_fit(model_random_subset= False, kernel = kernel, parameter_to_be_considered = parameter_to_be_considered, peak_to_be_considered=peak_to_be_considered)
original_mesh = Zp
discrete_coordinates = X
original_discrete_y = y
#print(X)
#print(X.shape)

#Plotting subset surface and scatterplot
#Zp, uncertainty_range, X0p, X1p, X,y,parameter_to_be_considered = calculate_fit(model_random_subset= True,random_subset_percentage = .3)
Zp, uncertainty_range, X0p, X1p, X,y,parameter_to_be_considered = calculate_fit(model_random_subset= True,random_subset_percentage = random_subset_percentage,kernel = kernel, parameter_to_be_considered = parameter_to_be_considered, peak_to_be_considered=peak_to_be_considered)
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax = fig.add_subplot(111, projection='3d')    
if not scatterplot_only:        
    surf = ax.plot_surface(X0p, X1p, Zp, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False)
subset_mesh = Zp
ax.scatter(X[:,0],X[:,1], y)
ax.set_xlabel('X coordinates (mm)')
ax.set_ylabel('Y coordinates (mm)')
ax.set_zlabel(parameter_to_be_considered)

#Plotting subset uncertainty range
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax = fig.add_subplot(111, projection='3d')    
if not scatterplot_only:        
    surf = ax.plot_surface(X0p, X1p, uncertainty_range, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False)
subset_mesh = Zp
ax.set_xlabel('X coordinates (mm)')
ax.set_ylabel('Y coordinates (mm)')
ax.set_zlabel("Distance of one standard deviation from mean")
#Saving uncertainties to text file
for iy, ix in np.ndindex(X0p.shape):
    #insert if statement here to only save a certain subset of uncertainties or coordinates
    new_row = [uncertainty_range[iy, ix], X0p[iy, ix], X1p[iy, ix]]
    uncertainty_matrix = np.vstack([uncertainty_matrix, new_row])

np.savetxt(uncertainty_matrix_filename, uncertainty_matrix, delimiter=",", fmt="%s")
print("Uncertainties file saved as " + uncertainty_matrix_filename)

#The mean squared error will be rewritten after the subset updates and such
#mse = (np.square(original_mesh - subset_mesh)).mean(axis=None)
#print("Mean squared error: " + str(mse))
#print("Mean of original mesh: " + str(np.average(original_mesh)))

discrete_uncertainties = [] #order will follow that of the coordinates array
for iiii in discrete_coordinates:
    discrete_x, discrete_y = iiii
    x_array = uncertainty_matrix[1:-1,1].astype('float')
    closest_continuous_x = x_array[np.argmin(np.abs(x_array-discrete_x))]
    y_array = uncertainty_matrix[1:-1,2].astype('float')
    closest_continuous_y = y_array[np.argmin(np.abs(y_array-discrete_y))]
    
    x_array_list = x_array.tolist()
    y_array_list = y_array.tolist()
    closest_continuous_x_indices = get_index_positions(x_array_list, closest_continuous_x)
    closest_continuous_y_indices = get_index_positions(y_array_list, closest_continuous_y)
    matching_index = 0
    for iiiii in closest_continuous_x_indices:
        if iiiii in closest_continuous_y_indices:
            matching_index = iiiii
            break
    uncertainty_array = uncertainty_matrix[1:-1,0].astype('float')
    discrete_uncertainties.append(uncertainty_array[matching_index])

discrete_uncertainties_sorted = copy.deepcopy(discrete_uncertainties)
discrete_uncertainties_sorted.sort(reverse = True)


original_subset = copy.deepcopy(X)
iteration = 1
while iteration < 5:
    X = copy.deepcopy(original_subset)
    original_length = X[:,0].size
    for Looping in range(10):
        coordinates_already_present = False
        for iiii in discrete_uncertainties_sorted:
            coordinate_index = discrete_uncertainties.index(iiii)
            discrete_x = discrete_coordinates[coordinate_index,0]
            discrete_y = discrete_coordinates[coordinate_index,1]
            for iiiii in range(X[:,0].size):
                if X[iiiii,0] == discrete_x and X[iiiii,1] == discrete_y:
                    coordinates_already_present = True
                    
            if not coordinates_already_present:
                if X[:,0].size - original_length > ((iteration*10)-1):
                    break
            
                new_row = [discrete_x, discrete_y]
                X = np.vstack([X, new_row])

    print(X.shape)
    Zp, uncertainty_range, X0p, X1p, X, y, parameter_to_be_considered = calculate_fit_X_input(X =X, original_discrete_y =original_discrete_y,discrete_coordinates =discrete_coordinates, kernel = kernel, parameter_to_be_considered = parameter_to_be_considered, peak_to_be_considered=peak_to_be_considered)
    subset_mesh = Zp
    mse = (np.square(original_mesh - subset_mesh)).mean(axis=None)
    print("Mean squared error: " + str(mse))
    print("Mean of original mesh: " + str(np.average(original_mesh)))
    
    
    
    error_mesh = original_mesh - subset_mesh
    #print(error_mesh)
    #plotting error graph
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax = fig.add_subplot(111, projection='3d')    
    if not scatterplot_only:        
        surf = ax.plot_surface(X0p, X1p, np.abs(error_mesh), rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False)
    subset_mesh = Zp
    ax.set_xlabel('X coordinates (mm)')
    ax.set_ylabel('Y coordinates (mm)')
    ax.set_zlabel("Error (absolute difference between original and prediction)")
    ax.set_zlim(0, 2.5*np.average(original_mesh))
    #ax.set_zlim(0,3)
    ax.set_title("Iteration " + str(iteration) + ", using " + str(X.size/2) + " points, MSE=" + str(mse)+ ", mean = " +str(np.average(original_mesh)))
    iteration+=1

