# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 00:39:46 2022

@author: Richard
"""
#peaks_to_ignore = [4,5,6,7,8]
#peaks_to_ignore = [5,6,7,8]
#peaks_to_ignore = [4,9]
#peaks_to_ignore = [5,6,7,8,9,10,11,12,13,14,15,16,17]
#peaks_to_ignore = [1,2,3,4,8,10,11,12,13,15,16,17]
#peaks_to_ignore = [8,10,11,12,13,15,16,17]
#peaks_to_ignore = [5,6,7,8,9,10,11,12,13,14,15,16,17]
#peaks_to_ignore = [8,10,11,12,13,15,16,17]
peaks_to_ignore = [10,11,12,13,15,16,17]
consider_amplitude = True
consider_sigma = True
consider_gamma = False
consider_fwhm = False
consider_fraction = False
center_range = 2.5
amplitude_range = 20
sigma_range = 1
gamma_range = 1
fwhm_range = 1
fraction_range = .2






background_polynomial_degree = 4

print("This is the peak fitting program, looping through all samples")
import matplotlib.pyplot as plt
import numpy as np
import math
from xrdprofile.background import BSplineModel
from lmfit.models import ExponentialModel, GaussianModel, PseudoVoigtModel, PolynomialModel
from numpy import genfromtxt
print("Imports successful")
input_matrix = genfromtxt("input_matrix_raw_data.csv", delimiter=',')
background_matrix = genfromtxt("background_of_raw_data.csv", delimiter=',')
x = genfromtxt("two_theta_values.csv", delimiter=',')
#peak_data_matrix = genfromtxt("CoCrAl_detected_peaks_trimmed_sorted.csv", delimiter=',')
peak_data_matrix = genfromtxt("CoCrAl_detected_peaks_with_Si.csv", delimiter=',')
print(peak_data_matrix)
(number_of_peaks, number_of_parameters) = peak_data_matrix.shape
(number_of_samples, two_theta_length) = np.shape(input_matrix)
fitted_peaks_matrix = np.zeros(np.shape(input_matrix))
composition_fitted_peaks_matrix = np.zeros((0,x.size))
output_data_matrix = np.array(["Sample Number", "Peak Name", "Amplitude", "Center","Sigma", "Fraction", "FWHM", "Height"])
#print(type(np.shape(input_matrix)))

for sample_number in range(number_of_samples):
    print(sample_number)

    #these two go into the loop- replace "2" with sample_number
    y = input_matrix[sample_number]
    background = background_matrix[sample_number]

    #print(number_of_samples)
    #raise error
    #for sample_number in range(177):
        #    print(background_matrix[sample_number])
    #raise error
    #Getting initial fit to background
    background_polynomial_model =PolynomialModel(degree = background_polynomial_degree,prefix='bkg_')
    background_pars = background_polynomial_model.guess(background, x=x)
    init = background_polynomial_model.eval(background_pars, x=x)
    out = background_polynomial_model.fit(background, background_pars, x=x)
    exp_mod = ExponentialModel(prefix='exp_')
    pars = exp_mod.guess(y, x=x)
    polynomial_model = PolynomialModel(degree = background_polynomial_degree,prefix = 'poly_')
    pars.update(polynomial_model.make_params())
    for i in background_polynomial_model.param_names:
        pars[('poly_' + i[len(i)-2:len(i)])].set(value=out.params[i].value, min = out.params[i].value*.999, max =out.params[i].value*1 )
        #pars[('poly_' + i[len(i)-2:len(i)])].set(value=out.params[i].value, min = out.params[i].value*.98, max =out.params[i].value*1.02)
        #pars[('poly_' + i[len(i)-2:len(i)])].set(value=out.params[i].value, min = out.params[i].value*.99, max =out.params[i].value*1.01)
        #pars[('poly_' + i[len(i)-2:len(i)])].set(value=out.params[i].value, min = out.params[i].value*.965, max =out.params[i].value*1.035 )
    mod = polynomial_model

    

    #polynomial fitting code is redunant and gets overwritten- I just keep it in here as a backup




    #spline model
    
    ncoefs = 10
    spline_polynomial_degree = 3
    xknots = np.linspace(x[0], x[-1], ncoefs)
    #xknots = np.array([60,70,80,90,100])
    #xknots = np.array([x[0],30,50,x[-1]])

    background_spline_model = BSplineModel(knots = xknots,degree = spline_polynomial_degree,prefix='bkg_spline_')
    background_pars_spline = background_spline_model.guess(background, x=x)
    init = background_spline_model.eval(background_pars_spline, x=x)
    out = background_spline_model.fit(background, background_pars_spline, x=x)
    spline_model = BSplineModel(knots = xknots,degree = spline_polynomial_degree, nan_policy ="raise", prefix='spline_')
    pars.update(spline_model.make_params())
    for i in background_spline_model.param_names:
        #print(i)
        if out.params[i].value != out.params[i].value/2 or out.params[i]==0:
            #pars[('spline_' + i[11:len(i)])].set(value=out.params[i].value, min = out.params[i].value*1, max =out.params[i].value*1.001 )
            #pars[('spline_' + i[11:len(i)])].set(value=out.params[i].value, min = out.params[i].value*.9999, max =out.params[i].value*1.0001 )
            pars[('spline_' + i[11:len(i)])].set(value=out.params[i].value, min = out.params[i].value*.98, max =out.params[i].value*1.02 )
            #pars[('spline_' + i[11:len(i)])].set(value=out.params[i].value )
        else:
            #pars[('spline_' + i[11:len(i)])].set(value=out.params[i].value )
            pars[('spline_' + i[11:len(i)])].set(value=0)

    mod = spline_model
    #polynomial_model2 = PolynomialModel(degree = background_polynomial_degree,prefix = 'poly2_')
    #pars.update(polynomial_model2.make_params())
    #for i in background_polynomial_model.param_names:
        #pars[('poly_' + i[len(i)-2:len(i)])].set(value=out.params[i].value, min = out.params[i].value*.999, max =out.params[i].value*1 )
        #pars[('poly_' + i[len(i)-2:len(i)])].set(value=out.params[i].value, min = out.params[i].value*.98, max =out.params[i].value*1.02)
   #     pars[('poly2_' + i[len(i)-2:len(i)])].set(value=out.params[i].value, min = out.params[i].value*.5, max =out.params[i].value*1.5)
    #mod = polynomial_model
    #mod = mod +polynomial_model2
    
    for i in range(1,number_of_peaks):
        if i not in peaks_to_ignore:
            #print(i)
            peak_prefix = "v" + str(i) + "_"
            voigti = PseudoVoigtModel(prefix=peak_prefix)
            pars.update(voigti.make_params())
        
            center_value = peak_data_matrix[i,1]
            if (center_value- center_range >=0):
                pars[peak_prefix +'center'].set(value=center_value,min =center_value- center_range,max =center_value+ center_range)
            else:
                pars[peak_prefix +'center'].set(value=center_value,min =0,max =center_value+ center_range)
        
        
            if consider_amplitude:
                amplitude_value = peak_data_matrix[i,2]
                if (amplitude_value- amplitude_range >=0):
                    pars[peak_prefix +'amplitude'].set(value=amplitude_value,min =amplitude_value- amplitude_range,max =amplitude_value+ amplitude_range)
                else:
                    pars[peak_prefix +'amplitude'].set(value=amplitude_value,min =0,max =amplitude_value+ amplitude_range)
            
        
        
            if consider_sigma:
                sigma_value = peak_data_matrix[i,3]
                if (sigma_value- sigma_range >=0):
                    pars[peak_prefix +'sigma'].set(value=sigma_value,min =sigma_value- sigma_range,max =sigma_value+ sigma_range)
                else:
                    pars[peak_prefix +'sigma'].set(value=sigma_value,min =0,max =sigma_value+ sigma_range)
            
        
            if consider_gamma:
                gamma_value = peak_data_matrix[i,4]
                pars[peak_prefix +'gamma'].set(value=gamma_value,min =gamma_value- gamma_range,max =gamma_value+ gamma_range)

            if consider_fwhm:
                fwhm_value = peak_data_matrix[i,5]
                if (fwhm_value- fwhm_range >=0):
                    pars[peak_prefix +'fwhm'].set(value=fwhm_value,min =fwhm_value- fwhm_range,max =fwhm_value+ fwhm_range)
                else:
                    pars[peak_prefix +'fwhm'].set(value=fwhm_value,min =0,max =fwhm_value+ fwhm_range)
            
        
            if consider_fraction:
                fraction_value = peak_data_matrix[i,6]
                if not math.isnan(fraction_value):
                    if (fraction_value- fraction_range >=0):
                        pars[peak_prefix +'fraction'].set(value=fraction_value,min =fraction_value- fraction_range,max =fraction_value+ fraction_range)
                    else:
                        pars[peak_prefix +'fraction'].set(value=fraction_value,min =0,max =fraction_value+ fraction_range)
       
            if mod == 0:
                mod = voigti
            else:   
                mod = mod + voigti

    init = mod.eval(pars, x=x)
    out = mod.fit(y, pars, x=x)

    #Logging fitted values in the peak data matrix
    for i in range (1,number_of_peaks):
        if i not in peaks_to_ignore:
            sample_number = sample_number
            peak_name = "v" + str(i) + "_"
            peak_amplitude = out.params[peak_name + "amplitude"].value
            peak_center = out.params[peak_name + "center"].value
            peak_sigma = out.params[peak_name + "sigma"].value
            peak_fraction = out.params[peak_name + "fraction"].value
            peak_fwhm = out.params[peak_name + "fwhm"].value
            peak_height = out.params[peak_name + "height"].value
            new_row = [sample_number, peak_name, peak_amplitude, peak_center, peak_sigma, peak_fraction, peak_fwhm, peak_height]
            output_data_matrix = np.vstack([output_data_matrix, new_row])



    comps = out.eval_components(x=x)
    for i in range(1,number_of_peaks):
        if i not in peaks_to_ignore:
            peak_name = "v" + str(i) + "_"
            #print(peak_name)
            new_row = comps[peak_name]
            composition_fitted_peaks_matrix = np.vstack([composition_fitted_peaks_matrix, new_row])
            
    new_row = comps["spline_"]
    composition_fitted_peaks_matrix = np.vstack([composition_fitted_peaks_matrix, new_row])
    new_row = np.zeros((1,x.size))
    composition_fitted_peaks_matrix = np.vstack([composition_fitted_peaks_matrix, new_row])
    #print(out.fit_report(min_correl=0.5))
    #plt.plot(x, y, label = "Original XRD Data")
    #plt.plot(x, init, '--', label='initial fit')
    #plt.plot(x, out.best_fit, '-', label='best fit')
    #plt.legend()
    #ax = plt.gca()
    #ax.set_ylim([0, 30])
    fitted_peaks_matrix[sample_number] = out.best_fit


print(fitted_peaks_matrix)
np.savetxt("fitted_peaks_matrix31.csv", fitted_peaks_matrix, delimiter=",", fmt="%s")
np.savetxt("fitted_peak_parameters31.csv", output_data_matrix, delimiter=",", fmt="%s")
np.savetxt("composition_fitted_peaks_matrix31.csv", composition_fitted_peaks_matrix, delimiter=",", fmt="%s")