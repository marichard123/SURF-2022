# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 02:01:52 2022

@author: Richard
"""
from __future__ import annotations
print("This is the peak detection program")
minimum_fwhm = 1
#minimum_fwhm = 0
input_data_type = 1 #1= sqrt without background, 2 = background removed, 3= sqrt with background, 4 = raw data
if input_data_type ==1:
    add_background = True
    square_data = True
    two_theta_filename = 'two_theta_values.csv'
    hidden_units_matrix_filename ='sqrt_rank4_hidden_units.csv'
    background_matrix_filename = 'background_of_sqrt_data.csv'
    print("Assuming square-root transformed data with background removed")
elif input_data_type ==2:
    add_background = True
    square_data = False
    two_theta_filename = 'two_theta_values.csv'
    hidden_units_matrix_filename ='background_removed_rank7_hidden_units.csv'
    background_matrix_filename = 'background_of_raw_data.csv'
    print("Assuming background-removed data")
elif input_data_type ==3:
    add_background = False
    square_data = True
    two_theta_filename = 'two_theta_values.csv'
    hidden_units_matrix_filename ='sqrt_with_background_rank7_hidden_units.csv'
    background_matrix_filename = 'background_of_raw_data.csv'
    print("Assuming square-root transformed with background included")
else:
    add_background = False
    square_data = False
    two_theta_filename = 'two_theta_values.csv'
    hidden_units_matrix_filename ='raw_data_rank8_hidden_units.csv'
    background_matrix_filename = 'background_of_raw_data.csv'
    print("Assuming raw data")
    
    
    
    
import sys
#print(sys.path)
#C:\Users\Richard\Documents\NIST_Summer_2022\XRD\Peak Detection
sys.path.append("C:\\Users\\Richard\\Documents\\NIST_Summer_2022\\XRD\\Peak Detection")
from xrdprofile import *
import matplotlib.pyplot as plt
import lmfit
import numpy as np
import pandas as pd
from numpy.polynomial import chebyshev
from pymatgen.analysis.diffraction import xrd as pmgxrd
from scipy import interpolate, signal
from scipy import *
import csv
from numpy import genfromtxt
print("Packages imported successfully")
from xrdprofile.background import (
    ChebyshevPolyModel,
    sonneveld_visser_background,
    sonneveld_visser_noise_level,
)
from xrdprofile.utils import (
    amp_from_height,
    d2theta,
    estimate_prominence_threshold,
    slice_baseline,
    theta2d,
)


def build_doublet_pv(prefix):
    prefix2 = f"{prefix}alpha2_"

    doublet = lmfit.models.PseudoVoigtModel(
        prefix=prefix
    ) + lmfit.models.PseudoVoigtModel(prefix=prefix2)
    pars = doublet.make_params()

    # add a pseudovoigt profile for alpha_2, tied to the params for alpha_1
    pars[f"{prefix2}center"].set(
        expr=f"360*asin(1.00248*sin({prefix}center*pi/360))/pi"
    )
    pars[f"{prefix2}sigma"].set(expr=f"{prefix}sigma")
    pars[f"{prefix2}amplitude"].set(expr=f".4829*{prefix}amplitude")
    pars[f"{prefix2}fraction"].set(expr=f"{prefix}fraction")

    return doublet, pars


def build_doublet_voigt(prefix):
    prefix2 = f"{prefix}alpha2_"

    doublet = lmfit.models.VoigtModel(prefix=prefix) + lmfit.models.VoigtModel(
        prefix=prefix2
    )
    pars = doublet.make_params()

    # add a pseudovoigt profile for alpha_2, tied to the params for alpha_1
    # pars[f'{prefix2}center'].set(expr=f'360*asin(1.00248*sin({prefix}center*pi/360))/pi')
    pars[f"{prefix2}center"].set(
        expr=f"2 * degrees( asin( 1.00248 * sin(radians({prefix}center / 2))))"
    )
    pars[f"{prefix2}sigma"].set(expr=f"{prefix}sigma")
    pars[f"{prefix2}amplitude"].set(expr=f".4829*{prefix}amplitude")
    pars[f"{prefix2}gamma"].set(expr=f"{prefix}gamma")

    return doublet, pars


def build_doublet_voigt_recip(prefix):
    prefix2 = f"{prefix}alpha2_"

    doublet = lmfit.models.VoigtModel(prefix=prefix) + lmfit.models.VoigtModel(
        prefix=prefix2
    )
    pars = doublet.make_params()

    # add a pseudovoigt profile for alpha_2, tied to the params for alpha_1
    # tt: 360 * asin(1.00248 * sin(angle * pi / 360)) / pi
    #     360 * asin(1.00248 * sin(radians(angle/2))) / pi
    #     2 * degrees(asin(1.00248 * sin(radians(angle/2))
    # pars[f'{prefix2}center'].set(expr=f'360*asin(1.00248*sin({prefix}center*pi/360))/pi')
    # pars[f'{prefix2}center'].set(expr=f'2 * degrees( asin( 1.00248 * sin(radians({prefix}center / 2))))')
    # pars[f'{prefix2}center'].set(expr=f'theta2d( asin( 1.00248 * sin(d2theta({prefix}center))))')
    pars[f"{prefix2}center"].set(expr=f"1.00248 * {prefix}center")
    pars[f"{prefix2}sigma"].set(expr=f"{prefix}sigma")
    pars[f"{prefix2}amplitude"].set(expr=f"0.4829 * {prefix}amplitude")
    pars[f"{prefix2}gamma"].set(expr=f"{prefix}gamma")

    return doublet, pars


def build_doublet_pv_recip(prefix):
    prefix2 = f"{prefix}alpha2_"

    doublet = lmfit.models.PseudoVoigtModel(
        prefix=prefix
    ) + lmfit.models.PseudoVoigtModel(prefix=prefix2)
    pars = doublet.make_params()

    # add a pseudovoigt profile for alpha_2, tied to the params for alpha_1
    # tt: 360 * asin(1.00248 * sin(angle * pi / 360)) / pi
    #     360 * asin(1.00248 * sin(radians(angle/2))) / pi
    #     2 * degrees(asin(1.00248 * sin(radians(angle/2))
    # pars[f'{prefix2}center'].set(expr=f'360*asin(1.00248*sin({prefix}center*pi/360))/pi')
    # pars[f'{prefix2}center'].set(expr=f'2 * degrees( asin( 1.00248 * sin(radians({prefix}center / 2))))')
    # pars[f'{prefix2}center'].set(expr=f'theta2d( asin( 1.00248 * sin(d2theta({prefix}center))))')
    pars[f"{prefix2}center"].set(expr=f"1.00248 * {prefix}center")
    pars[f"{prefix2}sigma"].set(expr=f"{prefix}sigma")
    pars[f"{prefix2}amplitude"].set(expr=f"0.4829 * {prefix}amplitude")
    pars[f"{prefix2}fraction"].set(expr=f"{prefix}fraction")
    # pars[f'{prefix2}gamma'].set(expr=f'{prefix}gamma')

    return doublet, pars


def peakdet_model(
    x, y, error=None, bg_order=3, prominence=1, w=10, logscale=False, bg_ddof=0
):

    # dx = np.diff(x).mean()
    dx = np.diff(x)

    if prominence == "uncertainty":
        if error is None:
            raise ValueError(
                "uncertainty-based peak prominence thresholding requires experimental uncertainty estimates"
            )
        prominence = 1.5 * error

    elif prominence == "spectral":
        prominence = estimate_prominence_threshold(x, y)

    if logscale:
        peaks, info = signal.find_peaks(np.log10(y), prominence=prominence, width=w)
    else:
        peaks, info = signal.find_peaks(y, prominence=prominence, width=w)

    # idea: pre-fit polynomial background using peak bases obtained from find_peaks
    # maybe just something like model.guess(y[peak_bases], x=x[peak_bases])
    xbase, ybase = slice_baseline(x, y, peaks, info)

    # model = lmfit.models.PolynomialModel(bg_order, prefix='bkgnd_')
    model = ChebyshevPolyModel(bg_order, prefix="bkgnd_")
    pars = model.guess(ybase, x=xbase, ddof=bg_ddof)

    for index, peak_id in enumerate(peaks):
        center = x[peak_id]
        sigma = dx[peak_id] * info["widths"][index] / 2
        if logscale:
            amplitude = amp_from_height(10 ** info["prominences"][index], sigma)
        else:
            amplitude = amp_from_height(info["prominences"][index], sigma)

        peak, peak_pars = build_doublet_voigt_recip(f"peak{index}_")
        model += peak
        pars.update(peak_pars)
        pars[f"peak{index}_center"].set(value=center, min=x.min(), max=x.max())
        pars[f"peak{index}_sigma"].set(value=sigma, min=dx[peak_id])
        pars[f"peak{index}_amplitude"].set(value=amplitude, min=0)
        # pars[f"peak{index}_fraction"].set(vary=False)

    return model, pars


def guess_structure(Q, y, structure, strain=(0.95, 1.1), tt_range=(21.2, 180)):
    """ assume cubic structure for now... """
    xrd = pmgxrd.XRDCalculator()
    structure = structure.copy()

    yy = interpolate.interp1d(Q, y, bounds_error=False)

    pattern = xrd.get_pattern(structure, scaled=True, two_theta_range=tt_range)
    q_baseline = theta2d(pattern.x / 2) / 10

    intensities = []
    svals = np.linspace(strain[0], strain[1], 500)
    for s in svals:
        qq = q_baseline * s
        intensities.append(yy(qq))

    intensities = np.array(intensities)

    s = svals[intensities.argmax(axis=0)]
    # peaks = q_baseline * np.nanmedian(s)

    # re-compute the structure factor in case any new reflections are in range?
    # alternatively, use a large range to begin with, and post-filter
    structure.apply_strain(1 - np.nanmedian(s))
    pattern = xrd.get_pattern(structure, scaled=True, two_theta_range=tt_range)
    peaks = theta2d(pattern.x / 2) / 10

    return peaks, pattern.hkls, intensities


def peakdet_structure_model(x, y, error=None, structure=None, bg_order=3, bg_ddof=0):

    dx = np.diff(x)

    # get initial structure estimate
    sf_peaks, sf_hkls, II = guess_structure(x, y, structure)
    sf_hkls = np.array(sf_hkls)

    bg_x, bg_y, resid = sonneveld_visser_background(
        x, y, interval=10, tol=1e-9, err_thresh=1e-12, fix_bounds=(True, False)
    )
    noise_baseline, noise_std = sonneveld_visser_noise_level(
        y[::10] - bg_y, noise_thresh=3
    )

    c = chebyshev.chebfit(bg_x, bg_y, 15)
    bg = chebyshev.chebval(x, c)
    if (y - bg).min() < 0:
        # shift the background down
        c[0] += (y - bg).min()
        bg = chebyshev.chebval(x, c)

    # peaks, info = signal.find_peaks(y-bg_y, width=2, height=3*noise_std, prominence=noise_std)
    peaks, info = signal.find_peaks(
        y - bg, width=5, height=1.96 * noise_std, prominence=noise_std
    )

    peak_dx = dx[peaks]
    center = x[peaks]
    sigma = dx[peaks] * info["widths"] / 2
    amplitude = np.array(
        [amp_from_height(prom, sig) for prom, sig in zip(info["prominences"], sigma)]
    )

    detections = pd.DataFrame(
        dict(center=center, sigma=sigma, amplitude=amplitude, dx=peak_dx)
    )

    # not all peaks in the model are guaranteed to be present
    # so for each detected peak, map it to the closest model peak,
    # then for each model peak, choose the closest paired detection
    # could be more detected peaks than in the model
    # could be more model peaks than detected...
    # compute relative distances in d-spacing
    reldist = np.abs(1 / detections["center"][:, None] - 1 / sf_peaks) / (
        1 / detections["center"][:, None]
    )
    # return p, detections
    matches, _ = np.where(reldist < 0.005)

    # if there are more peak matches than detections... filter them
    match_scores = np.abs(
        sf_peaks[:, None] - detections.iloc[matches]["center"].values
    ).min(axis=1)
    n_keep = len(matches)
    keep_ids = np.sort(np.argsort(match_scores)[:n_keep])
    p = sf_peaks[keep_ids]
    hkls = sf_hkls[keep_ids]

    detections["phase_id"] = np.nan
    detections["model_center"] = np.nan
    detections["hkl"] = np.nan

    detections["phase_id"].iloc[matches] = 1
    detections["model_center"].iloc[matches] = p
    detections["hkl"].iloc[matches] = hkls

    # if there is a peak within a FWHM of the edge?
    # ok. but make sure it hasn't already been matched!
    (edge_peaks,) = np.where(
        (sf_peaks > x.max() - np.median(detections["sigma"]))
        & (sf_peaks <= x.max() + np.median(detections["sigma"]))
    )
    print("edge", edge_peaks)
    if edge_peaks[0] not in keep_ids:
        d = detections.iloc[-1].copy()
        d["center"] = sf_peaks[edge_peaks[0]]
        d["model_center"] = sf_peaks[edge_peaks[0]]
        d["sigma"] = np.min(detections["sigma"])
        d["amplitude"] = np.min(detections["amplitude"])
        d["hkl"] = sf_hkls[edge_peaks[0]]
        d["dx"] = dx[-1]
        detections = detections.append(d, ignore_index=True)

    # model = lmfit.models.PolynomialModel(bg_order, prefix='bkgnd_')
    model = ChebyshevPolyModel(bg_order, prefix="bkgnd_")
    pars = model.guess(bg_y, x=bg_x, ddof=bg_ddof)

    for phase_id, group in detections.groupby("phase_id"):
        phase_id = int(phase_id)

        ref = group.iloc[0]
        for index, (peak_id, peakdata) in enumerate(group.iterrows()):
            prefix = f"phase{phase_id}_peak{index}_"
            ref_prefix = f"phase{phase_id}_peak0_"

            peak, peak_pars = build_doublet_voigt_recip(prefix)
            model += peak
            pars.update(peak_pars)

            if index == 0:

                pars[f"{prefix}center"].set(
                    value=peakdata["center"], min=x.min(), max=x.max()
                )
                pars[f"{prefix}sigma"].set(value=peakdata["sigma"], min=peakdata["dx"])
                pars[f"{prefix}amplitude"].set(value=peakdata["amplitude"], min=0)  #
                pars[f"{prefix}gamma"].set(
                    value=peakdata["sigma"], vary=True, min=peakdata["dx"]
                )
                # pars[f"peak{index}_fraction"].set(vary=False)

            else:
                peak_ratio = peakdata["model_center"] / ref["model_center"]
                pars[f"{prefix}center"].set(expr=f"{peak_ratio} * {ref_prefix}center")
                # pars[f"{prefix}sigma"].set(expr=f"{ref_prefix}sigma", min=peakdata["dx"])
                pars[f"{prefix}sigma"].set(
                    value=peakdata["sigma"], min=peakdata["dx"], max=0.5
                )
                pars[f"{prefix}amplitude"].set(value=peakdata["amplitude"], min=0)
                # pars[f"{prefix}fraction"].set(expr=f"{ref_prefix}fraction")
                pars[f"{prefix}gamma"].set(
                    expr=f"{ref_prefix}gamma", min=peakdata["dx"]
                )

    unmatched = detections[np.isnan(detections["phase_id"])]
    for index, (peak_id, peakdata) in enumerate(unmatched.iterrows()):

        prefix = f"unmatched_peak{index}_"

        peak, peak_pars = build_doublet_voigt_recip(prefix)
        model += peak
        pars.update(peak_pars)

        pars[f"{prefix}center"].set(value=peakdata["center"], min=x.min(), max=x.max())
        pars[f"{prefix}sigma"].set(value=peakdata["sigma"], min=peakdata["dx"])
        pars[f"{prefix}amplitude"].set(value=peakdata["amplitude"], min=0)  #
        pars[f"{prefix}gamma"].set(
            value=peakdata["sigma"], vary=True, min=peakdata["dx"]
        )
        # pars[f"peak{index}_fraction"].set(vary=False)

    return model, pars, detections, sf_peaks


def optimize_phases(x, y, model, pars, error=None):
    def freeze(pars, names):
        for name in names:
            for key in filter(
                lambda key: name in key and "alpha2" not in key, pars.keys()
            ):
                if pars[key].expr is not None:
                    pars[key].set(vary=False)

        return pars

    def unfreeze(pars, names, unconstrain=False):
        for name in names:
            for key in filter(
                lambda key: name in key and "alpha2" not in key, pars.keys()
            ):
                if pars[key].expr is None or unconstrain:
                    pars[key].set(vary=True)

        return pars

    # # freeze_groups = [("bkgnd", "center", "sigma"), ("center", "sigma", "amplitude")]
    # freeze_groups = [("bkgnd", "center", "gamma"), ()]
    freeze_groups = [("bkgnd", "center", "sigma", "gamma"), ("bkgnd", "gamma"), ()]
    res = lmfit.model.ModelResult(model, pars)

    for group in freeze_groups:
        print(f"optimizing with {group} fixed")
        pars = freeze(pars, group)
        res.fit(y, x=x, weights=1 / error, params=pars)
        pars = unfreeze(res.params, group)

    print("final fit")
    pars = unfreeze(res.params, "center", unconstrain=True)
    res.fit(y, x=x, weights=1 / error, params=pars)

    return res


def optimize(x, y, model, pars, error=None):
    """staged XRD profile model optimization
    First fit peak amplitudes and widths with fixed background and locations
    Then relax the entire model
    """

    for name in filter(lambda key: "bkgnd" in key, pars.keys()):
        pars[name].set(vary=False)

    for name in filter(lambda key: "center" in key, pars.keys()):
        pars[name].set(vary=False)

    res_init = model.fit(y, x=x, params=pars)

    pars = res_init.params
    for name in filter(lambda key: "bkgnd" in key, pars.keys()):
        pars[name].set(vary=True)

    for name in filter(lambda key: "center" in key, pars.keys()):
        pars[name].set(vary=True)

    res = model.fit(y, x=x, params=pars, method="lbfgsb")

    return res


def refine(x, y, model, pars, error=None):
    
    
    """staged XRD profile model optimization
    First fit peak amplitudes and widths with fixed background and locations
    Then relax the entire model
    """
    freeze_groups = [("bkgnd", "center")]
    # freeze_groups = [("bkgnd", "center", "fraction"), ("fraction")]
    # freeze_groups = [("bkgnd", "center", "width"), ("bkgnd", "center")]

    # for voigt profiles
    # freeze_groups = [("bkgnd", "center", "gamma"), ("gamma")]

    def freeze(pars, names):
        for name in names:
            for key in filter(
                lambda key: name in key and "alpha2" not in key, pars.keys()
            ):
                pars[key].set(vary=False)

        return pars

    def unfreeze(pars, names):
        for name in names:
            for key in filter(
                lambda key: name in key and "alpha2" not in key, pars.keys()
            ):
                pars[key].set(vary=True)

        return pars

    res = lmfit.model.ModelResult(model, pars)
    # res._asteval.symtable['theta2d'] = theta2d
    # res._asteval.symtable['d2theta'] = d2theta

    for group in freeze_groups:
        pars = freeze(pars, group)
        # res = model.fit(y, x=x, weights=1/error, params=pars)
        res.fit(y, x=x, weights=1 / error, params=pars)
        pars = unfreeze(res.params, group)

    # res = model.fit(y, x=x, weights=1/error, params=pars, max_nfev=1000)
    res.fit(y, x=x, weights=1 / error, params=pars)

    return res



#x = np.array([1.0,2,3,4,5,6,7,8,9,10])
#y = np.array([2,2,2,2,4.0,2,2,2,2,2])
#y = np.array([2,2,2,2,4.0,2,2,2,17,2])
two_theta = genfromtxt(two_theta_filename, delimiter=',')
hidden_units_matrix = genfromtxt(hidden_units_matrix_filename, delimiter=',')

if add_background:
    print("Adding Background")
    background_matrix = genfromtxt(background_matrix_filename, delimiter=',')
    
    rank, length = hidden_units_matrix.shape
    
    background_array = np.zeros([1,length])
    for i in range(length):
        background_array[0,i] = np.average(background_matrix[:,i])
        for j in range(rank):
            hidden_units_matrix[j,i] = hidden_units_matrix[j,i] + background_array[0,i]
    
if square_data:
    print("Squaring")
    hidden_units_matrix = np.square(hidden_units_matrix)
        


peak_data_matrix = np.array(["Peak Name", "Center", "Amplitude", "Sigma", "Gamma", "FWHM"])

rank, length = hidden_units_matrix.shape

for i in range(rank):
    #model, pars = peakdet_model(two_theta,hidden_units_matrix[i])
    #model, pars = peakdet_model(two_theta,hidden_units_matrix[i],bg_order=2, prominence=.35, w=5)
    model, pars = peakdet_model(two_theta,hidden_units_matrix[i],bg_order=2, prominence=.5, w=3)
    print("Row " + str(i+1))
    plt.figure(i)
    plt.plot(two_theta, hidden_units_matrix[i])
    peak_values = []
    peak_keys = list(pars.valuesdict().keys())
    for ii in peak_keys:
        #print(ii)
        if ii[len(ii)-6:len(ii)] == "center": #isolating the center value of the dictionary
            #print(ii)
            fwhm_name = ii[0:len(ii)-6] + "fwhm"
            if pars.valuesdict()[fwhm_name] > minimum_fwhm:
                peak_values.append(pars.valuesdict()[ii])  #the peaks that are acutally identified as real
                
                #Appending the peak data values to the numpy output matrix
                peak_name = ii[0:len(ii)-7]
                #print(peak_name)
                peak_center = pars.valuesdict()[peak_name + "_center"]
                peak_amplitude = pars.valuesdict()[peak_name + "_amplitude"]
                peak_sigma = pars.valuesdict()[peak_name + "_sigma"]
                peak_gamma = pars.valuesdict()[peak_name + "_gamma"]
                peak_fwhm = pars.valuesdict()[peak_name + "_fwhm"]
                
                #print(peak_center)
                new_row = [peak_name, peak_center, peak_amplitude, peak_sigma, peak_gamma,peak_fwhm]
                #print(new_row)
                #A = numpy.vstack([A, newrow])
                peak_data_matrix = np.vstack([peak_data_matrix, new_row])
                
                
                
                
    plt.vlines(x = peak_values, ymin = 0, ymax = (np.amax(hidden_units_matrix[i])*1.1), colors = "r")
    
    print(" ")
print(peak_data_matrix)
#np.savetxt("CoCrAl_detected_peaks.csv", peak_data_matrix, delimiter=",", fmt="%s")
np.savetxt("CoCrAl_detected_peaks_with_silicon.csv", peak_data_matrix, delimiter=",", fmt="%s")
plt.show()