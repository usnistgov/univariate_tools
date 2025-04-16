#-----------------------------------------------------------------------------
# Name:        Interpolation
# Purpose:     To hold functions and classes for interpolating data
# Author:      Aric Sanders
# Created:     04/03/2025
# License:     MIT License
#-----------------------------------------------------------------------------
""" Interpolation holds classes and functions important for interpolating data

 """

#-----------------------------------------------------------------------------
# Standard Imports
import os
import sys
import re
#-----------------------------------------------------------------------------
# Third Party Imports
sys.path.append(os.path.join(os.path.dirname( __file__ ), '..'))
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.interpolate import make_smoothing_spline
import statsmodels.api as sm
from scipy import signal
from skmisc.loess import loess
import scipy
#-----------------------------------------------------------------------------
# Module Constants

#-----------------------------------------------------------------------------
# Module Functions


def reverse_regressor(x_data,y_data,new_y_data,method="lowess",**options):
    """reverse_regressor returns a series of new_x_data points given observed x_data, y_data and the desired 
    new_y_data. This function is intended as a wrapper to create a clean interface to the large number of interpolation
    possibilities.
    
    Current methods are lowess,loess, 1d, gpr and spline. 
    *********************************************************************************************
    refs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html, options for kind are
    ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, 
    ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’. ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’
    ****************************************************************************************
    https://www.statsmodels.org/dev/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html
    ***********************************************************************************************
    https://has2k1.github.io/scikit-misc/stable/index.html
    ******************************************************
    https://scikit-learn.org/stable/modules/gaussian_process.html
    ***************************************************************
    https://docs.scipy.org/doc/scipy/tutorial/interpolate/smoothing_splines.html
    ********************************************************************************"""
    defaults = {}
    interpolation_options = {}
    for key,value in defaults.items():
        interpolation_options[key] = value
    for key, value in options.items():
        interpolation_options[key] = value
    new_x_data = None 
        
    if re.search("lowess",method,re.IGNORECASE):
        lowess_key_words = ['frac']
        lowess_options = {'frac':.2}
        for key,value in interpolation_options.items():
            if key in lowess_key_words:
                lowess_options[key]=value
        interpolation_result= sm.nonparametric.lowess(endog=x_data,
                                                      exog=y_data,
                                                      xvals = new_y_data,
                                                      **lowess_options)
        new_x_data = interpolation_result
        
    if re.search("loess",method,re.IGNORECASE):
        loess_key_words = ['p','span','family','degree','normalize']
        loess_options = {"span":0.65, "p":1, "family":'gaussian', "degree":1, "normalize":False}
        for key,value in interpolation_options.items():
            if key in loess_key_words:
                loess_options[key]=value
        lo = loess(y_data, x_data, **loess_options)
        lo.fit()
        pred = lo.predict(new_y_data, stderror=True)
        new_x_data = np.array(pred.values)
        
    if re.search("1d",method,re.IGNORECASE):
        interp1d_key_words = ["kind","axis","copy","bounds_error","fill_value","assume_sorted"]
        interp1d_options ={"fill_value":"extrapolate"}
        for key,value in interpolation_options.items():
            if key in interp1d_key_words:
                interp1d_options[key]=value
        interpolation_function = scipy.interpolate.interp1d(y_data,
                                   x_data,**interp1d_options)
        new_x_data = interpolation_function(new_y_data)
        
    elif re.search("gpr",method,re.IGNORECASE):
        gpr_key_words = ["kind","axis","copy","bounds_error","fill_value","assume_sorted"]
        gpr_options ={"kernel":1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
                     "n_restarts_optimizer":9}
        for key,value in interpolation_options.items():
            if key in gpr_key_words:
                gpr_options[key]=value

        gaussian_process = GaussianProcessRegressor(**gpr_options)
        gaussian_process.fit(y_data.reshape(-1,1), x_data)
        mean_prediction, std_prediction = gaussian_process.predict(new_y_data.reshape(-1,1), return_std=True)
        new_x_data = mean_prediction
        
    elif re.search("spline",method,re.IGNORECASE):
        coordinates = zip(y_data,x_data)
        ordered_array = np.array(sorted(coordinates))        
        interpolation_function = scipy.interpolate.make_smoothing_spline(ordered_array.T[0],ordered_array.T[1])
        new_x_data = interpolation_function(new_y_data)   

    return new_x_data    

def interpolate_data(x_data,y_data,new_x_data,method="lowess",**options):
    """interpolate_data returns a series of new_y_data points given observed x_data, y_data and the desired 
    new_x_data. This function is intended as a wrapper to create a clean interface to the large number of interpolation
    possibilites. Current methods are lowess,loess, 1d, gpr and spline. 
    *********************************************************************************************
    refs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html, options for kind are
    ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, 
    ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’. ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’
    ****************************************************************************************
    https://www.statsmodels.org/dev/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html
    ***********************************************************************************************
    https://scikit-learn.org/stable/modules/gaussian_process.html
    ***************************************************************
    https://docs.scipy.org/doc/scipy/tutorial/interpolate/smoothing_splines.html
    ********************************************************************************"""
    defaults = {}
    interpolation_options = {}
    for key,value in defaults.items():
        interpolation_options[key] = value
    for key, value in options.items():
        interpolation_options[key] = value
    new_y_data = None 
        
    if re.search("lowess",method,re.IGNORECASE):
        lowess_key_words = ['frac']
        lowess_options = {}
        for key,value in interpolation_options.items():
            if key in lowess_key_words:
                lowess_options[key]=value
        interpolation_result= sm.nonparametric.lowess(y_data,
                                                      x_data,
                                                      xvals = new_x_data,
                                                      **lowess_options)
        new_y_data = interpolation_result
        
            
    if re.search("loess",method,re.IGNORECASE):
        loess_key_words = ['p','span','family','degree','normalize']
        loess_options = {"span":0.65, "p":1, "family":'gaussian', "degree":1, "normalize":False}
        for key,value in interpolation_options.items():
            if key in loess_key_words:
                loess_options[key]=value
        lo = loess(x_data, y_data, **loess_options)
        lo.fit()
        pred = lo.predict(new_x_data, stderror=True)
        new_y_data = np.array(pred.values)
        
    if re.search("1d",method,re.IGNORECASE):
        interp1d_key_words = ["kind","axis","copy","bounds_error","fill_value","assume_sorted"]
        interp1d_options ={"fill_value":"extrapolate"}
        for key,value in interpolation_options.items():
            if key in interp1d_key_words:
                interp1d_options[key]=value
        interpolation_function = scipy.interpolate.interp1d(x_data,
                                   y_data,**interp1d_options)
        new_y_data = interpolation_function(new_x_data)
        
    elif re.search("gpr",method,re.IGNORECASE):
        gpr_key_words = ["kind","axis","copy","bounds_error","fill_value","assume_sorted"]
        gpr_options ={"kernel":1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
                     "n_restarts_optimizer":9}
        for key,value in interpolation_options.items():
            if key in gpr_key_words:
                gpr_options[key]=value
        gaussian_process = GaussianProcessRegressor(**gpr_options)
        gaussian_process.fit(x_data.reshape(-1,1), y_data)
        mean_prediction, std_prediction = gaussian_process.predict(new_x_data.reshape(-1,1), return_std=True)
        new_y_data = mean_prediction
        
    elif re.search("spline",method,re.IGNORECASE):
        coordinates = zip(x_data,y_data)
        ordered_array = np.array(sorted(coordinates))        
        interpolation_function = scipy.interpolate.make_smoothing_spline(ordered_array.T[0],ordered_array.T[1])
        new_y_data = interpolation_function(new_x_data)
        
    return new_y_data





#-----------------------------------------------------------------------------
# Module Classes

#-----------------------------------------------------------------------------
# Module Scripts
def test_interpolate_data():
    x_data = np.linspace(-6,6,500)
    signal = np.sin(x_data)+np.random.normal(scale=.5,size=len(x_data))
    plt.plot(x_data,signal,".",label="Original Data")
    for interp_type in ["lowess","loess", "1d", "gpr","spline"]:
        new_x = np.linspace(-2,2,500)
        interp_data = interpolate_data(x_data=x_data,y_data=signal,new_x_data=new_x,method=interp_type)
        plt.plot(new_x,interp_data,label=interp_type)
    plt.legend()
    plt.show()

def test_reverse_regressor():
    x_data = np.linspace(-6,6,200)
    signal = 2*x_data+1+np.random.normal(scale=.2,size=len(x_data))
    plt.plot(x_data,signal,".",label="Original Data",alpha=.3)

    for interp_type in ["lowess","loess", "1d", "gpr","spline"]:
        try:
            new_y = np.linspace(-2,5,10)
            new_x = reverse_regressor(x_data=x_data,y_data=signal,new_y_data=new_y,method=interp_type)
            print(f"{interp_type}:{new_x},{new_y}")
            plt.plot(new_x,new_y,label=interp_type,linewidth=2,linestyle="dashed")
        except Exception as e:
            print(e)
    plt.legend()
    plt.show()

def test_lowess():
    x_data = np.linspace(-6,6,100)
    y_data = 2*x_data+1+np.random.normal(scale=.1,size=len(x_data))
    new_y_data = np.linspace(-2,5,10)
    interpolation_result= sm.nonparametric.lowess(endog=x_data,
                                                    exog=y_data,
                                                    xvals = new_y_data,
                                                    frac=.2)
    #print(interpolation_result)
    new_x = reverse_regressor(x_data=x_data,y_data=y_data,new_y_data=new_y_data,method="lowess")
    plt.plot(x_data,y_data,label="original data")
    plt.plot(interpolation_result,new_y_data,label="lowess fit")
    plt.plot(new_x,new_y_data,label="lowess fit 2")
    plt.show()

#-----------------------------------------------------------------------------
# Module Runner
if __name__ == '__main__':
    #test_interpolate_data()
    test_reverse_regressor()
    #test_lowess()