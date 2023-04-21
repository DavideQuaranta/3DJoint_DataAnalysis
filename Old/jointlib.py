#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import inspect
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from IPython.display import display


# In[ ]:


t_unit = "s"
v_unit = "mVpp"
T_unit = "K"
f_unit = "Hz"


# In[ ]:


def error(value):
    if value < 1000 and value >= 100:
      sensibility_error = 0.001/np.sqrt(12)
    if value < 100 and value >= 10:
      sensibility_error = 0.0001/np.sqrt(12)
    if value < 10 and value >= 1:
      sensibility_error = 0.00001/np.sqrt(12)
    if value < 1 and value >= 0.1:
      sensibility_error = 0.000001/np.sqrt(12)
    reading_error = 0.0292*value #2.92% of the value
    scale_error = 0.00025*1000*(100)
    err = np.sqrt((sensibility_error)**2 + (reading_error)**2 + (scale_error)**2)
    return err


# In[ ]:


def chidof(obs, exp, sigma, dof):
    obs_arr = np.array(obs)
    exp_arr = np.array(exp)
    sigma_arr = np.array(sigma)
    return sum((obs_arr - exp_arr)**2/sigma_arr**2) / dof


# In[ ]:


def fitdata(path, filename, sheet, f, initial, plot = False, verbose = False):
    file = pd.ExcelFile(path+filename)
    sheets = file.sheet_names
    data = pd.read_excel(path+filename, sheet_name=sheet)


    v = data['Voltage']
    v_error = [error(val) for val in v]

    t = data['Time']

    T = data['T']
    T_value = np.mean(T)
    T_error = abs(T[1]-T[0])/2

    f_0 = data['f0'][0]
    errf_0 = (0.003/100)*f_0


    resval, rescov = curve_fit(f, t, v, initial, sigma = v_error)
    reserr = np.sqrt(np.diag(rescov))

    dof = len(v) - len(initial)
    chisq = chidof(v, f(t, *resval), v_error, dof)

    params_names = inspect.getfullargspec(f)[0]
    params_names = params_names[1:]

    if plot == True :
        h = max([abs((max(t)-min(t))/1000),1])
        fit_time = np.arange(min(t), max(t)+h, h)
        fit_amplitude = f(fit_time, *resval)

        fig = plt.figure(figsize=(6,4), dpi=100);
        fig.suptitle(r"Data from {0} of {1}".format(sheet, filename))
        plt.xlabel(r"$t$ ({0})".format(t_unit), size = 10)
        plt.ylabel(r"$Amplitude$ ({0})".format(v_unit), size = 10)
        plt.plot(t,v,'.',c='k', ms=6)
        plt.errorbar(t, v, yerr=v_error, fmt=".k", capsize=3,alpha = 0.65,label="Data")
        plt.plot(fit_time,fit_amplitude,'--',c='red',label="Fit")


        if verbose == True:
            fit_strings = [r'{0} = {1:.2f} $\pm$ {2:.2f}'.format(params_names[i], resval[i], reserr[i]) for i in range(len(resval))]
            fit_strings.append(r'T = {0:.2f} $\pm$ {1:.2f} $\mathrm{{{2}}}$'.format(T_value,T_error,T_unit))
            fit_strings.append(r'$f_0$ = {0} $\pm$ {1} $\mathrm{{{2}}}$'.format(f_0,errf_0,f_unit))
            fit_strings.append(r'$\chi^2$/dof = {0:.2f} ({1})'.format(chisq, dof))
            textstr = '\n'.join((fit_strings))

            props = dict(boxstyle='square', facecolor='white', alpha=1)
            plt.text(0.6, 0.6, textstr, fontsize=8,
                    verticalalignment='top',transform=fig.transFigure, bbox=props)

        plt.legend()
        get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
        plt.tight_layout()
        plt.grid()
        plt.show()

    return [params_names, resval, reserr], chisq, T_value, T_error, f_0, errf_0


# In[ ]:


def getQ(tau, tau_err, f_0, errf_0):
    Q = np.pi * f_0 * tau
    errQ = Q*np.sqrt((tau_err/tau)**2 + (errf_0/f_0)**2)
    return Q, errQ
