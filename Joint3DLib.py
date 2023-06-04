import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# # Definition of the function to add uncertainty # #
def error(value):
    if value >= 1000:
      sensibility_error = 0.01/np.sqrt(12)
    if value < 1000 and value >= 100:
      sensibility_error = 0.001/np.sqrt(12)
    if value < 100 and value >= 10:
      sensibility_error = 0.0001/np.sqrt(12)
    if value < 10 and value >= 1:
      sensibility_error = 0.00001/np.sqrt(12)
    if value < 1 and value >= 0.1:
      sensibility_error = 0.000001/np.sqrt(12)
    reading_error = 0.0292*value #2.92% of the value
    scale_error = 0.00025*10**3
    err = np.sqrt( (sensibility_error)**2 + (reading_error + scale_error)**2 )
    return err


def chidof(obs, exp, sigma, dof):
    obs_arr = np.array(obs)
    exp_arr = np.array(exp)
    sigma_arr = np.array(sigma)
    return sum((obs_arr - exp_arr)**2/sigma_arr**2) / dof


def singleExp(x, a, tau):
    return a*np.exp(-x/tau)

def ExpConst(x, a, tau, c):
    return a*np.exp(-x/tau) + c


def fitSheet(directory, sheet, initial, f = singleExp, cut = None, verbose = False):
    file = pd.ExcelFile(directory)
    sheets = file.sheet_names
    
    if int(sheet[5:]) > int(sheets[-1][5:]):
        print(f'Error : This file has {sheets[-1][5:]} sheets! {sheet} does not exist!')
        return
    
    data = pd.read_excel(directory, sheet_name = sheet)
    
    v = np.array(data['Voltage'])
    t = np.array(data['Time']) 
    if cut != 0:
        t = np.array(t[v<cut])
        v = v[v<cut]
    v_error = np.array([error(val) for val in v])
    
       
    
    T = data['T']
    T_value = np.mean(T)
    T_error = abs(T[1]-T[0])/2

    f_0 = data['f0'][0]
    errf_0 = (0.003/100)*f_0

    t_unit = "s"
    v_unit = "mVpp"
    T_unit = "K"
    f_unit = "Hz"
    
    resval,rescov = curve_fit(f, t, v, initial, sigma = v_error)
    reserr = np.sqrt(np.diag(rescov))
    dof = len(v) - len(initial)
    chi_norm = chidof(v, f(t,*resval), v_error, dof)

    Q_value = f_0*resval[1]*np.pi
    Q_error = Q_value*np.sqrt( (reserr[0]/resval[0])**2 + (errf_0/f_0)**2 )
    
    if verbose == True:
        print('--------------------------------------------------')
        print(f'Fit of Sheet{sheet_num} from {directory[-10:-5]}')
        print(f'Q = {Q_value:.3f} +/- {Q_error:.3f}')
        print(f'T = {T_value:.2f} +/- {T_error:.2f} K')
        print(f'Chi-Squared/dof = {chi_norm:.2f} ({dof})')
        print('--------------------------------------------------')
    else:
        return Q_value, Q_error, T_value, T_error, chi_norm
    
    
    
    
    
    
    
def plotSheet(directory, sheet, initial, f = singleExp, cut = 0, save = False):
    file = pd.ExcelFile(directory)
    sheets = file.sheet_names
    
    if int(sheet[5:]) > int(sheets[-1][5:]):
        print(f'Error : This file has {sheets[-1][5:]} sheets! {sheet} does not exist!')
        return
    
    data = pd.read_excel(directory, sheet_name = sheet)
    
    v = np.array(data['Voltage'])
    t = np.array(data['Time']) 
    if cut != 0:
        t = np.array(t[v<cut])
        v = v[v<cut]
    v_error = np.array([error(val) for val in v])
    
    T = data['T']
    T_value = np.mean(T)
    T_error = abs(T[1]-T[0])/2

    f_0 = data['f0'][0]
    range=12.5 #Hz
    N_lines=400
    errf_0 = (range/N_lines)/np.sqrt(12)

    t_unit = "s"
    v_unit = "mVpp"
    T_unit = "K"
    f_unit = "Hz"
    
    resval,rescov = curve_fit(f, t, v, initial, sigma = v_error)
    reserr = np.sqrt(np.diag(rescov))
    dof = len(v) - len(initial)
    chi_norm = chidof(v, f(t,*resval), v_error, dof)

    Q_value = f_0*resval[1]*np.pi
    Q_error = Q_value*np.sqrt( (reserr[0]/resval[0])**2 + (errf_0/f_0)**2 )
    
    h = max([abs((max(t)-min(t))/1000),1])
    fit_time = np.arange(min(t), max(t)+h, h)
    fit_amplitude = f(fit_time, *resval) 

    fig = plt.figure(figsize=(6,4), dpi=100);
    fig.suptitle(r"Data from {0} of {1}".format(sheet, directory[-10:-5]))
    plt.xlabel(r"$t$ ({0})".format(t_unit), size = 10)
    plt.ylabel(r"$Amplitude$ ({0})".format(v_unit), size = 10)
    plt.plot(t,v,'.',c='k', ms=6)
    plt.errorbar(t, v, yerr=v_error, fmt=".k", capsize=3,alpha = 0.65,label="Data")
    plt.plot(fit_time,f(fit_time, *resval),'--',c='red',label="Fit")

    textstr = '\n'.join((
    r'Q = {0:.2f} $\pm$ {1:.2f}'.format(Q_value,Q_error),
    r'T = {0:.2f} $\pm$ {1:.2f} $\mathrm{{{2}}}$'.format(T_value,T_error,T_unit),
    r'$f_0$ = {0:.3f} $\pm$ {1:.3f} $\mathrm{{{2}}}$'.format(f_0,errf_0,f_unit),
    r'$\chi^2$/dof = {0:.2f} ({1})'.format(chi_norm, dof)))

    props = dict(boxstyle='square', facecolor='white', alpha=1)

    plt.text(0.6, 0.7, textstr, fontsize=10,
            verticalalignment='top',transform=fig.transFigure, bbox=props)

    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()

    if save == True:
        plt.savefig(r"Plots/ExpConst/Data_{0}_{1}".format(sheet, directory[-10:-5]), dpi = 600)
