#--------------------------------file notes-----------------------------------

# Applies a Lorentzian fit to the D peak and Breit-Wigner-Fano fit to the G 
# peak. 

# To do:
#       1) Check - Calculate uncertainty of clean_data values and input those 
#          as 1D arrays to the sigma variable in scipy.optimize
#       2) Check that the integrals are right, surely we are wanting to find 
#          the area BETWEEN the curve and the baseline? 
#       3) p values?
#       4) Quantatative comparrison of fits
#       5) Error propagation (finish fit integral and check others)
#       6) Make a single point curve fitting file for examination


#--------------------------------user inputs----------------------------------

# Choose which data to chuck, below this threshold of reduced Chi2 (thresh_Chi)
# and fit_var standard deviations (thresh_sig) is kept.
thresh_Chi = 3
thresh_sig = 6

#------------------------------import modules---------------------------------

import time
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import scipy.integrate as integrate
import scipy.stats as stats
import matplotlib.pyplot as plt


#----------------------------start process timer------------------------------

start_time = time.perf_counter()

#-----------------------------define functions--------------------------------

# See cauchy distribution function at:
# https://www.itl.nist.gov/div898/handbook/eda/section3/eda3663.htm 
# Here, the  modified version with the max intensity (I0D) replaces the 
# the parameter (1/s*pi).
def Lorentzian(x, I0D, cenD, gammaD, m, c):
    return I0D*gammaD**2/((x-cenD)**2+gammaD**2) + (m * x + c)

# Function as described in SI of DOI: 10.1103/PhysRevB.84.241404    
def BWF(x, I0G, cenG, gammaG, q, m, c):
    numerator = (1+(x-cenG)/(q*gammaG))**2
    denominator = 1+((x-cenG)/(gammaG))**2
    return I0G*(numerator/denominator) + (m * x + c)

# Combined function of above two.
def LorBWF(x, I0D, cenD, gammaD, I0G, cenG, gammaG, q, m, c):
    numerator = (1+(x-cenG)/(q*gammaG))**2
    denominator = 1+((x-cenG)/(gammaG))**2
    return I0D*gammaD**2/((x-cenD)**2+gammaD**2) +\
        I0G*(numerator/denominator) + (m * x + c)

# Linear function used as baseline
def Baseline(x, m, c):
    return m * x + c


#--------------------------get & print fitting data---------------------------

# Choose data names for the dataframe, order must match the order that the 
# parameters are storred before entry.
data_names = ['I0D', 'er_I0D', 'cenD', 'er-cenD', 'gammaD', 'er-gammaD', 'I0G',
              'er-I0G', 'cenG', 'er-cenG', 'gammaG', 'er-gammaG', 'q', 'er-q',
              'm', 'er-m', 'c', 'er-c','Chi2','red_Chi2', 'peak D_G', 
              'er-peak D_G', 'int D_G', 'er-int D_G']

material_fit_data = dict()
report = dict()
for i in material: # for each material
    xmin = int(min(raw_data[i][:,1]))
    xmax = int(max(raw_data[i][:,1]))
    ymin = int(min(raw_data[i][:,0]))
    ymax = int(max(raw_data[i][:,0]))
    fit_data = pd.DataFrame({'Parameters': data_names})
    for x in range(xmin, xmax+xstep, xstep):
        for y in range(ymin, ymax+ystep, ystep):
            
            #----------------
            
            xs = clean_data[i][x,y][:,2]
            ys = clean_data[i][x,y][:,3]
            
            ys_err = np.sqrt(abs(ys)) # uncertainty in intensity values
            ys_err += 1*10**-20 # adds a tiny uncertainty to avoid dividing
                                # by zero later on in the code
            
            #--------------------------initial guesses------------------------    
        
            #--Lorentzian D-peak---
            I0D = max(ys[288:576]) # magnitude of the D peak
            cenD = 1340
            gammaD = 50 # half width at half maximum (HWHM)
            
            #--BWF G-peak---
            I0G = max(ys[:288]) # magnitude of the G peak
            gammaG = 35 # half width at half maximum (HWHM)
            cenG = 1600
            q = -4.5 # where 1/q is the Fano parameter
            
            #---Background---
            m = 0 # gradient
            c = min(ys) # intercept
    
            #---------------------find BWF/Lorentzian fit---------------------   
            
            # The below is taken from the documentation on scipy.org:
            # scipy.optimize.curve_fit(f, xdata, ydata, p0)
            # uses a non-linear least squares to fit a function, f, to data.
            # f = the model function
            # xdata = the independent variable where the data is measured
            # ydata = the dependent data
            # p0 = initial guess for the parameters of f
            # sigma = uncertainty in ydata
            # absolute_sigma = True, since the uncertainty in the ydata is 
            #   absolute and not relative
            # method = lm, since the problem is not constrained, this can be
            #   used and offers the fastest computational time. The other two 
            #   methods also work, but take about twice as long. All methods
            #   give the same reduced chi2 statistic. 
            
            # returns: 
            #   popt: optimal values for the parameters so that the sum of the 
            #         squared residuals of f(xdata, *popt) - ydata is minimized
            #   pcov: the estimated covarience of popt. The diagonals provide 
            #         the variance of the parameter estimate.
            
            checker = True # if the optimization works, checker stays True
            try:
                popt_LorBWF, pcov_LorBWF = curve_fit(LorBWF, xs, ys,
                                                     p0=[I0D, cenD,
                                                         gammaD, I0G,
                                                         cenG, gammaG,
                                                         q, m, c],
                                                     sigma = ys_err,
                                                     absolute_sigma=True,
                                                     method='lm')
            
            # if optimisation fails, set values to NaN and checker to False
            except RuntimeError: 
                checker = False
                popt_LorBWF = np.zeros(popt_LorBWF.shape)
                pcov_LorBWF = np.zeros(pcov_LorBWF.shape)
                popt_LorBWF[:] = np.nan              
                pcov_LorBWF[:] = np.nan
                print(i + " failed optimization at " + str((x,y)) + \
                      '\nreplaced with np.nan')
                fig, ax = plt.subplots()
                ax.plot(xs, ys, label= 'material: ' + \
                        i + '\ncoords: ' + str((x,y)))
                ax.legend()
                        
            #---sanity checks---
            if np.isnan(pcov_LorBWF).any():
                print(i + " nan at " + str((x,y)))
            if any(np.diag(pcov_LorBWF) < 0) == True:
                checker = False
                print(i + " negative covarience found at " + str((x,y)) + \
                      '\nreplaced with np.nan')
                popt_LorBWF = np.zeros(popt_LorBWF.shape)
                pcov_LorBWF = np.zeros(pcov_LorBWF.shape)
                popt_LorBWF[:] = np.nan              
                pcov_LorBWF[:] = np.nan
                fig, ax = plt.subplots()
                ax.plot(xs, ys, label= 'material: ' + \
                        i + '\ncoords: ' + str((x,y)))
                ax.legend()
            #-------------------
                            
            #-----------------------------stats-------------------------------
            
            residual = ys - (LorBWF(xs, *popt_LorBWF)) # (1)
            normres = residual/ys_err # (2)
            
            chi2 = sum(normres**2) # (3)
            K = len(xs) - len(popt_LorBWF) # (4)
            chi2_red = chi2/K # (5)
            

            perr_LorBWF = np.sqrt(np.diag(pcov_LorBWF)) # (6)
            
            
            # (1) calculate the residuals (difference between fit and data 
            #     points). 
            # (2) calculates the normalised residuals, res/err_dep_var, 
            #     in this experiment, the dependant var is the intensity, 
            #     which is a number of counts, the error is sqrt(N) - Chris
            # (3) Chi squared statistic, goodness of fit is maximised when 
            #     this is minimised - PHAS0007 UCL
            # (4) K is the number of degrees of freedom, number of variables 
            #     minus number of parameters.
            # (5) A reduced chi2 statistic close to one is 'good'
            # (6) This calculates the one standard deviation errors of the 
            #     fit parameters since var = sigma^2
            
            
            #--------------------------store paramaters-----------------------
            
            # combine fit parameters with their errors in the sequence:
            # param, param error, param, param error, etc.
            p_opt_err = np.zeros((len(popt_LorBWF)+len(perr_LorBWF))) # (1)
            counter = -1
            for j in range(len(p_opt_err)): # (2)
                if j % 2 == 0: # (3)
                    counter += 1 # (4)
                    p_opt_err[j] = popt_LorBWF[counter]               
                else: # (5)
                    p_opt_err[j] = perr_LorBWF[counter]
            
            # (1) initialise matrix.
            # (2) cycles through each element of initialised matrix.
            # (3) if the matrix element is odd, then insert a parameter.
            # (4) since there are half as many params/errors as there are
            #     params/errors combined, a counter is used to correctly
            #     index the parms/errors.
            # (5) if the matrix element is even, then insert an error.
            
            
            [I0D, cenD, gammaD, I0G, cenG, gammaG, q, m, c] = popt_LorBWF
            
            # ratio of intensity maxima, I0D/I0G
            D_to_G = I0D/I0G
            # error of this ratio, function derived from error propagation
            er_D_to_G = np.sqrt((perr_LorBWF[0]/I0G)**2 +\
                                ((I0D*perr_LorBWF[0])/(I0G**2))**2)
            
            # integrate individual curves to get their area, here custom
            # bounds that exceed the range of the dataset are used
            
            intmin = min(xs)
            intmax = max(xs)
            
            if checker == True: # if the optimisation was successful 
                intD_temp, abserrD = integrate.quad(Lorentzian, intmin, intmax,
                                               args=(I0D, cenD,
                                                     gammaD, m, c))
                
                intG_temp, abserrG = integrate.quad(BWF, intmin, intmax,
                                               args=(I0G, cenG,
                                                     gammaG, q, m, c))
                
                intbaseline, abserrb = integrate.quad(Baseline, intmin, intmax,
                                                      args=(m, c))
                
                intD = intD_temp - intbaseline
                intG = intG_temp - intbaseline
                
                
                # ratio of peak fit area
                int_D_to_G = intD/intG
                
                #---error propagation---
                # area of a Lorentzian lineshape
                d1 = (gammaD*(np.arctan((intmax-cenD)/gammaD) - \
                             np.arctan((intmin-cenD)/gammaD)))**2 * \
                      perr_LorBWF[0]**2
                
                d2 = (I0D*(np.arctan((intmax-cenD)/gammaD) - \
                           np.arctan((intmin-cenD)/gammaD)))**2 * \
                      perr_LorBWF[2]**2 
                      
                d3 = (gammaD*I0D*((1/(gammaD*((intmin-cenD)**2/gammaD**2 + 1))) - \
                                  gammaD*I0D*((1/(gammaD*((intmax-cenD)**2/gammaD**2 + 1))))))**2 * \
                    perr_LorBWF[1]**2
                    
                err_intD = np.sqrt(d1+d2+d3)
                
                # area of a BWF lineshape
                # !!! THIS IS JUST TEMPORARY !!!
                er_int_D_to_G = float(np.random.random(1)) #!!! NEED TO FINISH

            else: # if the optimisation failed
                int_D_to_G = np.nan
                er_int_D_to_G = np.nan
                
            
            # update dataframe variables, order must match 'data_names' 
            df_var = np.concatenate((p_opt_err, np.array([chi2, 
                                                            chi2_red,
                                                            D_to_G,
                                                            er_D_to_G,
                                                            int_D_to_G,
                                                            er_int_D_to_G])))
                        
            parameters = dict() # initialise dictionary
            for j in range(len(data_names)):
                # match parameters to dict
                parameters[data_names[j]] = df_var[j] 
                                   
            fit_data[x,y] = pd.DataFrame(list(parameters.values()))

    material_fit_data[i] = fit_data
    
    
    #---generate report---  
    df = material_fit_data[i] # simplify
    x0 = np.zeros(len(data_names)) # initialise
    sigma = np.zeros(len(data_names))
    x0_err = np.zeros(len(data_names))
    counter = 0
    
    for name in data_names: # cycle through parameters
        var = df.loc[df['Parameters'] == name] # for each parameter
        var = var.drop(labels='Parameters', axis='columns') # drop label
        var = var.dropna(axis=1) # drops NaNs
        var = np.array(var)# convert to array
              
        #---chuck 'bad' data---
        chi = df.loc[df['Parameters'] == 'red_Chi2']
        chi = chi.drop(labels='Parameters', axis='columns') 
        chi = chi.dropna(axis=1)
        chi = np.array(chi)
        var = var[(np.logical_and(chi>0.95, chi<thresh_Chi))]
        av, stdv = stats.norm.fit(var) 
        var = var[(np.logical_and(var> av - thresh_sig*stdv,
                                  var< av + thresh_sig*stdv))] 
        #----------------------
            
        
        x0[counter], sigma[counter] = stats.norm.fit(var) # get mean and stdv
        x0_err[counter] = sigma[counter]/np.sqrt(var.shape[0]) # standard error

        counter += 1
        
    report[i] = pd.DataFrame({'Parameters': data_names}) # create a dataframe
    report[i]['x0'] = x0 # save lists to dataframe
    report[i]['x0_err'] = x0_err
    report[i]['sigma'] = sigma
    
    # output to excel
    #report[i].to_excel('C:/Users/Hector/Desktop/Data/Reports/'+ \
                       #i + '_report.xlsx', sheet_name = i, index = False) 


#-----------------------------------------------------------------------------

# End process timer
end_time = time.perf_counter()
print("\nScript runtime: %.2f \bs" % (end_time - start_time))

# last runtime = 7.40s
# 400 micron = 50s

#---------------------------------Script End----------------------------------              
