#--------------------------------file notes-----------------------------------

# Creates a combined plot containing all map height histograms and their 
# normal curves. 

#------------------------------import modules---------------------------------

import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.stats as stats
import matplotlib.ticker as ticker


#--------------------------------user inputs----------------------------------

# Choose if figure is half or full page width
full_width = 6.75
half_width = 3.375
width = full_width # change this part
height = 0.75 * width # can set aspect ratio here

# Choose the order in which the plots appear, higher number = appear front. 
# Hash out any extra references. 
zorder = dict()
zorder[material[0]] = 5
#zorder[material[1]] = 3
#zorder[material[2]] = 4
#zorder[material[3]] = 2

#----------------------------start process timer------------------------------

start_time = time.perf_counter()
    
#-----------------------------process fit data--------------------------------

dataset = dict() # initialise variable

for i in material:

    df = material_fit_data[i] # (1) 
    temp = np.array(df.loc[df['Parameters'] == fit_var]) # (2)
    temp = np.delete(temp, 0) # (3)
    temp = temp.astype('float64') #(4)

    if fit_var == 'q':
        dataset[i] = abs(1/temp) # (5)
        axis_label = 'abs. Fano factor $| 1/q |$'
    elif fit_var == 'gammaD' or fit_var == 'gammaG':
        dataset[i] = 2*temp # (6) 
        axis_label = 'FWHM (cm$^{-1}$)'
    elif fit_var == 'peak D_G':
        dataset[i] = temp
        axis_label = 'D/G peak height ratio'
    elif fit_var == 'int D_G':
        dataset[i] = temp
        axis_label = 'D/G peak area ratio'        
    else:
        dataset[i] = temp
        axis_label = 'Raman shift (cm$^{-1}$)'

# (1) select material dataframe
# (2) create a temporary file w/ chosen variable data as a np array
# (3) delete first item in temp, this is a string w/ the variable name
# (4) set data type for manipulation
# (5) this is the fano factor, 1/q, take the absolute value
# (6) convert from half width half maxima (HWHM) to FWHM


#-----------------------remove outliers from dataset--------------------------


for i in material: 

    df = material_fit_data[i]
      
    #---reduced chi2---
    chi = df.loc[df['Parameters'] == 'red_Chi2'] # (1)
    chi = chi.drop(labels='Parameters', axis='columns') 
    chi = np.array(chi)
    size = chi.shape[1]
    chi = chi.reshape((size,))
    #------------------
    
    dataset[i] = dataset[i][(np.logical_and(chi>0.95, chi<thresh_Chi))] # (2)
    
    #---fit_var---
    fv = dataset[i]
    x0, sigma = stats.norm.fit(fv) 
    #-------------

    dataset[i] = dataset[i][(np.logical_and(fv> x0 - thresh_sig*sigma,
                                            fv< x0 + thresh_sig*sigma))] # (3)
    

    
# (1) get the mean and standard deviation data from the full dataset
# (2) if a data point is outwith the threshold range of reduced chi2 then
#     remove it. Also gets rid of NaNs.
# (3) if a data point is outwith a threshold number of standard deviations 
#     from the data's mean, then remove it. 


#---------------------------define boundaries---------------------------------

maxlist = list()
minlist = list()

# find max and min over all modified datasets to set the histogram range
for i in material:    
    minlist.append(np.amin(dataset[i]))
    maxlist.append(np.amax(dataset[i]))

# extend range to make graph prettier
if max(maxlist) > 2:
    minimum = int(min(minlist)) - 2
    maximum = int(max(maxlist)) + 2
else: # for 1/q
    minimum = min(minlist) - 0.025
    maximum = max(maxlist) + 0.025

if fit_var == 'q' or fit_var == 'peak D_G' or fit_var == 'int D_G':
    bins = int(round(100*(maximum-minimum))) + 1
else:
    bins = maximum - minimum + 1

binps = np.linspace(minimum,maximum,bins)

#---------------------------------plotting------------------------------------

# plot defaults
colors = ['cornflowerblue', 'plum', 'mediumseagreen', 'lightcoral']
lines = ['-.', ':', '--', '-'] # list of linestyles
alphas = [0.9,1,0.8,0.4] # list of transparancies

# initialise parameters
gaussian = dict()
histogram = dict()
s_err = dict()
counter = -1

# initialise the figure, it's dimensions and axis parameters 'ax'
fig, ax  = plt.subplots(figsize=(width,height))

x0 = dict()
sigma = dict()

for i in material:
    
    x0[i], sigma[i] = stats.norm.fit(dataset[i]) # get mean and standev   
    # create an evenly spaced array of 100 point between dataset range.
    xs = np.linspace(minimum, maximum, 100)
    gaussian[i] = stats.norm.pdf(xs,x0[i],sigma[i]) # create Guassian fit data
    s_err[i] = sigma[i]/np.sqrt(len(dataset[i])) # standard error of the mean
    
    counter += 1 # see 'histogram notes' below for more details
    histogram[i] = plt.hist(dataset[i], bins=binps,
                            rwidth=0.95, density=True,
                            alpha=alphas[counter], color=colors[counter],
                            label = i, zorder = zorder[i])
    
    plt.plot(xs, gaussian[i], linestyle=lines[counter], zorder = 10, 
             label = i + " stats.norm.pdf", color = "k")
    plt.legend(loc='upper left')
    plt.xlabel(axis_label)
    plt.ylabel('Probability density')
    
    if fit_var != 'q' and fit_var != 'peak D_G' and fit_var != 'int D_G':
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))    

# histogram notes:
    # density=True sets to a normalised probability density
    # rwidth sets the column width               
    
# save figure
fig.tight_layout()
#fig.savefig('C:/Users/Hector/Desktop/Data/Figures/' + ID + '_' + \
#            '_'.join(material) + '_' + fit_var +  '_histogram.pdf', dpi=1200)
fig.savefig('C:/Users/Hector/Desktop/Data/Figures/' + '_' + \
            '_'.join(material) + '_' + fit_var + '_histogram.tif', dpi=550)


#---------------------------------print data----------------------------------

for i in material:
    print("\n" + i + ": mean = (%.5f +/- %0.5f) cm-1 and sd = %.5f" % \
          (x0[i], s_err[i], sigma[i]))


#-----------------------------statistical tests-------------------------------

# All code in this section modified from machinelearningmaster.com
# "A Gentle Introduction to Normality Tests in Python"

for i in material:
    data = dataset[i]

    #-----Shapiro-Wilk Test-----
    from scipy.stats import shapiro
    stat, p = shapiro(data)
    print('\n---' + i + '---\nShapiro-Wilk Statistic: %.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
    	print('Looks Gaussian (fail to reject H0)')
    else:
    	print('Does not look Gaussian (reject H0)')
        
    #-----D'Agostino and Pearson Test----- 
    from scipy.stats import normaltest
    stat, p = normaltest(data)
    print('\nDAgostino and Pearson Statistic: %.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
    	print('Looks Gaussian (fail to reject H0)')
    else:
    	print('Does not look Gaussian (reject H0)')
        
    #-----Anderson-Darling Test-----
    from scipy.stats import anderson
    result = anderson(data)
    print('\nAnderson-Darling Statistic: %.3f' % result.statistic)
    p = 0
    for i in range(len(result.critical_values)):
    	sl, cv = result.significance_level[i], result.critical_values[i]
    	if result.statistic < result.critical_values[i]:
    		print('%.3f: %.3f, looks Gaussian (fail to reject H0))' % (sl, cv))
    	else:
    		print('%.3f: %.3f, does not look Gaussian (reject H0)' % (sl, cv))
        

        
#-----------------------------end process timer-------------------------------

# End process timer
end_time = time.perf_counter()
print("\nScript runtime: %.2f \bs" % (end_time - start_time))
# last runtime = 3.31s

#---------------------------------script end----------------------------------
