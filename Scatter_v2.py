#--------------------------------file notes-----------------------------------

# Creates an xy-scatter plot of e.g. FWHM vs D-peak pos.

# To do:
    # (1) elipse fit, try https://stackoverflow.com/questions/39693869/fitting-an-ellipse-to-a-set-of-data-points-in-python

#--------------------------------user inputs----------------------------------

# Choose the dataset from cenD, cenG, gammaD, gammaD, q
fit_var_x = 'cenD'
fit_var_y = 'gammaD'

# Choose if figure is half or full page width
full_width = 6.75
half_width = 3.375
width = full_width # change this part
height = 0.67 * width # can set aspect ratio here


#----------------------------start process timer------------------------------

start_time = time.perf_counter()


#---------------------------------plot data-----------------------------------

# plot defaults
colors = [f'C{i}' for i in range(10)] # standard matplotlib colour set


# define a linear equation for a least squares fit
def linear(x, m, c):
    return m*x + c

# initialise figure and axes
fig, ax = plt.subplots(figsize=(width,height)) 


x0 = dict()
sigma = dict()
counter = 0
for i in material:

    #---get x and y data---
    df = material_fit_data[i]
    
    ys = np.array(df.loc[df['Parameters'] == fit_var_y])
    ys = np.delete(ys, 0)
    ys = ys.astype('float64')
    ys = ys*2 # from HWHM to FWHM
    
    xs = np.array(df.loc[df['Parameters'] == fit_var_x])
    xs = np.delete(xs, 0)
    xs = xs.astype('float64')
    #----------------------
    
    #---reduced chi2---
    chi = df.loc[df['Parameters'] == 'red_Chi2'] # (1)
    chi = chi.drop(labels='Parameters', axis='columns') 
    chi = np.array(chi)
    size = chi.shape[1]
    chi = chi.reshape(size,)
    xs = xs[(np.logical_and(chi>0.95, chi<thresh_Chi))] 
    ys = ys[(np.logical_and(chi>0.95, chi<thresh_Chi))] 
    #------------------
    
    #---chuck 'bad' data---
    x0['xs'], sigma['xs'] = stats.norm.fit(xs) # get mean and sd info
    x0['ys'], sigma['ys'] = stats.norm.fit(ys)
    
    # Choose only data that is within the threshold
    ys = ys[(np.logical_and(xs < x0['xs'] + thresh_sig*sigma['xs'],
                            xs > x0['xs'] - thresh_sig*sigma['xs']))]        
    xs = xs[(np.logical_and(xs < x0['xs'] + thresh_sig*sigma['xs'],
                            xs > x0['xs'] - thresh_sig*sigma['xs']))]

    xs = xs[(np.logical_and(ys < x0['ys'] + thresh_sig*sigma['ys'],
                            ys > x0['ys'] - thresh_sig*sigma['ys']))] 
    ys = ys[(np.logical_and(ys < x0['ys'] + thresh_sig*sigma['ys'],
                            ys > x0['ys'] - thresh_sig*sigma['ys']))]
    #---------------------

    #---line of best fit---    
    m = 0
    c = max(ys)   
    popt, pcov = curve_fit(linear, xs, ys, p0 = (m, c))
    #----------------------
    
    #---plotting---
    plt.plot(xs,ys, '^', markersize = 1, label= i + " data",
             color = colors[counter])
 
    lin_fit = plt.plot(xs, linear(xs, *popt), color = colors[counter],
             label = i + " curvefit")
    
    plt.text(max(xs)-2.5,max(linear(xs,*popt))+0.5,
             "y = (%.1f)x + (%.1f)" % (popt[0], popt[1]),
             Bbox = dict(facecolor='w', alpha = 0.75, edgecolor='none',
                         pad=1))
     
    plt.legend()
    plt.xlabel('D peak position (cm$^{-1}$)')
    plt.ylabel('Full width half maximum (cm$^{-1}$)')
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    #-------------
    
    counter += 1
 
fig.tight_layout()    
#fig.savefig('C:/Users/Hector/Desktop/Data/Figures/' + ID + '_' + \
#            '_'.join(material) + '_' + fit_var_x + '_' + fit_var_y + \
#                "_scatter.pdf", dpi=1200)
fig.savefig('C:/Users/Hector/Desktop/Data/Figures/' + '_' + \
            '_'.join(material) + '_' + fit_var_x + '_' + fit_var_y + \
                "_scatter.tif", dpi=1200)
    

    
#-----------------------------end process timer-------------------------------

# End process timer
end_time = time.perf_counter()
print("\nScript runtime: %.2f \bs" % (end_time - start_time))
# last runtime = 5.79s

#---------------------------------script end----------------------------------