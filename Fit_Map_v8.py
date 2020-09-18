#--------------------------------file notes-----------------------------------

# Creates a combined plot containing all map g peak postition wrt their xy 
# coordinates for each material. Defualt is set up for a 1x2 comparrison. If 
# more data, play with nrows, ncols and width_ratios.

# To do:
    # (1) create my own colour code?
    
#--------------------------------user inputs----------------------------------

# Choose the dataset from cenD, cenG, gammaG, gammaD, q, peak D_G, int D_G
fit_var = 'cenG'

# Interpolation: choose either "lanczos" or "none"
inter = "none"

# Enter the grid size of the figure, here it is a one by two.
nrows = 1
ncols = len(material)

# Here we have a 21,21 array corresponding to 100x100 microns, the gap between
# two points is thus 4 microns, so by setting the gap to span 5 points, this 
# corresponds to a length of 20 microns.
scalebar_length = 5
scalebar_height = 1

# Choose if figure is half or full page width
full_width = 6.75
half_width = 3.375
width = full_width # change this part

if ncols == 1:
    height = 0.8*width # this roughly matches cbar height to map height
    wr = [1,0.08] # sets the width ratios for the map 
elif ncols == 2:
    height = 0.4 * width
    wr = [1,1,0.08]
elif ncols == 3:
    height = 0.275 * width
    wr = [1,1,1,0.08]

#------------------------------import modules---------------------------------

import matplotlib as mpl
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import copy
import time
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0}) # supresses runtime info

#----------------------------start process timer------------------------------

start_time = time.perf_counter()


#-----------------------------process fit data--------------------------------

mat_grid = dict() # initialise variable
chuck = dict()

for i in material:
    xmin = int(min(raw_data[i][:,1])) # see data and image for column index
    xmax = int(max(raw_data[i][:,1]))
    ymin = int(min(raw_data[i][:,0]))
    ymax = int(max(raw_data[i][:,0]))
    
    #---creat dataframe of coordinates---
    temp = material_fit_data[i].drop(labels='Parameters', axis='columns') # (1)     
    coords = list(temp.columns) # list of all coords (x,y)
    xcoords = list()
    ycoords = list()
    for j in range(len(coords)):
        xcoords.append(coords[j][0]) # take all x coords
        ycoords.append(coords[j][1])# take all y coords
    xset = list(set(xcoords)) # make set to get rid of multiples
    yset = list(set(ycoords)) # then list so that it can be sorted below
    xset.sort() # sort in ascending order
    yset.sort()
    grid = pd.DataFrame(index = yset, columns = xset, dtype='float64') # (2)
    
    # (1) make a temporary dataframe of all fit data without variable names,
    #     this is to allow for data manipulation (removes the 'string')
    # (2) make a 2D grid of x and y coordinates
    #------------------------------------
    
    #---get relevant fit data---
    df = material_fit_data[i] # for ease of writing
    data = df.loc[df['Parameters'] == fit_var] # (1)
    data = data.drop(labels='Parameters', axis='columns') # (2)
    
    # (1) make dataframe (data) containing only the fit variable of interest
    # (2) drop the parameters column to allow for data manipulation
    #---------------------------
    
    #---tweak fit data---
    if fit_var == 'q':
        data = abs(1/data) # (1)
        axis_label = 'abs. Fano factor $| 1/q |$'
        decimals = '%.3f'
    elif fit_var == 'gammaD' or fit_var == 'gammaG':
        data = 2*data # (2) 
        axis_label = 'FWHM (cm$^{-1}$)'
        decimals = '%.0f'
    elif fit_var == 'peak D_G':
        data = data
        axis_label = 'D/G peak height ratio'
    elif fit_var == 'int D_G':
        data = data
        axis_label = 'D/G peak area ratio'   
    else:
        data = data
        axis_label = 'Raman shift (cm$^{-1}$)'
        decimals = '%.0f'
        
    # (1) this is the fano factor, 1/q, take the absolute value
    # (2) convert from half width half maxima (HWHM) to FWHM
    #--------------------     

    #---reduced chi2---
    chi = df.loc[df['Parameters'] == 'red_Chi2']
    chi = chi.drop(labels='Parameters', axis='columns') 
    chi_na = chi.dropna(axis=1)
    #------------------
    
    #---data stats---
    dstat = data.dropna(axis=1) # drop any NaNs from failed fits
    dstat = np.array(dstat)[np.where(chi_na<thresh_Chi)] # ignore 'bad' data
    x0, sigma = stats.norm.fit(dstat) 
    #----------------
    
    #---populate grid---
    chuck[i] = list()
    for x in range(xmin, xmax+xstep, xstep):
        for y in range(ymin, ymax+ystep, ystep):
            dp = float(data[x,y])
            dpchi = float(chi[x,y])
            dpdata = float(data[x,y])
            if dpchi > thresh_Chi or dpchi < 0.95: # (1)
                dp = np.nan
                chuck[i].append((x,y)) # (2)
            if dpdata > x0 + thresh_sig*sigma or \
                dpdata < x0 - thresh_sig*sigma: # (3)
                dp = np.nan
                chuck[i].append((x,y))
            grid[x][y] = dp
    
    # (1) if the datapoint (dp) of interest lies outwith the reduced Chi2
    #     threshold, then set to NaN
    # (2) record any spectra that have been chucked, for visual inspection
    # (3) if the dp lies outwith the threshold number of standard deviations 
    #     from the data set left by (1), then set to NaN

    
    mat_grid[i] = grid # assign grid to material

#----------------------------plot "chucked data"------------------------------

for i in material:
    counter = 0
    for j in range(len(chuck[i])):
        
        coord = chuck[i][counter] # get coordinates
        
        cleanx = clean_data[i][coord][:,2] # get clean data
        cleany = clean_data[i][coord][:,3]
        
        # get fitting parameters
        popt_LorBWF = np.array(material_fit_data[i][chuck[i][counter]][[0,2,4,6,8,10,12,14,16]])
        
        # fitting p's for Lorenzian peak
        pars_1 = popt_LorBWF[[0,1,2,7,8]]
        Lor_peak = Lorentzian(cleanx, *pars_1)
        
        # fitting p's for BWF peak
        pars_2 = popt_LorBWF[[3,4,5,6,7,8]]
        BWF_peak = BWF(cleanx, *pars_2)
        
        # define sloping background eqn
        background = popt_LorBWF[7] * cleanx + popt_LorBWF[8]
        
        # initialise new figure
        fig, ax = plt.subplots(figsize=(width,width)) 
    
        # plot clean raman data
        ax.plot(cleanx, cleany, "o-", markersize=1, color="gray",
                 label= 'material: ' + i + '\ncoords: ' + str(coord))     

        # plot D peak
        ax.plot(cleanx, Lor_peak, color="orange", alpha=0.5)
        ax.fill_between(cleanx, background, Lor_peak,
                         facecolor="orange", alpha=0.5)
          
        # plot G peak
        ax.plot(cleanx, BWF_peak, color="cornflowerblue", alpha=0.5)
        ax.fill_between(cleanx, background, BWF_peak,
                         facecolor="cornflowerblue", alpha=0.5)  
        
        # plot fit: scipy.optimize.curve_fit
        ax.plot(cleanx, LorBWF(cleanx, *popt_LorBWF), 'k--', 
                 label= "curvefit")
                   
          
        ax.set_xlabel('Raman shift (cm$^{-1}$)')
        ax.set_ylabel('Intensity (arb. units)')
        ax.legend()
        ax.set_title(r'\textbf{Spectrum set to NaN}', color = 'red')
        plt.tight_layout() 
        
        counter += 1
        
#--------------------------------find range-----------------------------------

maxlist = list()
minlist = list()
for i in material:
    minlist.append(np.nanmin(np.array(mat_grid[i]))) # (1)
    maxlist.append(np.nanmax(np.array(mat_grid[i])))
if max(maxlist) > 2: # (2)
    minimum = int(min(minlist)) - 2
    maximum = int(max(maxlist)) + 2
else:
    minimum = min(minlist) - 0.025
    maximum = max(maxlist) + 0.025

# (1) this creats a list with the max/min of each material, ignoring any 
#     numpy NaNs
# (2) this is to accomodate for the Fano parameter, integer values give
#     a nicer colorbar scale. The additions ensure the top range of the cbar 
#     is not used, I think this makes things clearer. 


#---------------------------------plot data-----------------------------------

# if the number of maps is not two, then must change 'width_ratios'.
fig = plt.figure(figsize=(width,height)) 
gs = GridSpec(nrows, ncols+1, figure=fig,
              left=0.05, right=0.85, top=0.95, bottom=0.05,
              wspace=0.1, width_ratios=wr)

arts = list()
axs = list()

counter = -1
for nrow in range(nrows):
    for ncol in range(ncols):
        counter += 1
        data = material[counter] # links material name to gridspec coord
        
        masked_array = np.ma.array(mat_grid[data],
                                   mask=np.isnan(mat_grid[data])) # (1)
        cmap = copy.copy(mpl.cm.jet) # (2)
        cmap.set_bad('magenta', 1.) # (3)
        
        # (1) create a masked array, that overlays a boolean array that gives 
        #     True if condition np.isnan is met
        # (2) create a copy of a colour scale
        # (3) if colour scale encounteres True/"1" then set to megenta
        
        
        ax = plt.subplot(gs[nrow,ncol]) # save gridspec location in list 'ax'  
        art = ax.imshow(masked_array, vmin = minimum, vmax = maximum,
                        cmap=cmap, interpolation=inter)  # save map image in list 'art'
        axs.append(ax) # append ax and art to their repective lists,
        arts.append(art) # these will have the same index

# add colourbar to last gridspec axis        
cb = plt.colorbar(arts[0], cax=plt.subplot(gs[:,-1]))
cb.set_label(axis_label)

# add label to images, transfrom makes numbers represent grid type coords
alphabet = ['a','b','c','d','e','f']
for i in range(len(material)):
    fig.text(0,1.025, r'\bf\large ' + alphabet[i], transform=axs[i].transAxes)


# remove ticks/add a scalebar
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
    bar_height = scalebar_height # scalebar height, in pixels
    # scalebar start position
    pos = (ax.get_xlim()[0] + 1, ax.get_ylim()[0] - 1 - bar_height) 
    ax.add_patch(Rectangle(pos, scalebar_length, bar_height, color='w'))


#fig.savefig('C:/Users/Hector/Desktop/Data/Figures/' + '_' + \
#            '_'.join(material) + '_' + fit_var +  '_colourmap.pdf', dpi=1200)
fig.savefig('C:/Users/Hector/Desktop/Data/Figures/' + '_' + \
            '_'.join(material) + '_' + fit_var + '_colourmap.tif', dpi=1200)

#-----------------------------end process timer-------------------------------

# End process timer
end_time = time.perf_counter()
print("\nScript runtime: %.2f \bs" % (end_time - start_time))

# last runtime = 10.21s

#---------------------------------script end----------------------------------