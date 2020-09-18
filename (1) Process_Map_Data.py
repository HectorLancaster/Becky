#--------------------------------File notes-----------------------------------

# Filenames must be named as "x map.txt" where x is the material ID.
# The filepath is C:\\Users\\Hector\\Desktop\\Data\\Map_Data\\ so this must
# be changed if the file is in a different location.

# This code produces a library of spectra with the keys being the materials'
# names, and within those keys a second key being the xy position. e.g. to 
# access KC10 spectra x=0, y=0 then type: sliced_data['material ID'][0,0]


#---------------------------------User Inputs---------------------------------

# Step size in data in x (column 2) and y (column 1):
xstep = 5
ystep = 5

# Modified z-score threshold:
threshold = 6


#-------------------------------Import Modules--------------------------------

import numpy as np
import time
import glob
import copy


#--------------------------------Import Data----------------------------------

# Start process timer
start_time = time.perf_counter() 

# Create a list of all files in filepath that have the form *.txt
filenames = sorted(glob.glob("C:\\Users\\Hector\\Desktop\\Data\\Map_Data\\*.txt"),
                   reverse = False)

# Extract the material names from the filepath name
material = list()
for i in range(len(filenames)):
    material.append(filenames[i][38:-8]) # Corresponds to position of name


# This numpy method loads the text in as an array   
raw_data = dict()
for i in range(len(filenames)):
        raw_data[material[i]] = np.loadtxt(filenames[i])


#---------------------------------Slice Data----------------------------------

# Slice data into individual spectra
sliced_data = dict()
for i in material:
    xmin = int(min(raw_data[i][:,1])) # see data and image for column index
    xmax = int(max(raw_data[i][:,1]))
    ymin = int(min(raw_data[i][:,0]))
    ymax = int(max(raw_data[i][:,0]))
    spectra = dict()
    for x in range(xmin, xmax+xstep, xstep):
        for y in range(ymin, ymax+ystep, ystep):
            posx = np.where(raw_data[i][:,1] == x) # coords when true
            posy = np.where(raw_data[i][:,0] == y) # coords when true
            inter = np.intersect1d(posx,posy) # intersection of posx & posy
            spectra[x,y] = raw_data[i][inter] # assign each spectrum to dict
    sliced_data[i] = spectra # for each material, embed spectra dict in dict


#--------------------------------Data Cleaning--------------------------------

# Method adapted from: 
# Whitaker, D and Hayes, K. “A simple algorithm for despiking Raman spectra.”
# Chemometrics and Intelligent Laboratory Systems 179 (2018): 82–84.


#---define zscore---

def modified_z_score(f): # Find the modified z-score of a dataset f
    median = np.median(f)
    mad = np.median([np.abs(f-median)]) # Median absolute deviation (MAD)
    z = 0.6745 * (f - median) / mad # The z-score, 0.6745 adjusting factor
    return z


#---define fixer---

def fixer(f, z_bool, ma=10):
    n = len(z) # for indexing the arrays, (575,)
    f_out = f.copy()
    z_bool[0] = z_bool[n-1] = True # set the first and last value to True
    spikes = np.array(np.where(z_bool == True)) # gives spike locations
    spikes = np.reshape(spikes,(len(spikes[0]),)) # (1)
    for j in spikes: # uses the locations of spikes as j values
        w = np.arange(max(0, j - ma), min(n, j + 1 + ma)) # (2)
        w = w[z_bool[w] == 0] # returns the non-peak locations from the range
        f_out[j] = np.mean(f[w]) # (3)
        #f_out[j] = np.nan
    return f_out # (576,)

# (1) must reshape from (1,len) to (len,) for the cycling through j in the for
# loop to work.
# (2) sets a range around j, if close to array boundaries, limits range to
# boundary, +1 becuase the way python references does not include top value.
# (3) replace peak with mean of surrounding non-peak data.


#---clean data---

# num_spikes = dict() this would be to count number of spikes/extremes removed
clean_data = copy.deepcopy(sliced_data) # need deepcopy otherwise overwrites
for i in material:
    # num_spikes[i] = 0
    xmin = int(min(raw_data[i][:,1])) # see data and image for column index
    xmax = int(max(raw_data[i][:,1]))
    ymin = int(min(raw_data[i][:,0]))
    ymax = int(max(raw_data[i][:,0]))
    spectra = dict()
    for x in range(xmin, xmax+xstep, xstep):
        for y in range(ymin, ymax+ystep, ystep):
            f = sliced_data[i][(x,y)][:,3] # (1)
            z = modified_z_score(np.diff(f)) # (2)   
            z_bool = np.abs(z) > threshold # (3)
            f_fixed = fixer(f,z_bool) # send to fixer function
            clean_data[i][(x,y)][:,3] = f_fixed # (4)
            
            # num_spikes[i] = num_spikes[i] + len(np.setdiff1d(f,f_fixed))

# (1) create a (n,) array of intensity values f
# (2) send a (n-1,) array of differences in f to zscore function
# (3) create a (n-1,) array of booleans which give true if a spike is detected
#     in the data, correspinding to a zscore greater than the threshold.
# (4) update column w/ despiked data

    
#----------------------------G-Peak Normalisation-----------------------------

norm_data = copy.deepcopy(clean_data)
for i in material:
    xmin = int(min(raw_data[i][:,1])) # see data and image for column index
    xmax = int(max(raw_data[i][:,1]))
    ymin = int(min(raw_data[i][:,0]))
    ymax = int(max(raw_data[i][:,0]))
    flength = (norm_data[i][(xmin,ymin)][:,3]).shape[0]
    spectra = dict()
    for x in range(xmin, xmax+xstep, xstep):
        for y in range(ymin, ymax+ystep, ystep):
            f = norm_data[i][(x,y)][:,3] # intensity data column
            max_g = max(f[:flength//2]) # max in top half of column
            f_norm = f/max_g # normalise to g_peak w/ value = 1
            norm_data[i][(x,y)][:,3] = f_norm # replace intensity data
            

#--------------------------------Average Data---------------------------------

av_data = dict() # dictionary for storing averaged spectra intensity
for i in material:
    xmin = int(min(raw_data[i][:,1])) # see data and image for column index
    xmax = int(max(raw_data[i][:,1]))
    ymin = int(min(raw_data[i][:,0]))
    ymax = int(max(raw_data[i][:,0]))
    f_sum = np.zeros(norm_data[i][xmin,ymin][:,3].shape) # initialise array
    for x in range(xmin, xmax, xstep):
            for y in range(ymin, ymax, ystep):  
                spectrum = copy.deepcopy(norm_data[i][x,y][:,3]) # (1)
                f_sum = f_sum + spectrum # (2)
    ys = f_sum/len(clean_data[i]) # (3)
    xs = copy.deepcopy(norm_data[i][(xmin,ymin)][:,2]) # (4)
    temp = np.zeros((len(ys), 2)) # (5)
    temp[:,0] = xs # (6)
    temp[:,1] = ys
    av_data[i] = temp   

# (1) copy all intensity data for a specific spectrum
# (2) adds together all intensity data at each index
# (3) devide summed data by the number of spectra to get the average
#     intensity data to use as the ycoords for the average spectrum
# (4) copy all the spectral wavenumber coords, which are consistent over the 
#     the whole map, to use as the xcoords for the average spectrum
# (5) create a temporary zero array to add data to
# (6) add data to respective array columns then save in dictionary 
            
            
#-----------------------------------------------------------------------------

# End process timer
end_time = time.perf_counter()
print("\nScript runtime: %.2f \bs" % (end_time - start_time))

# last runtime = 8.11s
# 400 micron map = 227s / 3:47

#---------------------------------Script End----------------------------------
