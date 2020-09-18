
#---Plotting a single spectrum---
fig, ax = plt.subplots()
i = material[0]
coord = (-150,-150)
xs = clean_data[i][coord][:,2]
ys = clean_data[i][coord][:,3]
ax.plot(xs, ys, "o-", markersize=1, color="gray",
         label= 'material: ' + i + '\ncoords: ' + str(coord))
ax.legend(loc='lower left')
ax.set_xlabel('Raman shift (cm$^{-1}$)')
ax.set_ylabel('Intensity (arb. units)')
plt.tight_layout()
#--------------------------------

#---Plotting a fit---
i = material[0]
coord = (-100,0)
#---
df = material_fit_data[i]
fit_var = np.array(df[coord])[[0,2,4,6,8,10,12,14,16]]
d_fit = np.array(df[coord])[[0,2,4,14,16]]
g_fit = np.array(df[coord])[[6,8,10,12,14,16]]
#---
fig, ax = plt.subplots()
xs = clean_data[i][coord][:,2]
ys = clean_data[i][coord][:,3]
lorbwf = LorBWF(xs, *fit_var)
lor = Lorentzian(xs, *d_fit)
bwf = BWF(xs, *g_fit)
background = fit_var[7] * xs + fit_var[8]
#---
ax.plot(xs, ys, linestyle='-', color="grey", label= 'Raw Data', alpha = 0.5)
ax.plot(xs, lorbwf, linestyle='-', color="k", label= 'Fit ' + str(coord))

ax.plot(xs, lor, color="orange", alpha=0.5, label = 'Lorentzian')
ax.fill_between(xs, background, lor, facecolor="orange", alpha=0.6)

ax.plot(xs, bwf, color="cornflowerblue", alpha=0.5, label = 'BWF')
ax.fill_between(xs, background, bwf, facecolor="cornflowerblue", alpha=0.6)
#---
ax.legend(loc='upper right')
ax.set_xlabel('Raman shift (cm$^{-1}$)')
ax.set_ylabel('Intensity (arb. units)')

fig.savefig('C:/Users/Hector/Desktop/Data/Figures/fit_data.tif', dpi=500)
#--------------------