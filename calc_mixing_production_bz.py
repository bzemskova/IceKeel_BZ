# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 14:07:47 2023

@author: Barbara
"""

# Import modules
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# Import density field
rho = json.load(open('F20H20_rho_220.txt')) # F20H20 (vortex shedding regime) at snapshot t=220t_0 (1280x640 2D array)
rho = np.array(rho)[:, ::-1] # Reorder so rho[0][0] is top left of domain

#%% Constants / set-up
z0 = 8 # Depth of mixed layer [m]
rho1 = 1022.49 # Density of mixed layer (used as reference density rho0)
rho2 = 1024.12 # Density of deeper ocean
mu = 2e-3 # Salt-mass diffusivity [m^2/s]
g = 9.8 # Acceleration due to gravity [m/s^2]
L = 960 # Domain length [m]
H = 80 # Domain height [m]
h = 2*z0 # Keel height (for F20H20) [m]
l = 75*z0 # Keel center location [m]
Nx = 1280 # Number of grid points in horizontal
Nz = 640 # Number of grid points in vertical
x = np.linspace(0, L, Nx) # Horizontal grid points [m]
z = np.linspace(0, H, Nz) # Vertical grid points (increasing downwards) [m]

sigma = 3.9 * h # Keel characteristic width
dx = L/Nx # Grid spacing in x
dz = H/Nz # Grid spacing in z
vol = dx*dz # "Volume" (area) of each grid cell

zv, xv = np.meshgrid(z,x) # create meshgrid of (x,z) coordinates

Nx_mid = int(np.where(np.abs(x-l) == np.min(np.abs(x-l)))[0])

#%% Functions

def keel(h, l, x):  # Eqn for keel (from SD)
    """
    h: Keel maximum height
    l: Where the keel is centered in the domain
    """
    sigma = 3.9 * h # Keel characteristic width
    return h*sigma**2/(sigma**2+4*(x-l)**2)


def find_mask(h, l, Nx, Nz, zv): # Mask out the keel based on cell height
    keel_height = keel(h, l, x)
    for ind in range(Nx):
        for indz in range(Nz):
            if zv[ind,indz]<=keel_height[ind]:
                zv[ind,indz]=np.NaN
    zv_masked = np.ma.masked_invalid(zv)
    keel_mask = np.ma.getmask(zv_masked)
    return keel_mask  # returns mask of which elements are within the keel

def pad(data):
    bad_indexes = np.isnan(data)
    good_indexes = np.logical_not(bad_indexes)
    good_data = data[good_indexes]
    interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
    data[bad_indexes] = interpolated
    return data

def find_zstar(rho,h,l,Nx,Nz,zv,Nx1,Nx2,dz,vol,upstream_flag): 
    # Find the profile z*(rho) and dz*/drho for each non-masked element
    #       within the domain [Nx1:Nx2, :] (can separate down- and upstream)
    
    # upstream_flag = 1 (= 0) looking at upstream (downstream)
    #   Need this flag to distinguish positive and negative roots of the keel
    #       function
    
    keel_mask = find_mask(h, l, Nx, Nz, zv)
    rho_masked = np.ma.array(rho,mask=keel_mask) #mask the keel in rho
    
    num_elem = (Nx2-Nx1)*Nz #number of elements in this domain
    
    #get rho in the proper domain part (upstream or downstream), reshape
    #   as a column vector, and fill the masked parts with NaN
    rho_dom_col = rho_masked[Nx1:Nx2,:].reshape((num_elem,1)).filled(np.NaN)
    
    #column vector of elements numbers, so that we can keep track of where
    #   each cell goes back to in the physical (x,z) space
    elem_ind = np.arange(num_elem).reshape((num_elem,1))
    
    #remove the elements that are within the keel (marked with NaNs)
    elem_ind = elem_ind[~np.isnan(rho_dom_col)]
    rho_dom_col = rho_dom_col[~np.isnan(rho_dom_col)]
    
    n = len(rho_dom_col) # How many elements we have not in the keel
    #column vector of each cell volume (here all cells are the same)
    vol_ar = vol*np.ones((n,1))
    #put all three columns (rho, cell volume, and cell element number) together
    ar = np.concatenate((rho_dom_col.reshape(n,1),
                         vol_ar,
                         elem_ind.reshape(n,1)),
                        axis=1)
    
    #sort in the DESCENDING order of density \rho
    ind_rho = np.argsort(-ar[:,0] )
    ar = ar[ind_rho]
    
    #Find the "area" (horizontal distance in 2D case) unoccupied by the keel
    #   at each vertical level
    Area = np.zeros((Nz,))
    for i in range(1,Nz):
        if z[i]<=h:
            # x(z) for the keel -> inverse of keel function given above
            if upstream_flag==1:
                Area[i] = -0.25*np.sqrt((h*sigma**2)/z[i] - sigma**2) + l
            else:
                Area[i] = x[Nx2] - (0.25*np.sqrt((h*sigma**2)/z[i] - sigma**2)+l)
        else:
            Area[i] = np.abs(x[Nx2] - x[Nx1])    
    Area = np.flip(Area)
    Volz = Area*dz
    
    #Now let the stacking of the cells begin!
    totvol = 0.0  #"volume" (area in 2D) tracker
    ind_a = 0 #vertical level tracker
    #vector for each cell -> to be filled with \delta z (vertical space) each 
    #   cell will occupy
    dzs = np.zeros((n,))
    
    for i in range(n):
        totvol += ar[i,1]
        if totvol < Volz[ind_a]:
            dzs[i] = ar[i,1]/Area[ind_a]
        else:
            if ind_a < Nz-1:
                ind_a +=1
                dzs[i] = ar[i,1]/Area[ind_a]
                totvol = 0.0
            else:
                dzs[i] = ar[i,1]/Area[ind_a]
     
    #Now we have for each element, stacked from the densest to the lightest,
    #   the vertical spacing (\delta z) that it occupies.
    
    #z* is just a cumulative sum of these vertical spacings, i.e., how far
    #   above the bottom is a particular element stacked
    z_star = -(np.cumsum(dzs)-H) #sign change etc needed because of the particular
            #coordinate system here
            
    #applying some spline interpolation so that the derivative dz*/drho is smoother
    spl2 = UnivariateSpline(np.flip(ar[:,0]),np.flip(z_star),k=5,s=1.8)
    dzsdb = spl2.derivative(1)(ar[:,0]) #this is dz*/drho
    #adjust for minor discrepancy of negative gradients 
    #   introduced by spline interpolation
    dzsdb[dzsdb<0] = np.NaN
    dzsdb = np.apply_along_axis(pad, 0, dzsdb)

    #Now put together in one array three column vectors:
    #       (z*, element number, dz*/drho) 
    ZS_tot = np.concatenate((np.reshape(z_star,(n,1)),
                         np.reshape(ar[:,2],(n,1)),
                         np.reshape(dzsdb,(n,1))),
                    axis=1)

    
    #Using the element number, find where dz*/drho belongs in physical (x,z) space.
    #This is the needed value dz*/drho evaluated at each local rho
    dzdb_final_col = np.nan*np.ones((num_elem,))
    dzdb_final_col[ZS_tot[:,1].astype(int)] = ZS_tot[:,2]
    dzdb_final= dzdb_final_col.reshape((Nx2-Nx1,Nz))
    
    return ar[:,0], z_star, dzdb_final #return column vectors of \rho and z*
                # and dz*/drho(x,z) matrix
                
def rho_deriv(rho,x,z):
    rho_z = np.gradient(rho,z,axis=1);
    rho_x = np.gradient(rho,x,axis=0);
    nabla_rho = (rho_z)**2 + (rho_x)**2
    return nabla_rho

def calc_mixing(rho,h,l,Nx,Nz,zv,Nx1,Nx2,dz,vol,upstream_flag,x,z):
    b, zs, dzdb = find_zstar(rho,h,l,Nx,Nz,zv,Nx1,Nx2,dz,vol,upstream_flag)
    nabla_rho = rho_deriv(rho,x,z)
    
    mixing = nabla_rho[Nx1:Nx2,:]*dzdb
    
    return b, zs,  mixing

#%% Calculate and plot mixing
#Calculate upstream
b_up, zs_up, mixing_up = calc_mixing(rho,h,l,Nx,Nz,zv,0,Nx_mid,dz,vol,1,x,z)

#Calculate downstream
b_down, zs_down, mixing_down = calc_mixing(rho,h,l,Nx,Nz,zv,Nx_mid,Nx-1,dz,vol,0,x,z)

#%% "Volume" (area) integrated mixing for each domain
keel_mask = find_mask(h, l, Nx, Nz, zv) #mask out the keel
mixing_up_ma = np.ma.array(mixing_up,mask=keel_mask[:Nx_mid,:])
mixing_dn_ma = np.ma.array(mixing_down,mask=keel_mask[Nx_mid:Nx-1,:])

tot_mix_up = np.sum(mixing_up_ma)*vol*g*mu #tot mixing upstream
tot_mix_dn = np.sum(mixing_dn_ma)*vol*g*mu #tot mixing downstream

rho_deriv_up = rho_deriv(rho,x,z)[:Nx_mid,:]
rho_deriv_down = rho_deriv(rho,x,z)[Nx_mid:Nx-1,:]


#Plot mixing and density gradients ((\del \rho)^2) both upstream and downstream 
# fig, axs = plt.subplots(2, 2, sharey=True)
# ax1 = axs[0,0]
# c = ax1.imshow(np.transpose(np.log10(mixing_up)), vmin=-0.5, vmax=0.5,
#                    cmap='hot_r', 
#                    extent=(0, l/z0, H/z0, 0))
# ax1.set_xlim(0, l/z0)
# ax1.set_ylim(H/(2*z0), 0)
# ax1.set_aspect('auto')
# ax1.fill_between(x/z0, 0, keel(h, l, x)/z0, facecolor="gray", zorder=10)
# cbar = fig.colorbar(c, ax=ax1, orientation='horizontal', pad=0.2)
# ax1.set_title('(a) $log_{10}(mixing_{up})$')

# ax2 = axs[0,1]
# c = ax2.imshow(np.transpose(np.log10(mixing_down)), vmin=-0.5, vmax=0.5,
#                    cmap='hot_r', 
#                    extent=(l/z0, L/z0, H/z0, 0))
# ax2.set_xlim(l/z0, L/z0)
# ax2.set_ylim(H/(2*z0), 0)
# ax2.set_aspect('auto')
# ax2.fill_between(x/z0, 0, keel(h, l, x)/z0, facecolor="gray", zorder=10)
# cbar = fig.colorbar(c, ax=ax2, orientation='horizontal', pad=0.2)
# ax2.set_title('(b) $log_{10}(mixing_{down})$');

# ax3 = axs[1,0]
# c = ax3.imshow(np.transpose(rho_deriv_up), vmin=-1, vmax=1,
#                    cmap='bwr', 
#                    extent=(0, l/z0, H/z0, 0))
# ax3.set_xlim(0, l/z0)
# ax3.set_ylim(H/(2*z0), 0)
# ax3.set_aspect('auto')
# ax3.fill_between(x/z0, 0, keel(h, l, x)/z0, facecolor="gray", zorder=10)
# cbar = fig.colorbar(c, ax=ax3, orientation='horizontal', pad=0.2)
# ax3.set_title('(c) $(\\nabla \\rho)^2$ upstream');

# ax4 = axs[1,1]
# c = ax4.imshow(np.transpose(rho_deriv_down), vmin=-1, vmax=1,
#                    cmap='bwr', 
#                    extent=(l/z0, L/z0, H/z0, 0))
# ax4.set_xlim(l/z0, L/z0)
# ax4.set_ylim(H/(2*z0), 0)
# ax4.set_aspect('auto')
# ax4.fill_between(x/z0, 0, keel(h, l, x)/z0, facecolor="gray", zorder=10)
# cbar = fig.colorbar(c, ax=ax4, orientation='horizontal', pad=0.2)
# ax4.set_title('(d) $(\\nabla \\rho)^2$ downstream');
# fig.set_size_inches(8, 6)

