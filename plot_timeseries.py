import numpy as np
import matplotlib.pyplot as plt

#Import values
import json
with open('K_values_160-600.txt') as f:
    Kdown_json = json.load(f)
    
with open('K_values_600-920.txt') as f:
    Kup_json = json.load(f)
    
with open('phi_d_values_160-600.txt') as f:
    Phidown_json = json.load(f)
    
with open('phi_d_values_600-920.txt') as f:
    Phiup_json = json.load(f)    

#Get simulation names
sim_names = list(Kdown_json.keys())

#Separate simulations into regimes
#Upstream
unstab_super = ['F20H05','F20H09','F20H12','F20H20']
unstab_sub = ['F15H05','F15H09','F15H12','F15H20','F10H09','F10H12','F10H20']
stab_sub = ['F10H05','F05H05','F05H09','F05H12','F05H20']

#Downstream
vortex = ['F20H05','F20H09','F20H12','F20H20','F15H09','F15H12','F15H20']
laminar = ['F15H05','F10H05','F10H09','F10H12','F10H20','F05H20']
lee_waves = ['F05H05','F05H09','F05H12']


#PLOT
fig = plt.figure()
ax1 = fig.add_subplot(231)
for name in unstab_super:
    ax1.plot(Phiup_json[name][1],Phiup_json[name][0],label=name)
ax1.legend()
ax1.set_yscale('log')
ax1.set_ylabel('$\\Phi_{upstream} $')
ax1.set_title('Unstable Supercritical - upstream')

ax2 = fig.add_subplot(232, sharey = ax1)
for name in unstab_sub:
    ax2.plot(Phiup_json[name][1],np.array(Phiup_json[name][0]),label=name)
ax2.legend()
ax2.set_yscale('log')
ax2.set_title('Unstable Subcritical - upstream')

ax3 = fig.add_subplot(233, sharey = ax1)
for name in stab_sub:
    ax3.plot(Phiup_json[name][1],np.array(Phiup_json[name][0]),label=name)
ax3.legend()
ax3.set_yscale('log')
ax3.set_title('Stable Subcritical - upstream')


ax4 = fig.add_subplot(234, sharey = ax1)
for name in vortex:
    ax4.plot(Phidown_json[name][1],np.array(Phidown_json[name][0]),label=name)
ax4.legend()
ax4.set_yscale('log')
ax4.set_xlabel('time')
ax4.set_ylabel('$\\Phi_{downstream}$')
ax4.set_title('Vortex Shedding - downstream')

ax5 = fig.add_subplot(235, sharey = ax1)
for name in laminar:
    ax5.plot(Phidown_json[name][1],np.array(Phidown_json[name][0]),label=name)
ax5.legend()
ax5.set_yscale('log')
ax5.set_xlabel('time')
ax5.set_title('Fast Laminar - downstream')

ax6 = fig.add_subplot(236, sharey = ax1)
for name in lee_waves:
    ax6.plot(Phidown_json[name][1],np.array(Phidown_json[name][0]),label=name)
ax6.legend()
ax6.set_yscale('log')
ax6.set_xlabel('time')
ax6.set_title('Lee Waves - downstream')

fig.set_size_inches(12, 8)
fig.savefig('Phi_timeseries.pdf')