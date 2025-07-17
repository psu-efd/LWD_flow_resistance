#plot the relationship between Cd and n using the equation Cd=2gLon^2/h^{4/3}

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import griddata, interp2d

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"


Cd_simulation = np.array([36.5, 34.4, 58.3, 53.4, 16.5, 14.1, 20.5, 25.6, 60, 56.1, 51.5, 52.5, 64.8, 53.4, 45.4, 40])
n_simulation = np.array([2.1, 2.2, 2.5, 2.6, 1.3, 1.17, 1.27, 1.6, 2.4, 2.36, 1.82, 1.97, 2.64, 2.38, 1.8, 1.7])

h_upstream = np.array([0.5, 0.51, 0.374, 0.377, 0.48, 0.48, 0.36, 0.36, 0.52, 0.54, 0.41, 0.44, 0.48, 0.48, 0.35, 0.36])

Lo = 0.2 #LWD streamwise length

Cd_computed = 2*9.81*Lo*n_simulation**2/h_upstream**(4/3)

#compute RMSE between Cd_simulation and Cd_computed
RMSE = np.sqrt(np.mean((Cd_simulation - Cd_computed)**2))
print("RMSE:", RMSE)

#plot the relationship between Cd and n
plt.figure(figsize=(6,6))
plt.scatter(Cd_simulation, Cd_computed, label='Simulation', marker='o', color='blue', s=150, facecolor='gray', edgecolor='blue')
plt.xlabel('$C_d$ (simulation)', fontsize=16)
plt.ylabel('$C_d = 2gL_o n^2/h^{4/3}$', fontsize=16)

#add a diagonal line
plt.plot([0, 80], [0, 80], color='black', linestyle='--', linewidth=2)

#add text to the plot
plt.text(20, 70, f'RMSE: {RMSE:.2f}', fontsize=14)

#set axis tick label size 
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

#equal aspect ratio
plt.gca().set_aspect('equal', adjustable='box')

#set axis limit
plt.xlim(0, 80)
plt.ylim(0, 80)

#plt.legend(loc='upper right',fontsize=14,frameon=True, facecolor='white')
#plt.grid(True)
plt.tight_layout()
plt.savefig('Cd_n_relationship.png', dpi=300, bbox_inches='tight', pad_inches=0)
#plt.show()
plt.close()


