import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import griddata, interp2d

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

Fr = np.array([0.070, 0.097, 0.109, 0.137, 0.077, 0.097, 0.127, 0.153, 0.064, 0.077, 0.090, 0.099, 0.076, 0.097, 0.123, 0.149])
ManningN_all = np.array([2.1,2.2,2.5,2.6,1.3,1.17,1.27,1.6,2.4,2.36,1.82,1.97,2.64,2.38,1.8,1.7])

#case IDs with sediment
case_IDs_with_sediment = [1, 2, 3, 4, 5, 6, 7, 8]
case_IDs_without_sediment = [9, 10, 11, 12, 13, 14, 15, 16]

case_IDs_full = [1, 2, 3, 4, 9, 10, 11, 12]
case_IDs_half = [5, 6, 7, 8, 13, 14, 15, 16]

#case IDs for full LWD and with sediment
case_IDs_full_with_sediment = list(set(case_IDs_full) & set(case_IDs_with_sediment))

#case IDs for full LWD and without sediment
case_IDs_full_without_sediment = list(set(case_IDs_full) & set(case_IDs_without_sediment))

#case IDs for half LWD and with sediment
case_IDs_half_with_sediment = list(set(case_IDs_half) & set(case_IDs_with_sediment))

#case IDs for half LWD and without sediment
case_IDs_half_without_sediment = list(set(case_IDs_half) & set(case_IDs_without_sediment))

print("case IDs full and with sediment:", case_IDs_full_with_sediment)
print("case IDs full and without sediment:", case_IDs_full_without_sediment)
print("case IDs half and with sediment:", case_IDs_half_with_sediment)
print("case IDs half and without sediment:", case_IDs_half_without_sediment)

#get Cd values (subtract 1 from case IDs since arrays are 0-indexed)
if case_IDs_full_with_sediment:
    n_full_with_sediment = ManningN_all[np.array(case_IDs_full_with_sediment) - 1]
    Fr_full_with_sediment = Fr[np.array(case_IDs_full_with_sediment) - 1]
else:
    n_full_with_sediment = np.array([])
    Fr_full_with_sediment = np.array([])

if case_IDs_full_without_sediment:
    n_full_without_sediment = ManningN_all[np.array(case_IDs_full_without_sediment) - 1]
    Fr_full_without_sediment = Fr[np.array(case_IDs_full_without_sediment) - 1]
else:
    n_full_without_sediment = np.array([])
    Fr_full_without_sediment = np.array([])

if case_IDs_half_with_sediment:
    n_half_with_sediment = ManningN_all[np.array(case_IDs_half_with_sediment) - 1]
    Fr_half_with_sediment = Fr[np.array(case_IDs_half_with_sediment) - 1]
else:
    n_half_with_sediment = np.array([])
    Fr_half_with_sediment = np.array([])

if case_IDs_half_without_sediment:
    n_half_without_sediment = ManningN_all[np.array(case_IDs_half_without_sediment) - 1]
    Fr_half_without_sediment = Fr[np.array(case_IDs_half_without_sediment) - 1]
else:
    n_half_without_sediment = np.array([])
    Fr_half_without_sediment = np.array([])


#Plot 1: all data points
plt.figure(figsize=(8,6))

# Plot all four cases
plt.scatter(Fr_full_without_sediment, n_full_without_sediment, label='Full-span, no sediment', marker='o', color='blue', s=200, facecolor='gray', edgecolor='blue')
plt.scatter(Fr_full_with_sediment, n_full_with_sediment, label='Full-span, with sediment', marker='o', color='blue', s=200, facecolor='gold', edgecolor='blue')
plt.scatter(Fr_half_without_sediment, n_half_without_sediment, label='Half-span, no sediment', marker='^', color='green', s=200, facecolor='gray', edgecolor='green')
plt.scatter(Fr_half_with_sediment, n_half_with_sediment, label='Half-span, with sediment', marker='^', color='green', s=200, facecolor='orange', edgecolor='green')

plt.xlabel('Froude Number ($Fr$)', fontsize=32)
plt.ylabel('Manning\' $n$', fontsize=32)
#plt.title('Effect of $Fr$, Blockage, and Sediment Bed on $C_d$')

#set axis tick label size 
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

#set axis limit
plt.xlim(0.06, 0.18)
plt.ylim(1, 3)

plt.legend(loc='upper center', bbox_to_anchor=(0.45, 1.35), ncol=2, fontsize=20, frameon=False, facecolor='white', columnspacing=0.5, handletextpad=0.5)
plt.grid(True)
plt.tight_layout()
plt.savefig('ManningN_Fr_dependence_all_cases.png', dpi=300, bbox_inches='tight', pad_inches=0)
#plt.show()
plt.close()

#Plot 2: full span cases
plt.figure(figsize=(8,6))

# Plot all four cases
plt.scatter(Fr_full_without_sediment, n_full_without_sediment, label='Full-span, no sediment', marker='o', color='blue', s=200, facecolor='gray', edgecolor='blue')
plt.scatter(Fr_full_with_sediment, n_full_with_sediment, label='Full-span, with sediment', marker='o', color='blue', s=200, facecolor='gold', edgecolor='blue')

plt.xlabel('Froude Number ($Fr$)', fontsize=32)
plt.ylabel('Manning\' $n$', fontsize=32)
#plt.title('Effect of $Fr$, Blockage, and Sediment Bed on $C_d$')

#set axis tick label size 
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

#set axis limit
plt.xlim(0.06, 0.18)
plt.ylim(1, 3)

plt.legend(loc='upper center', bbox_to_anchor=(0.45, 1.2), ncol=2, fontsize=20, frameon=False, facecolor='white', columnspacing=0.5, handletextpad=0.5)
plt.grid(True)
plt.tight_layout()
plt.savefig('ManningN_Fr_dependence_full_span_cases.png', dpi=300, bbox_inches='tight', pad_inches=0)
#plt.show()
plt.close()

#Plot 3: half span cases
plt.figure(figsize=(8,6))

# Plot all four cases
plt.scatter(Fr_half_without_sediment, n_half_without_sediment, label='Half-span, no sediment', marker='^', color='green', s=200, facecolor='gray', edgecolor='green')
plt.scatter(Fr_half_with_sediment, n_half_with_sediment, label='Half-span, with sediment', marker='^', color='green', s=200, facecolor='orange', edgecolor='green')

plt.xlabel('Froude Number ($Fr$)', fontsize=32)
plt.ylabel('Manning\' $n$', fontsize=32)
#plt.title('Effect of $Fr$, Blockage, and Sediment Bed on $C_d$')

#set axis tick label size 
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

#set axis limit
plt.xlim(0.06, 0.18)
plt.ylim(1, 3)

plt.legend(loc='upper center', bbox_to_anchor=(0.45, 1.2), ncol=2, fontsize=20, frameon=False, facecolor='white', columnspacing=0.5, handletextpad=0.5)
plt.grid(True)
plt.tight_layout()
plt.savefig('ManningN_Fr_dependence_half_span_cases.png', dpi=300, bbox_inches='tight', pad_inches=0)
#plt.show()
plt.close()

#Plot 4: no sediment cases
plt.figure(figsize=(8,6))

# Plot all four cases
plt.scatter(Fr_full_without_sediment, n_full_without_sediment, label='Full-span, no sediment', marker='o', color='blue', s=200, facecolor='gray', edgecolor='blue')
plt.scatter(Fr_half_without_sediment, n_half_without_sediment, label='Half-span, no sediment', marker='^', color='green', s=200, facecolor='gray', edgecolor='green')

plt.xlabel('Froude Number ($Fr$)', fontsize=32)
plt.ylabel('Manning\' $n$', fontsize=32)
#plt.title('Effect of $Fr$, Blockage, and Sediment Bed on $C_d$')

#set axis tick label size 
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

#set axis limit
plt.xlim(0.06, 0.18)
plt.ylim(1, 3)

plt.legend(loc='upper center', bbox_to_anchor=(0.45, 1.2), ncol=2, fontsize=20, frameon=False, facecolor='white', columnspacing=0.5, handletextpad=0.5)
plt.grid(True)
plt.tight_layout()
plt.savefig('ManningN_Fr_dependence_no_sediment_cases.png', dpi=300, bbox_inches='tight', pad_inches=0)
#plt.show()
plt.close()

#Plot 5: with sediment cases
plt.figure(figsize=(8,6))

# Plot all four cases
plt.scatter(Fr_full_with_sediment, n_full_with_sediment, label='Full-span, with sediment', marker='o', color='blue', s=200, facecolor='gold', edgecolor='blue')
plt.scatter(Fr_half_with_sediment, n_half_with_sediment, label='Half-span, with sediment', marker='^', color='green', s=200, facecolor='orange', edgecolor='green')

plt.xlabel('Froude Number ($Fr$)', fontsize=32)
plt.ylabel('Manning\' $n$', fontsize=32)
#plt.title('Effect of $Fr$, Blockage, and Sediment Bed on $C_d$')

#set axis tick label size 
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

#set axis limit
plt.xlim(0.06, 0.18)
plt.ylim(1, 3)

plt.legend(loc='upper center', bbox_to_anchor=(0.45, 1.2), ncol=2, fontsize=20, frameon=False, facecolor='white', columnspacing=0.5, handletextpad=0.5)
plt.grid(True)
plt.tight_layout()
plt.savefig('ManningN_Fr_dependence_with_sediment_cases.png', dpi=300, bbox_inches='tight', pad_inches=0)
#plt.show()
plt.close()





