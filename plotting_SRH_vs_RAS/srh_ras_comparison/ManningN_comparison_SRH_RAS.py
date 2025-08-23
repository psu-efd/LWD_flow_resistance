#plot the comparison of the calibrated Manning's N between SRH-2D and HEC-RAS 2D

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import griddata, interp2d

import pandas as pd

import os
import vtk
from vtk.util import numpy_support

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

def plot_comparison_ManningN_SRH_RAS_barchart(case_IDs, Frs, ManningN_SRH_2D, ManningN_HEC_RAS_2D, betas, alphas_exp, alphas_simulation_SRH_2D, alphas_simulation_HEC_RAS_2D):
    """
    Plots the comparison of the calibrated Manning's N between SRH-2D and HEC-RAS 2D using a barchart
    """    

    
    plt.figure(figsize=(10, 6))
    
    # Set bar width and positions for side-by-side bars
    bar_width = 0.35
    x_positions = [x - bar_width/2 for x in case_IDs]
    x_positions_shifted = [x + bar_width/2 for x in case_IDs]
    
    plt.bar(x_positions, ManningN_SRH_2D, width=bar_width, label='SRH-2D', color='blue')
    plt.bar(x_positions_shifted, ManningN_HEC_RAS_2D, width=bar_width, label='HEC-RAS 2D', color='red', hatch='//')
    
    plt.xlabel('Case ID', fontsize=24)
    plt.ylabel('Calibrated Manning\'s $n$', fontsize=24)

    #format the x-axis ticks fontsize
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    #set x and y limits
    plt.xlim(0, 17)
    plt.ylim(1, 4)

    plt.legend(fontsize=24, loc='upper right', frameon=False)
    plt.savefig('ManningN_comparison_SRH_RAS.png', dpi=300, bbox_inches='tight', transparent=False)

def plot_comparison_ManningN_SRH_RAS_scatter(case_IDs, Frs, ManningN_SRH_2D, ManningN_HEC_RAS_2D, betas, alphas_exp, alphas_simulation_SRH_2D, alphas_simulation_HEC_RAS_2D):
    """
    Plots the comparison of the calibrated Manning's N between SRH-2D and HEC-RAS 2D using a scatter plot: horizontal axis is Manning's N from SRH-2D, vertical axis is Manning's N from HEC-RAS 2D
    """
    
    plt.figure(figsize=(6, 6))  # Square figure for equal aspect ratio
    
    # Create scatter plot
    plt.scatter(ManningN_SRH_2D, ManningN_HEC_RAS_2D, marker='o', color='blue', s=120, facecolors='none')

    #compute the RMSE of the comparison
    RMSE = np.sqrt(np.mean((np.array(ManningN_SRH_2D) - np.array(ManningN_HEC_RAS_2D))**2))
    print(f'RMSE of the comparison: {RMSE}')

    #add text with RMSE error (positioned relative to data range)
    x_pos = 0.05 * (3 - 1) + 1
    y_pos = 0.95 * (3 - 1) + 1
    plt.text(x_pos, y_pos, f'RMSE = {RMSE:.4f}', fontsize=18)
    
    # Add diagonal line (y = x)
    min_val = 1.0
    max_val = 3.0
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--')
    
    # Set equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Add labels and title
    plt.xlabel('Manning\'s $n$ from SRH-2D', fontsize=16)
    plt.ylabel('Manning\'s $n$ from HEC-RAS 2D', fontsize=16)
    #plt.title('Comparison of Calibrated Manning\'s $n$ Values', fontsize=18, pad=20)

    #set x and y ticks fontsize
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Add legend
    #plt.legend(fontsize=14, loc='upper left')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Set axis limits with some padding
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    # Add case ID annotations
    # for i, case_id in enumerate(case_IDs):
    #     plt.annotate(f'Case {case_id}', 
    #                 (ManningN_SRH_2D[i], ManningN_HEC_RAS_2D[i]),
    #                  xytext=(5, 5), textcoords='offset points',
    #                 fontsize=10, alpha=0.8)
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig('ManningN_comparison_SRH_RAS_scatter.png', dpi=300, bbox_inches='tight', transparent=False)

    #plt.show()

    
    

if __name__ == "__main__":
    #define data 
    case_IDs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    #case_IDs = [1]

    Frs = [0.070, 0.097, 0.109, 0.137, 0.077, 0.097, 0.127,	0.153, 0.064, 0.077, 0.090, 0.099, 0.076, 0.097, 0.126, 0.163]
    
    ManningN_SRH_2D = [2.13, 2.16, 2.66, 2.72, 1.26, 1.17, 1.26, 1.5, 2.42, 2.36, 1.92, 1.98, 2.63,2.37,1.78,1.62]
    ManningN_HEC_RAS_2D = [2.24, 2.24, 2.66, 2.72, 1.31, 1.18, 1.35, 1.49, 2.56, 2.42, 1.91, 1.93, 2.84,2.34,1.98,1.84]
    
    betas = [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5]
    alphas_exp = [1, 1, 1, 1, 0.779, 0.775, 0.824, 0.832, 1, 1, 1, 1, 0.868, 0.857, 0.846, 0.836]
    alphas_simulation_SRH_2D = [1, 1, 1, 1, 0.779, 0.777, 0.823, 0.831, 1, 1, 1, 1, 0.868, 0.856, 0.846, 0.837]
    alphas_simulation_HEC_RAS_2D = [1, 1, 1, 1, 0.777, 0.7746, 0.82457, 0.8302, 1, 1, 1, 1, 0.869, 0.857, 0.8486, 0.8364]

    plot_comparison_ManningN_SRH_RAS_barchart(case_IDs, Frs, ManningN_SRH_2D, ManningN_HEC_RAS_2D, betas, alphas_exp, alphas_simulation_SRH_2D, alphas_simulation_HEC_RAS_2D)

    plot_comparison_ManningN_SRH_RAS_scatter(case_IDs, Frs, ManningN_SRH_2D, ManningN_HEC_RAS_2D, betas, alphas_exp, alphas_simulation_SRH_2D, alphas_simulation_HEC_RAS_2D)

    print("Done")