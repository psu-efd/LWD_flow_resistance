"""
Plot the effect of turbulence model on the calibration results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import griddata, interp2d

import pandas as pd

import os


plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

if __name__ == "__main__":

    ManningN_SRH_2D_w_turb = np.array([2.13, 2.16, 2.66, 2.72, 1.26, 1.17, 1.26, 1.5, 2.42, 2.36, 1.92, 1.98, 2.63, 2.37, 1.78, 1.62])
    ManningN_SRH_2D_wo_turb = np.array([2.25, 2.36, 2.73, 2.83, 1.3, 1.16, 1.24, 1.44, 2.55, 2.5, 1.95, 2, 2.49, 2.33, 1.82, 1.9])

    ManningN_HEC_RAS_2D_w_turb = np.array([2.24, 2.24, 2.66, 2.72, 1.31, 1.18, 1.35, 1.49, 2.56, 2.42, 1.91, 1.93, 2.84, 2.34, 1.98, 1.84])
    ManningN_HEC_RAS_2D_wo_turb = np.array([2.25, 2.25, 2.66, 2.72, 1.33, 1.18, 1.29, 1.48, 2.46, 2.42, 1.92, 1.96, 2.84, 2.54, 1.92, 1.79])

    Cd_SRH_2D_w_turb = np.array([36.4, 34.2, 58.6, 53.4, 16.4, 14, 20.3, 25.4, 58.9, 56, 51.2, 51.3, 64.2, 53.2, 43.8, 38.1])
    Cd_SRH_2D_wo_turb = np.array([37.1, 35, 60.4, 55, 15.5, 11.5, 17.5, 23, 61.1, 57.5, 52.5, 52.5, 55.8, 49.1, 40, 34.9])

    Cd_HEC_RAS_2D_w_turb = np.array([31.24, 31.68, 49.08, 48.36, 15.6, 12, 18.48, 22.68, 56, 52, 46.24, 47.28, 59.8, 49.84, 41.6, 35])
    Cd_HEC_RAS_2D_wo_turb = np.array([31.28, 31.72, 49.16, 48.44, 14.16, 13.96, 17, 23.68, 58.52, 52.04, 46.24, 47.24, 59.8, 49.84, 42.96, 37.96])

    #compute the RMSE error between with and without turbulence model
    rmse_ManningN_SRH_2D = np.sqrt(np.mean((ManningN_SRH_2D_w_turb - ManningN_SRH_2D_wo_turb)**2))
    rmse_ManningN_HEC_RAS_2D = np.sqrt(np.mean((ManningN_HEC_RAS_2D_w_turb - ManningN_HEC_RAS_2D_wo_turb)**2))
    rmse_Cd_SRH_2D = np.sqrt(np.mean((Cd_SRH_2D_w_turb - Cd_SRH_2D_wo_turb)**2))
    rmse_Cd_HEC_RAS_2D = np.sqrt(np.mean((Cd_HEC_RAS_2D_w_turb - Cd_HEC_RAS_2D_wo_turb)**2))

    print("RMSE error between SRH-2D with and without turbulence model for Manning's n: ", rmse_ManningN_SRH_2D)
    print("RMSE error between HEC-RAS 2D with and without turbulence model for Manning's n: ", rmse_ManningN_HEC_RAS_2D)
    print("RMSE error between SRH-2D with and without turbulence model for Cd: ", rmse_Cd_SRH_2D)
    print("RMSE error between HEC-RAS 2D with and without turbulence model for Cd: ", rmse_Cd_HEC_RAS_2D)

    #plot the comparison of Manning's n between with and without turbulence model
    plt.figure(figsize=(10, 6))
    plt.scatter(ManningN_SRH_2D_w_turb, ManningN_SRH_2D_wo_turb, label='SRH-2D', marker='o', color='blue', s=120, facecolors='none')       #no fill 
    plt.scatter(ManningN_HEC_RAS_2D_w_turb, ManningN_HEC_RAS_2D_wo_turb, label='HEC-RAS 2D', marker='^', color='red', s=120, facecolors='none')       #no fill 
    plt.xlabel('Manning\'s $n$ (w/ turbulence model)', fontsize=24)
    plt.ylabel('Manning\'s $n$ (wo/ turbulence model)', fontsize=24)

    #set font size and number of decimal places for x and y ticks 
    plt.tick_params(axis='both', which='major', labelsize=24)
    # Format tick labels to show one decimal place
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    #add diagonal line
    plt.plot([1, 4], [1, 4], color='black', linestyle='--')

    #set x and y limits
    plt.xlim(1, 4)
    plt.ylim(1, 4)

    #add text with RMSE error (positioned relative to data range)
    x_pos = 0.05 * (4 - 1) + 1
    y_pos = 0.95 * (4 - 1) + 1
    plt.text(x_pos, y_pos, f'RMSE (SRH-2D) = {rmse_ManningN_SRH_2D:.2f}', fontsize=18)
    y_pos = y_pos - 0.25
    plt.text(x_pos, y_pos, f'RMSE (HEC-RAS 2D) = {rmse_ManningN_HEC_RAS_2D:.2f}', fontsize=18)

    #equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    plt.legend(fontsize=24, loc='lower right', frameon=False)  
    plt.savefig('effect_of_turbulence_model_on_ManningN.png', dpi=300, bbox_inches='tight', transparent=False)
    #plt.show()

    #plot the comparison of Cd between with and without turbulence model
    plt.figure(figsize=(10, 6))
    plt.scatter(Cd_SRH_2D_w_turb, Cd_SRH_2D_wo_turb, label='SRH-2D', marker='o', color='blue', s=120, facecolors='none')       #no fill 
    plt.scatter(Cd_HEC_RAS_2D_w_turb, Cd_HEC_RAS_2D_wo_turb, label='HEC-RAS 2D', marker='^', color='red', s=120, facecolors='none')       #no fill 
    plt.xlabel('$C_d$ (w/ turbulence model)', fontsize=24)
    plt.ylabel('$C_d$ (wo/ turbulence model)', fontsize=24)

    #set font size and number of decimal places for x and y ticks 
    plt.tick_params(axis='both', which='major', labelsize=24)
    # Format tick labels to show one decimal place
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    #add diagonal line
    plt.plot([10, 80], [10, 80], color='black', linestyle='--')

    #set x and y limits
    plt.xlim(10, 80)
    plt.ylim(10, 80)

    #add text with RMSE error (positioned relative to data range)
    x_pos = 0.05 * (80 - 10) + 10
    y_pos = 0.95 * (80 - 10) + 10
    plt.text(x_pos, y_pos, f'RMSE (SRH-2D) = {rmse_Cd_SRH_2D:.2f}', fontsize=18)
    y_pos = y_pos - 5
    plt.text(x_pos, y_pos, f'RMSE (HEC-RAS 2D) = {rmse_Cd_HEC_RAS_2D:.2f}', fontsize=18)

    #equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    plt.legend(fontsize=24, loc='lower right', frameon=False)  
    plt.savefig('effect_of_turbulence_model_on_Cd.png', dpi=300, bbox_inches='tight', transparent=False)
    #plt.show()

    print("All done!")