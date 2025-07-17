"""
Plot the calibration results collectively (data collected from all cases).
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

    # List of directories
    directories = ['Exp_1_Cd',   #full
                   'Exp_2_Cd',
                   'Exp_3_Cd',
                   'Exp_4_Cd',
                   'Exp_5_Cd',   #half
                   'Exp_6_Cd',
                   'Exp_7_Cd',
                   'Exp_8_Cd',
                   'Exp_9_Cd',   #full
                   'Exp_10_Cd',
                   'Exp_11_Cd',
                   'Exp_12_Cd',
                   'Exp_13_Cd',  #half
                   'Exp_14_Cd',
                   'Exp_15_Cd',
                   'Exp_16_Cd'
                   ]

    # array for all experiment results: WSE_upstream WSE_downstream Flow_Fraction_open
    experiment_results = [
        [0.78, 0.75, 0], #case 1
        [0.79, 0.74, 0], #case 2
        [0.654, 0.603, 0], #case 3
        [0.657, 0.584, 0], #case 4
        [0.76, 0.76, 0.779], #case 5
        [0.76, 0.76, 0.775], #case 6
        [0.64, 0.64, 0.824], #case 7
        [0.64, 0.64, 0.832], #case 8
        [0.52, 0.44, 0],    #case 9
        [0.54, 0.42, 0],    #case 10
        [0.41, 0.29, 0],    #case 11
        [0.44, 0.26, 0],    #case 12
        [0.48, 0.48, 0.868], #case 13
        [0.48, 0.48, 0.857], #case 14
        [0.35, 0.345, 0.846], #case 15
        [0.36, 0.34, 0.836] #case 16
    ]

    #convert to numpy array
    experiment_results = np.array(experiment_results)

    # calibration results for each case: Cd error_total error_H error_Q_split H_p1 H_p2 H_p3 Q_Line2 Q_Line3
    calibration_results = [
        [36.524328776157226, 0.0003964400216361681, 0.0003964400216361681, 0.0, 0.481249331, 0.750057686, 0.493157526, 0.0582201199, 0.0564998801],  #case 1
        [34.36480375124306, 0.00048675487077511434, 0.00048675487077511434, 0.0, 0.293053717, 0.740120546, 0.503162899, 0.0841519335, 0.0817680665], #case 2
        [58.68846793458072, 0.0008871945223924589, 0.0008871945223924589, 0.0, 0.216940087, 0.603088663, 0.367271638, 0.0565081445, 0.0566243555],   #case 3
        [53.44007604862924, 0.0018839035634951587, 0.0018839035634951587, 0.0, 0.23576547, 0.584150461, 0.369398282, 0.0731910311, 0.0732464689],    #case 4
        [16.49099847773346, 0.0075656913586554944, 0.007293441418159589, 0.00027224994049590556, 0.486532972, 0.758815983, 0.475712903, 0.0920086772, 0.0260613262],  #case 5
        [14.124932076365171, 0.008068960121771692, 0.0058226818459997404, 0.0022462782757719513, 0.533059581, 0.75917493, 0.475240631, 0.114799261, 0.0329007206],    #case 6
        [20.539990528570144, 0.0073278077043102885, 0.00648041985658642, 0.0008473878477238683, 0.518019322, 0.639342522, 0.354924948, 0.0982226554, 0.0211023081],   #case 7
        [25.606818277119103, 0.008629953316946852, 0.007838534259914902, 0.0007914190570319501, 0.567575089, 0.63988762, 0.355705018, 0.119694064, 0.0243059701],     #case 8
        [59.97663666472649, 0.0015141851398601759, 0.0015141851398601759, 0.0, 0.440438436, 0.440260817, 0.520479138, 0.0571, 0.0571],        #case 9
        [56.13874423451235, 0.002071836243386375, 0.002071836243386375, 0.0, 0.420795532, 0.420473622, 0.539490151, 0.07205, 0.07205],        #case 10
        [52.46495499708974, 0.005002712699747735, 0.005002712699747735, 0.0, 0.291457676, 0.29086987, 0.410821296, 0.055, 0.055],             #case 11
        [52.46495499708974, 0.009413106993007068, 0.009413106993007068, 0.0, 0.26316125, 0.261896534, 0.440932248, 0.068, 0.068],             #case 12
        [64.84816980807996, 0.011519132226981773, 0.011474964583333322, 4.4167643648451715e-05, 0.478303513, 0.479184616, 0.484692599, 0.104154707, 0.0158453012],  #case 13
        [53.44766113820366, 0.018336271392813047, 0.017642516666666747, 0.0006937547261463006, 0.477507892, 0.478755956, 0.487224364, 0.129302251, 0.0216977583], #case 14
        [45.427198377248025, 0.017770567046462467, 0.0173917548654245, 0.00037881218103796854, 0.342641734, 0.343120012, 0.35417988, 0.101565341, 0.0184345214], #case 15
        [39.95327332945298, 0.02362319695089455, 0.022671023529411756, 0.0009521734214827937, 0.336444981, 0.337109329, 0.354899142, 0.127216994, 0.024783321] #case 16
    ]

    #convert to numpy array
    calibration_results = np.array(calibration_results)

    # Bed elevation at upstream and downstream monitoring points
    bed_elevation_upstream = np.array([0.287, 0.287, 0.287, 0.287, 0.287, 0.287, 0.287, 0.287, 0, 0, 0, 0, 0, 0, 0, 0])
    bed_elevation_downstream = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    #compute upstream and downstream water depth at monitoring points 
    water_depth_upstream_exp = experiment_results[:, 0] - bed_elevation_upstream
    water_depth_downstream_exp = experiment_results[:, 1] - bed_elevation_downstream

    water_depth_upstream_simulation = calibration_results[:, 6] 
    water_depth_downstream_simulation = calibration_results[:, 5] 

    print("measured upstream water depth", water_depth_upstream_exp)
    print("measured downstream water depth", water_depth_downstream_exp)

    print("simulated upstream water depth", water_depth_upstream_simulation)
    print("simulated downstream water depth", water_depth_downstream_simulation)

    #compute RMSE error between simulation and measurement
    error_upstream = np.abs(water_depth_upstream_simulation - water_depth_upstream_exp)
    rmse_upstream = np.sqrt(np.mean(error_upstream**2))
    print("RMSE error between simulation and measurement for upstream water depth: ", rmse_upstream)

    error_downstream = np.abs(water_depth_downstream_simulation - water_depth_downstream_exp)
    rmse_downstream = np.sqrt(np.mean(error_downstream**2))
    print("RMSE error between simulation and measurement for downstream water depth: ", rmse_downstream)

    # Plot the comparison of upstream water depth between measurement and simulation
    plt.figure(figsize=(10, 6))
    plt.scatter(water_depth_upstream_simulation, water_depth_upstream_exp, label='Simulation', marker='o', color='blue', s=120, facecolors='none')       #no fill 
    plt.xlabel('Simulated Upstream Water Depth (m)', fontsize=24)
    plt.ylabel('Measured Upstream Water Depth (m)', fontsize=24)
    #plt.title('Comparison of Upstream Water Depth between Measurement and Simulation')

    #set font size for x and y ticks 
    plt.tick_params(axis='both', which='major', labelsize=24)

    #add diagonal line
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')

    #set x and y limits
    plt.xlim(0.3, 0.6)
    plt.ylim(0.3, 0.6)

    #add text with RMSE error (positioned relative to data range)
    x_pos = 0.1 * (0.6 - 0.3) + 0.3
    y_pos = 0.75 * (0.6 - 0.3) + 0.3
    plt.text(x_pos, y_pos, f'RMSE = {rmse_upstream:.4f} m', fontsize=24)

    #equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    #plt.legend()  
    plt.savefig('calibration_results_Cd_upstream_water_depth.png', dpi=300, bbox_inches='tight', transparent=False)
    #plt.show()

    # Plot the comparison of downstream water depth between measurement and simulation
    plt.figure(figsize=(10, 6))
    plt.scatter(water_depth_downstream_simulation, water_depth_downstream_exp, label='Simulation', marker='o', color='blue', s=120, facecolors='none')       #no fill 
    plt.xlabel('Simulated Downstream Water Depth (m)', fontsize=24)
    plt.ylabel('Measured Downstream Water Depth (m)', fontsize=24)
    #plt.title('Comparison of Downstream Water Depth between Measurement and Simulation')

    #set font size for x and y ticks 
    plt.tick_params(axis='both', which='major', labelsize=24)

    #add diagonal line
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')

    #set x and y limits
    plt.xlim(0.2, 0.8)
    plt.ylim(0.2, 0.8)

    #add text with RMSE error (positioned relative to data range)
    x_pos = 0.1 * (0.8 - 0.2) + 0.2
    y_pos = 0.75 * (0.8 - 0.2) + 0.2
    plt.text(x_pos, y_pos, f'RMSE = {rmse_downstream:.4f} m', fontsize=24)

    #equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    #plt.legend()  
    plt.savefig('calibration_results_Cd_downstream_water_depth.png', dpi=300, bbox_inches='tight', transparent=False)
    #plt.show()

    #compute the flow partition 
    flow_partition_exp = experiment_results[:, 2]
    flow_partition_simulation = calibration_results[:, 7]/(calibration_results[:, 7] + calibration_results[:, 8])
    
    # Extract only data at indices 4 to 7 and 12 to 15 (half blockage cases; python index starts from 0)
    selected_indices = [4, 5, 6, 7, 12, 13, 14, 15]
    flow_partition_exp_selected = flow_partition_exp[selected_indices]
    flow_partition_simulation_selected = flow_partition_simulation[selected_indices]

    print("flow_partition_exp_selected: ", flow_partition_exp_selected)
    print("flow_partition_simulation_selected: ", flow_partition_simulation_selected)

    #computer RMSE error between simulation and measurement for flow partition
    error_flow_partition = np.abs(flow_partition_simulation_selected - flow_partition_exp_selected)
    rmse_flow_partition = np.sqrt(np.mean(error_flow_partition**2))
    print("RMSE error between simulation and measurement for flow partition: ", rmse_flow_partition)

    # Plot the comparison of flow partition between measurement and simulation
    plt.figure(figsize=(10, 6)) 
    plt.scatter(flow_partition_simulation_selected, flow_partition_exp_selected, label='Simulation', marker='o', color='blue', s=120, facecolors='none')       #no fill 
    plt.xlabel('Simulated Flow Partition $\\alpha$', fontsize=24)
    plt.ylabel('Measured Flow Partition $\\alpha$', fontsize=24)
    #plt.title('Comparison of Flow Partition between Measurement and Simulation')

    #set font size for x and y ticks 
    plt.tick_params(axis='both', which='major', labelsize=24)

    #add diagonal line
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')

    #set x and y limits
    plt.xlim(0.7, 1)
    plt.ylim(0.7, 1)

    #add text with RMSE error (positioned relative to data range)
    x_pos = 0.1 * (1 - 0.7) + 0.7
    y_pos = 0.75 * (1 - 0.7) + 0.7
    plt.text(x_pos, y_pos, f'RMSE = {rmse_flow_partition:.4f}', fontsize=24)

    #equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    #plt.legend()
    plt.savefig('calibration_results_Cd_flow_partition.png', dpi=300, bbox_inches='tight', transparent=False)
    #plt.show()

    print("All done!")