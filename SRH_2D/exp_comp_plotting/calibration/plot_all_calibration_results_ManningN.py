"""
Plot the calibration results collectively (data collected from all cases). This is for the case of calibration with Manning's n.
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
        [2.091772483488109, 0.002228198495449543, 0.002228198495449543, 0.0, 0.481211367, 0.750059407, 0.493294236, 0.0593112823, 0.0576887208],  #case 1
        [2.246495499708974, 0.021877097566297788, 0.021877097566297788, 0.0, 0.292950723, 0.740119093, 0.505995424, 0.0840086079, 0.081991393], #case 2
        [2.523584689751015, 0.012562036412868646, 0.012562036412868646, 0.0, 0.216715105, 0.603085668, 0.367143201, 0.0580891947, 0.0589108053],   #case 3
        [2.613390762836266, 0.006686241889839171, 0.006686241889839171, 0.0, 0.23565546, 0.584150839, 0.369996574, 0.0744334109, 0.0745665892],    #case 4
        [1.3002815054659593, 0.009522808356247569, 0.008457062460670172, 0.0010657458955773969, 0.48649853, 0.75875452, 0.475900933, 0.0943878916, 0.0266120268],    #case 5
        [1.173799093774194, 0.007768597161093189, 0.0060925239234448985, 0.00167607323764829, 0.533008439, 0.7591759, 0.47536887, 0.116771895, 0.0342280818],    #case 6
        [1.2748189423007321, 0.009204563522625574, 0.007716651363314362, 0.001487912159311211, 0.517921781, 0.639279788, 0.355326736, 0.104459085, 0.0225409756],   #case 7
        [1.5991330021177168, 0.012490290991780006, 0.00978305926434131, 0.002707231727438697, 0.567408441, 0.639862547, 0.356377606, 0.129379519, 0.0256203589],    #case 8
        [2.4181678420159063, 0.0011238984265734332, 0.0011238984265734332, 0.0, 0.440423523, 0.440255119, 0.5200839, 0.0565, 0.0565],        #case 9
        [2.361980842295816, 0.002148749735449864, 0.002148749735449864, 0.0, 0.420782988, 0.420472214, 0.539846374, 0.072, 0.072],        #case 10
        [1.8220883364044105, 0.011890664676198554, 0.011890664676198554, 0.0, 0.291473397, 0.290884973, 0.407207906, 0.0555, 0.0555],             #case 11
        [1.9674218336379679, 0.015187288636363474, 0.015187288636363474, 0.0, 0.263139123, 0.261894033, 0.441370045, 0.068, 0.068],             #case 12
        [2.6421534604321333, 0.014038886796065802, 0.013706139583333337, 0.0003327472127324649, 0.47830952, 0.479120936, 0.484888467, 0.10333157, 0.015668399],  #case 13
        [2.3769970251709998, 0.021958666007085677, 0.021077656249999962, 0.0008810097570857156, 0.477492107, 0.478658532, 0.487609382, 0.129273954, 0.0217260302], #case 14
        [1.8070007936604389, 0.02265275355736706, 0.02010604629399581, 0.0025467072633712506, 0.342609412, 0.342986225, 0.354611882, 0.101825454, 0.0181743682], #case 15
        [1.6989731211360986, 0.027326412585346255, 0.022813785457516406, 0.004512627127829849, 0.336342386, 0.336870446, 0.355659805, 0.12775834, 0.0242421605] #case 16
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
    plt.savefig('calibration_results_ManningN_upstream_water_depth.png', dpi=300, bbox_inches='tight', transparent=False)
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
    plt.savefig('calibration_results_ManningN_downstream_water_depth.png', dpi=300, bbox_inches='tight', transparent=False)
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
    plt.savefig('calibration_results_ManningN_flow_partition.png', dpi=300, bbox_inches='tight', transparent=False)
    #plt.show()

    print("All done!")