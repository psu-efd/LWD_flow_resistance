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

    # calibration results for each case: ManningN error_total error_H error_Q_split H_p1 H_p2 H_p3 Q_Line2 Q_Line3
    calibration_results_SRH_2D = [
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

    # calibration results for each case: b_Coefficient error_total error_H error_Q_split H_upstream H_downstream Q_split
    calibration_results_HEC_RAS_2D = [
        [78.09114082172809, 0.0011615790130877221, 0.0011615790130877221, 0.0, 0.49244916, 0.7500332, 0.5045783001753052],  #case 1
        [79.23308105889305, 0.00148502853992155, 0.00148502853992155, 0.0, 0.5036986, 0.7400712, 0.5050578586433077], #case 2
        [122.67221733708962, 0.0009383347550352152, 0.0009383347550352152, 0.0, 0.36730498, 0.6030647, 0.4991508938811488],   #case 3
        [120.8697972197206, 0.0012337703580946203, 0.0012337703580946203, 0.0, 0.37038165, 0.5841181, 0.4998251214848847],    #case 4
        [38.98943455036432, 0.012007472721147579, 0.011249576167969207, 0.0007578965531783721, 0.478177, 0.76023144, 0.7782421034468217],    #case 5
        [29.991225140019786, 0.009298291678633542, 0.00526800600629256, 0.004030285672340983, 0.47529048, 0.7596766, 0.770969714327659],    #case 6
        [46.20852533720391, 0.0071687153555331775, 0.006328001615521799, 0.0008407137400113784, 0.35510767, 0.63977134, 0.8231592862599886],   #case 7
        [56.72671368063625, 0.01288659874990343, 0.011671881639254741, 0.0012147171106486887, 0.35707337, 0.64008486, 0.8307852828893513],    #case 8
        [139.96222852282267, 0.0026118917898697892, 0.0026118917898697892, 0.0, 0.52113235, 0.4401911, 0.5001994381305627],    #case 9
        [130.01932872443226, 0.0032014244447939955, 0.0032014244447939955, 0.0, 0.54126024, 0.4203644, 0.500233975112799],    #case 10
        [115.60550771789084, 0.0027351063575136335, 0.0027351063575136335, 0.0, 0.41006067, 0.29075027, 0.5002300258856474],    #case 11
        [118.13935457790382, 0.006383782500153626, 0.006383782500153626, 0.0, 0.4399861, 0.26165158, 0.5002317321306708],    #case 12
        [149.53273329452986, 0.015408083402015013, 0.014523727198441824, 0.0008843562035731889, 0.4869207, 0.47994933, 0.8671156437964268],  #case 13
        [124.64954997089741, 0.024576400202657316, 0.023933393259843266, 0.0006430069428140506, 0.49062547, 0.48086256, 0.857643006942814], #case 14
        [104.05657173180725, 0.018165870666995192, 0.018075817366811447, 9.005330018374558e-05, 0.35628363, 0.3450423, 0.8459099466998162], #case 15
        [87.54682187404055, 0.007351065895248448, 0.004035052715563341, 0.0033160131796851067, 0.3593544, 0.3392378, 0.8326839868203149] #case 16
    ]

    #convert to numpy array
    calibration_results_SRH_2D = np.array(calibration_results_SRH_2D)
    calibration_results_HEC_RAS_2D = np.array(calibration_results_HEC_RAS_2D)

    # Bed elevation at upstream and downstream monitoring points
    bed_elevation_upstream = np.array([0.287, 0.287, 0.287, 0.287, 0.287, 0.287, 0.287, 0.287, 0, 0, 0, 0, 0, 0, 0, 0])
    bed_elevation_downstream = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    #compute upstream and downstream water depth at monitoring points 
    water_depth_upstream_exp = experiment_results[:, 0] - bed_elevation_upstream
    water_depth_downstream_exp = experiment_results[:, 1] - bed_elevation_downstream

    water_depth_upstream_simulation_SRH_2D = calibration_results_SRH_2D[:, 6] 
    water_depth_downstream_simulation_SRH_2D = calibration_results_SRH_2D[:, 5] 

    water_depth_upstream_simulation_HEC_RAS_2D = calibration_results_HEC_RAS_2D[:, 4] 
    water_depth_downstream_simulation_HEC_RAS_2D = calibration_results_HEC_RAS_2D[:, 5] 

    #compute RMSE error between simulation and measurement
    error_upstream_SRH_2D = np.abs(water_depth_upstream_simulation_SRH_2D - water_depth_upstream_exp)
    rmse_upstream_SRH_2D = np.sqrt(np.mean(error_upstream_SRH_2D**2))
    print("RMSE error between SRH-2Dsimulation and measurement for upstream water depth: ", rmse_upstream_SRH_2D)

    error_downstream_SRH_2D = np.abs(water_depth_downstream_simulation_SRH_2D - water_depth_downstream_exp)
    rmse_downstream_SRH_2D = np.sqrt(np.mean(error_downstream_SRH_2D**2))
    print("RMSE error between SRH-2D simulation and measurement for downstream water depth: ", rmse_downstream_SRH_2D)

    error_upstream_HEC_RAS_2D = np.abs(water_depth_upstream_simulation_HEC_RAS_2D - water_depth_upstream_exp)
    rmse_upstream_HEC_RAS_2D = np.sqrt(np.mean(error_upstream_HEC_RAS_2D**2))
    print("RMSE error between HEC-RAS 2D simulation and measurement for upstream water depth: ", rmse_upstream_HEC_RAS_2D)

    error_downstream_HEC_RAS_2D = np.abs(water_depth_downstream_simulation_HEC_RAS_2D - water_depth_downstream_exp)
    rmse_downstream_HEC_RAS_2D = np.sqrt(np.mean(error_downstream_HEC_RAS_2D**2))
    print("RMSE error between HEC-RAS 2D simulation and measurement for downstream water depth: ", rmse_downstream_HEC_RAS_2D)

    # Plot the comparison of upstream water depth between measurement and simulation
    plt.figure(figsize=(10, 6))
    plt.scatter(water_depth_upstream_simulation_SRH_2D, water_depth_upstream_exp, label='SRH-2D', marker='o', color='blue', s=120, facecolors='none')       #no fill 
    plt.scatter(water_depth_upstream_simulation_HEC_RAS_2D, water_depth_upstream_exp, label='HEC-RAS 2D', marker='^', color='red', s=120, facecolors='none')       #no fill 
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
    x_pos = 0.05 * (0.6 - 0.3) + 0.3
    y_pos = 0.95 * (0.6 - 0.3) + 0.3
    plt.text(x_pos, y_pos, f'RMSE (SRH-2D) = {rmse_upstream_SRH_2D:.4f} m', fontsize=18)
    y_pos = y_pos - 0.025
    plt.text(x_pos, y_pos, f'RMSE (HEC-RAS 2D) = {rmse_upstream_HEC_RAS_2D:.4f} m', fontsize=18)

    #equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    plt.legend(fontsize=24, loc='lower right', frameon=False)  
    plt.savefig('calibration_results_ManningN_upstream_water_depth.png', dpi=300, bbox_inches='tight', transparent=False)
    #plt.show()

    # Plot the comparison of downstream water depth between measurement and simulation
    plt.figure(figsize=(10, 6))
    plt.scatter(water_depth_downstream_simulation_SRH_2D, water_depth_downstream_exp, label='SRH-2D', marker='o', color='blue', s=120, facecolors='none')       #no fill 
    plt.scatter(water_depth_downstream_simulation_HEC_RAS_2D, water_depth_downstream_exp, label='HEC-RAS 2D', marker='^', color='red', s=120, facecolors='none')       #no fill 
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
    x_pos = 0.05 * (0.8 - 0.2) + 0.2
    y_pos = 0.95 * (0.8 - 0.2) + 0.2
    plt.text(x_pos, y_pos, f'RMSE (SRH-2D) = {rmse_downstream_SRH_2D:.4f} m', fontsize=18)
    y_pos = y_pos - 0.05
    plt.text(x_pos, y_pos, f'RMSE (HEC-RAS 2D) = {rmse_downstream_HEC_RAS_2D:.4f} m', fontsize=18)

    #equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    plt.legend(fontsize=24, loc='lower right', frameon=False)  
    plt.savefig('calibration_results_ManningN_downstream_water_depth.png', dpi=300, bbox_inches='tight', transparent=False)
    #plt.show()

    #compute the flow partition 
    flow_partition_exp = experiment_results[:, 2]
    flow_partition_simulation_SRH_2D = calibration_results_SRH_2D[:, 7]/(calibration_results_SRH_2D[:, 7] + calibration_results_SRH_2D[:, 8])
    flow_partition_simulation_HEC_RAS_2D = calibration_results_HEC_RAS_2D[:, 6]

    # Extract only data at indices 4 to 7 and 12 to 15 (half blockage cases; python index starts from 0)
    selected_indices = [4, 5, 6, 7, 12, 13, 14, 15]
    flow_partition_exp_selected = flow_partition_exp[selected_indices]
    flow_partition_simulation_selected_SRH_2D = flow_partition_simulation_SRH_2D[selected_indices]
    flow_partition_simulation_selected_HEC_RAS_2D = flow_partition_simulation_HEC_RAS_2D[selected_indices]

    print("flow_partition_exp_selected: ", flow_partition_exp_selected)
    print("flow_partition_simulation_selected_SRH_2D: ", flow_partition_simulation_selected_SRH_2D)
    print("flow_partition_simulation_selected_HEC_RAS_2D: ", flow_partition_simulation_selected_HEC_RAS_2D)

    #computer RMSE error between simulation and measurement for flow partition
    error_flow_partition_SRH_2D = np.abs(flow_partition_simulation_selected_SRH_2D - flow_partition_exp_selected)
    rmse_flow_partition_SRH_2D = np.sqrt(np.mean(error_flow_partition_SRH_2D**2))
    print("RMSE error between SRH-2D simulation and measurement for flow partition: ", rmse_flow_partition_SRH_2D)

    error_flow_partition_HEC_RAS_2D = np.abs(flow_partition_simulation_selected_HEC_RAS_2D - flow_partition_exp_selected)
    rmse_flow_partition_HEC_RAS_2D = np.sqrt(np.mean(error_flow_partition_HEC_RAS_2D**2))
    print("RMSE error between HEC-RAS 2D simulation and measurement for flow partition: ", rmse_flow_partition_HEC_RAS_2D)

    # Plot the comparison of flow partition between measurement and simulation
    plt.figure(figsize=(10, 6)) 
    plt.scatter(flow_partition_simulation_selected_SRH_2D, flow_partition_exp_selected, label='SRH-2D', marker='o', color='blue', s=120, facecolors='none')       #no fill 
    plt.scatter(flow_partition_simulation_selected_HEC_RAS_2D, flow_partition_exp_selected, label='HEC-RAS 2D', marker='^', color='red', s=120, facecolors='none')       #no fill 
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
    x_pos = 0.05 * (1 - 0.7) + 0.7
    y_pos = 0.95 * (1 - 0.7) + 0.7
    plt.text(x_pos, y_pos, f'RMSE (SRH-2D) = {rmse_flow_partition_SRH_2D:.4f}', fontsize=18)
    y_pos = y_pos - 0.025
    plt.text(x_pos, y_pos, f'RMSE (HEC-RAS 2D) = {rmse_flow_partition_HEC_RAS_2D:.4f}', fontsize=18)

    #equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    plt.legend(fontsize=24, loc='lower right', frameon=False)  
    plt.savefig('calibration_results_ManningN_flow_partition.png', dpi=300, bbox_inches='tight', transparent=False)
    #plt.show()

    print("All done!")