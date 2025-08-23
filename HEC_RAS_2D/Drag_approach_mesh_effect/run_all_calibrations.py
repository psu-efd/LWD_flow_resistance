#This script is used to run the calibration for all the cases: The b_coefficient is calibrated for each case

import os
import subprocess

import HEC_RAS_solver_module

import pyHMT2D

if __name__ == "__main__":

    pyHMT2D.setVerbose(True)

    # List of case IDs (1-based)
    #case_IDs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    #case_indicies = [1, 2, 3, 4]
    #case_indicies = [4, 7, 8, 11, 14, 15, 16]
    case_indicies = [8]
    case_IDs = [4, 41, 42, 43, 8, 81, 82, 83, 12, 121, 122, 123, 16, 161, 162, 163]
    case_names = ['case_4', 'case_41', 'case_42', 'case_43', 
                  'case_8', 'case_81', 'case_82', 'case_83',
                  'case_12', 'case_121', 'case_122', 'case_123',
                  'case_16', 'case_161', 'case_162', 'case_163']
    
    #upper and lower bounds for the b_coefficient for each case
    b_Coefficient_lower_bounds = [50.0, #case 4
                             50.0, #case 41
                             50.0, #case 42
                             20.0, #case 43
                             50.0, #case 8
                             50.0, #case 81
                             50.0, #case 82
                             120.0, #case 83
                             50.0, #case 12
                             50.0, #case 121
                             120.0, #case 122
                             50.0, #case 123
                             50.0, #case 16
                             100.0, #case 161
                             150.0, #case 162
                             150.0] #case 163
    
    b_Coefficient_upper_bounds = [250.0, #case 4
                             250.0, #case 41
                             250.0, #case 42
                             100.0, #case 43
                             150.0, #case 8
                             150.0, #case 81
                             150.0, #case 82
                             300.0, #case 83
                             160.0, #case 12
                             160.0, #case 121
                             300.0, #case 122
                             160.0, #case 123
                             250.0, #case 16
                             200.0, #case 161
                             220.0, #case 162
                             300.0] #case 163
                         
    

    # Loop over the list of directories
    for case_index in case_indicies:
        case_ID = case_IDs[case_index-1]
        case_name = case_names[case_index-1]

        #try:          
        print(f"Running calibration for case {case_ID} with case name {case_name}")

        #clear the history
        HEC_RAS_solver_module.clear_global_variables()

        #set the global variable current_case_ID
        HEC_RAS_solver_module.current_case_ID = case_ID

        #set the parameter bounds
        parameter_bounds = [(b_Coefficient_lower_bounds[case_index-1], b_Coefficient_upper_bounds[case_index-1])]

        print(f"parameter_bounds = {parameter_bounds}")

        # Run the calibration for the current case
        HEC_RAS_solver_module.optimize_model_parameter(parameter_bounds)          

        #plot the results
        HEC_RAS_solver_module.plot_optimization_results(parameter_bounds)
        HEC_RAS_solver_module.plot_calibration_history()

        #run the cases with the optimized b_coefficient
        HEC_RAS_solver_module.run_case_with_optimized_b_Coefficient(case_ID)            
        
        #except Exception as e:
        #    # Handle general exceptions (e.g., directory change errors)
        #    print(f"Error in calibration of case {case_ID}: {e}")
        

    print("All done!")