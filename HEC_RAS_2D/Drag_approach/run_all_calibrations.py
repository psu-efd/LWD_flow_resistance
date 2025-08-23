#This script is used to run the calibration for all the cases: The b_coefficient is calibrated for each case

import os
import subprocess

import HEC_RAS_solver_module

if __name__ == "__main__":

    # List of case IDs (1-based)
    case_IDs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    #case_IDs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] 
    case_IDs = [9]
    
    #upper and lower bounds for the b_coefficient for each case
    b_Coefficient_lower_bounds = [50.0, #case 1
                             50.0, #case 2
                             100.0, #case 3
                             100.0, #case 4
                             20.0, #case 5
                             20.0, #case 6
                             20.0, #case 7
                             20.0, #case 8
                             50.0, #case 9
                             100.0, #case 10
                             50.0, #case 11
                             50.0, #case 12
                             50.0, #case 13
                             50.0, #case 14
                             50.0, #case 15
                             50.0] #case 16
    
    b_Coefficient_upper_bounds = [150.0, #case 1
                             150.0, #case 2
                             200.0, #case 3
                             200.0, #case 4
                             80.0, #case 5
                             80.0, #case 6
                             80.0, #case 7
                             80.0, #case 8
                             160.0, #case 9
                             200.0, #case 10
                             200.0, #case 11
                             200.0, #case 12
                             250.0, #case 13
                             200.0, #case 14
                             200.0, #case 15
                             200.0] #case 16
                         
    

    # Loop over the list of directories
    for case_ID in case_IDs:
        try:          
            print(f"Running calibration for case {case_ID}")

            #set the global variable current_case_ID
            HEC_RAS_solver_module.current_case_ID = case_ID

            #set the parameter bounds
            parameter_bounds = [(b_Coefficient_lower_bounds[case_ID-1], b_Coefficient_upper_bounds[case_ID-1])]

            print(f"parameter_bounds = {parameter_bounds}")

            #clear the global variables
            #HEC_RAS_solver_module.clear_global_variables()

            # Run the calibration for the current case
            #HEC_RAS_solver_module.optimize_model_parameter(parameter_bounds)          

            #plot the results
            #HEC_RAS_solver_module.plot_optimization_results(parameter_bounds)
            #HEC_RAS_solver_module.plot_calibration_history()

            #run the cases with the optimized b_coefficient
            HEC_RAS_solver_module.run_case_with_optimized_b_Coefficient(case_ID)            
        
        except Exception as e:
            # Handle general exceptions (e.g., directory change errors)
            print(f"Error in calibration of case {case_ID}: {e}")
        

    print("All done!")