import os
import subprocess

import HEC_RAS_solver_module

if __name__ == "__main__":

    # List of case IDs (1-based)
    #case_IDs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    #case_IDs = [9, 13, 15, 16] 
    case_IDs = [9]

    ManningN_or_Cd = 'ManningN'  #'ManningN' or 'Cd'
    #ManningN_or_Cd = 'Cd'  #'ManningN' or 'Cd'
   
    #upper and lower bounds for the Manning's n for each case
    ManningN_lower_bounds = [1.0, #case 1
                             1.0, #case 2
                             1.5, #case 3
                             1.5, #case 4
                             0.5, #case 5
                             0.4, #case 6
                             0.5, #case 7
                             0.5, #case 8
                             1, #case 9
                             1.5, #case 10
                             1.5, #case 11
                             1.0, #case 12
                             1.0, #case 13
                             1.0, #case 14
                             1.0, #case 15
                             0.5] #case 16
    
    ManningN_upper_bounds = [3.5, #case 1
                             3.5, #case 2
                             3.0, #case 3
                             3.5, #case 4
                             2.0, #case 5
                             1.5, #case 6
                             1.6, #case 7
                             2.0, #case 8
                             5.0, #case 9
                             3.0, #case 10
                             3.0, #case 11
                             3.0, #case 12
                             4.0, #case 13
                             5.0, #case 14
                             5.0, #case 15
                             4.0] #case 16
                         
    

    # Loop over the list of directories
    for case_ID in case_IDs:
        try:          
            print(f"Running calibration for case {case_ID}")

            #set the global variable current_case_ID
            HEC_RAS_solver_module.current_case_ID = case_ID

            #set the parameter bounds
            parameter_bounds = [(ManningN_lower_bounds[case_ID-1], ManningN_upper_bounds[case_ID-1])]

            print(f"parameter_bounds = {parameter_bounds}")

            # Run the calibration for the current case
            #HEC_RAS_solver_module.optimize_model_parameter(ManningN_or_Cd, parameter_bounds)          

            #plot the results
            #HEC_RAS_solver_module.plot_optimization_results(ManningN_or_Cd, parameter_bounds)
            #HEC_RAS_solver_module.plot_calibration_history()

            #run the cases with the optimized ManningN
            HEC_RAS_solver_module.run_case_with_optimized_ManningN(case_ID)
        
        except Exception as e:
            # Handle general exceptions (e.g., directory change errors)
            print(f"Error in calibration of case {case_ID}: {e}")
        

    print("All done!")