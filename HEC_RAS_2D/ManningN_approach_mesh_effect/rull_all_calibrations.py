import os
import subprocess

import HEC_RAS_solver_module

if __name__ == "__main__":

    # List of case IDs (1-based)
    #case_IDs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    case_indicies = [14]
    case_IDs = [4, 41, 42, 43, 8, 81, 82, 83, 12, 121, 122, 123, 16, 161, 162, 163]
    case_names = ['case_4', 'case_41', 'case_42', 'case_43', 
                  'case_8', 'case_81', 'case_82', 'case_83',
                  'case_12', 'case_121', 'case_122', 'case_123',
                  'case_16', 'case_161', 'case_162', 'case_163']

    ManningN_or_Cd = 'ManningN'  #'ManningN' or 'Cd'
    #ManningN_or_Cd = 'Cd'  #'ManningN' or 'Cd'
   
    #upper and lower bounds for the Manning's n for each case
    ManningN_lower_bounds = [1.0, #case 4
                             1.0, #case 41
                             1.0, #case 42
                             1.0, #case 43
                             1.0, #case 8
                             1.0, #case 81
                             1.0, #case 82
                             1.0, #case 83
                             1.0, #case 12
                             1.0, #case 121
                             1.0, #case 122
                             1.0, #case 123
                             1.0, #case 16
                             1.0, #case 161
                             1.0, #case 162
                             3.0] #case 163
    
    ManningN_upper_bounds = [2.5, #case 4
                             2.5, #case 41
                             2.5, #case 42
                             3.5, #case 43
                             2.5, #case 8
                             2.5, #case 81
                             2.5, #case 82
                             2.5, #case 83
                             5.0, #case 12
                             5.0, #case 121
                             5.0, #case 122
                             5.0, #case 123
                             4.0, #case 16
                             3.0, #case 161
                             4.0, #case 162
                             10.0] #case 163
                         
    

    # Loop over the list of directories
    for case_index in case_indicies:
        case_ID = case_IDs[case_index-1]
        case_name = case_names[case_index-1]

        try:          
            print(f"Running calibration for case {case_ID}")

            #set the global variable current_case_ID
            HEC_RAS_solver_module.current_case_ID = case_ID

            #set the parameter bounds
            parameter_bounds = [(ManningN_lower_bounds[case_index-1], ManningN_upper_bounds[case_index-1])]

            print(f"parameter_bounds = {parameter_bounds}")

            # Run the calibration for the current case
            HEC_RAS_solver_module.optimize_model_parameter(ManningN_or_Cd, parameter_bounds)          

            #plot the results
            HEC_RAS_solver_module.plot_optimization_results(ManningN_or_Cd, parameter_bounds)
            HEC_RAS_solver_module.plot_calibration_history()

            #run the cases with the optimized ManningN
            HEC_RAS_solver_module.run_case_with_optimized_ManningN(case_ID, faceless=False)
        
        except Exception as e:
            # Handle general exceptions (e.g., directory change errors)
            print(f"Error in calibration of case {case_ID}: {e}")
        

    print("All done!")