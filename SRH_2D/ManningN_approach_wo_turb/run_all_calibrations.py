import os
import subprocess

import sys
sys.path.append('..')
from LWD_module import clear_history

if __name__ == "__main__":

    # List of directories
    directories = ['Exp_1_ManningN',  #full
                   'Exp_2_ManningN',
                   'Exp_3_ManningN',
                   'Exp_4_ManningN',
                   'Exp_5_ManningN',   #half
                   'Exp_6_ManningN',
                   'Exp_7_ManningN',
                   'Exp_8_ManningN',
                   'Exp_9_ManningN',   #full
                   'Exp_10_ManningN',
                   'Exp_11_ManningN',
                   'Exp_12_ManningN',
                   'Exp_13_ManningN',  #half
                   'Exp_14_ManningN',
                   'Exp_15_ManningN',
                   'Exp_16_ManningN'
                   ]

    #directories = ['Exp_8_ManningN']

    script_name = 'calibration_run_LWD_gp.py'

    # Loop over the list of directories
    for directory in directories:
        try:
            # Clear the history
            clear_history()

            # Save the current working directory
            original_dir = os.getcwd()

            # Change the current working directory to the target directory
            os.chdir(directory)
            print(f"Entering directory: {directory}")

            # Run the Python script in the current directory using subprocess
            result = subprocess.run(['python', script_name], check=True, capture_output=True, text=True)

            # Print the output from the script
            print(f"Output from {script_name} in {directory}:")
            print(result.stdout)

        except subprocess.CalledProcessError as e:
            # Handle errors in the subprocess call (e.g., script not found or runtime error)
            print(f"Error running {script_name} in {directory}: {e.stderr}")

        except Exception as e:
            # Handle general exceptions (e.g., directory change errors)
            print(f"Error in directory {directory}: {e}")

        finally:
            # Always go back to the original working directory
            os.chdir(original_dir)
            print(f"Returning to directory: {original_dir}")

    print("All done!")