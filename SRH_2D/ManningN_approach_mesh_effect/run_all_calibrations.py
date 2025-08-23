import os
import subprocess

import sys
sys.path.append('..')
from LWD_module import clear_history

if __name__ == "__main__":

    # List of directories
    directories = ['Exp_4_ManningN_coarse_1',
                   'Exp_4_ManningN_coarse_2',
                   'Exp_4_ManningN_coarse_3',
                   'Exp_8_ManningN_coarse_1',
                   'Exp_8_ManningN_coarse_2',
                   'Exp_8_ManningN_coarse_3',
                   'Exp_12_ManningN_coarse_1',
                   'Exp_12_ManningN_coarse_2',
                   'Exp_12_ManningN_coarse_3',
                   'Exp_16_ManningN_coarse_1',
                   'Exp_16_ManningN_coarse_2',
                   'Exp_16_ManningN_coarse_3',
                   ]


    #directories = ['Exp_4_ManningN_coarse_3',
    #               'Exp_8_ManningN_coarse_3']

    script_name = 'calibration_run_LWD_gp.py'

    # Loop over the list of directories
    for directory in directories:
        try:
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