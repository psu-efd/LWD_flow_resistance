# Plot all velocity contours for a given directory (case)

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize
import shutil

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"


if __name__ == "__main__":

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    case_dir = script_dir
    
    # Get the list of all subdirectories in the case directory: 16 cases from Case1 to Case16
    case_dirs = [os.path.join(case_dir, f"Case{i}") for i in range(1, 17)]
    case_IDs = [i for i in range(1, 17)]

    half_span_cases = [5, 6, 7, 8, 13, 14, 15, 16]
    full_span_cases = [1, 2, 3, 4, 9, 10, 11, 12]

    #For each case, set the upstream and downstream water depths (for plotting purposes)
    upstream_water_depths = [0.5, 0.51, 0.374, 0.377, 0.48, 0.48, 0.36, 0.36, 0.52, 0.54, 0.41, 0.44, 0.48, 0.48, 0.35, 0.36]
    downstream_water_depths = [0.47, 0.46, 0.323, 0.304, 0.48, 0.48, 0.36, 0.36, 0.44, 0.42, 0.29, 0.26, 0.48, 0.48, 0.345, 0.34]

    #Channel width
    channel_width = 1.5

    #Set the number of cases
    num_cases = len(case_dirs)

    #Set the color range
    vmins=[0, 0, 0, 0, -0.02, 0, 0, -0.02, 0, 0, 0, 0, 0, 0, 0, 0]
    vmaxs=[0.35, 0.4, 0.45, 0.5, 0.4, 0.4, 0.4, 0.5, 0.4, 0.55, 0.6, 0.65, 0.4, 0.5, 0.55, 0.68]
    
    # Loop through all cases
    for case_ID, case_dir in zip(case_IDs, case_dirs):

        #print the case number
        print(f"Case {case_ID}: {case_dir}")

        # Get the list of all subdirectories in the case directory
        subdirs = [os.path.join(case_dir, d) for d in os.listdir(case_dir) if os.path.isdir(os.path.join(case_dir, d))]

        #Excel data files are: 36inch_downstream.xlsx, 36inch_upstream.xlsx, 72inch_downstream.xlsx, 72inch_upstream.xlsx
        #Read the data from the excel files
        upstream36inch_data = pd.read_excel(os.path.join(case_dir, "36inch_upstream.xlsx"))
        downstream36inch_data = pd.read_excel(os.path.join(case_dir, "36inch_downstream.xlsx"))
        upstream72inch_data = pd.read_excel(os.path.join(case_dir, "72inch_upstream.xlsx"))
        downstream72inch_data = pd.read_excel(os.path.join(case_dir, "72inch_downstream.xlsx"))

        #read the data at LWD if half_span cases
        if case_ID in half_span_cases:
            LWD_data = pd.read_excel(os.path.join(case_dir, "00inch_obstruction.xlsx"))

        #remove the first column (which is NaN)
        upstream36inch_data = upstream36inch_data.iloc[:, 1:]
        downstream36inch_data = downstream36inch_data.iloc[:, 1:]
        upstream72inch_data = upstream72inch_data.iloc[:, 1:]
        downstream72inch_data = downstream72inch_data.iloc[:, 1:]

        if case_ID in half_span_cases:
            LWD_data = LWD_data.iloc[:, 1:]

        #pad to the left, bottom, and right of the data array with zeros
        upstream36inch_data = np.pad(upstream36inch_data, ((0, 1), (1, 1)), mode='constant', constant_values=0)
        downstream36inch_data = np.pad(downstream36inch_data, ((0, 1), (1, 1)), mode='constant', constant_values=0)
        upstream72inch_data = np.pad(upstream72inch_data, ((0, 1), (1, 1)), mode='constant', constant_values=0)
        downstream72inch_data = np.pad(downstream72inch_data, ((0, 1), (1, 1)), mode='constant', constant_values=0)

        if case_ID in half_span_cases:
            LWD_data = np.pad(LWD_data, ((0, 1), (1, 1)), mode='constant', constant_values=0)

        #print the data
        #print(upstream72inch_data)
        #print(upstream36inch_data)
        #print(downstream36inch_data)        
        #print(downstream72inch_data)

        #get min and max of the data combined
        min_value = np.min(np.concatenate([upstream72inch_data, upstream36inch_data, downstream36inch_data, downstream72inch_data]))
        max_value = np.max(np.concatenate([upstream72inch_data, upstream36inch_data, downstream36inch_data, downstream72inch_data]))

        if case_ID in half_span_cases:
            min_value = np.min(np.concatenate([upstream72inch_data, upstream36inch_data, downstream36inch_data, downstream72inch_data, LWD_data]))
            max_value = np.max(np.concatenate([upstream72inch_data, upstream36inch_data, downstream36inch_data, downstream72inch_data, LWD_data]))

        print("min value: ", min_value)
        print("max value: ", max_value)

        norm = Normalize(vmin=vmins[case_ID-1], vmax=vmaxs[case_ID-1])

        #compute the x and y coordinates of the data
        #x along the channel width
        x_upstream36inch = np.linspace(0, channel_width, upstream36inch_data.shape[1])
        x_downstream36inch = np.linspace(0, channel_width, downstream36inch_data.shape[1])
        x_upstream72inch = np.linspace(0, channel_width, upstream72inch_data.shape[1])
        x_downstream72inch = np.linspace(0, channel_width, downstream72inch_data.shape[1])
        #y along the water depth (flip so first row = top of water column)
        y_upstream36inch = np.linspace(upstream_water_depths[case_ID-1], 0, upstream36inch_data.shape[0])
        y_downstream36inch = np.linspace(downstream_water_depths[case_ID-1], 0, downstream36inch_data.shape[0])
        y_upstream72inch = np.linspace(upstream_water_depths[case_ID-1], 0, upstream72inch_data.shape[0])
        y_downstream72inch = np.linspace(downstream_water_depths[case_ID-1], 0, downstream72inch_data.shape[0])

        if case_ID in half_span_cases:
            x_LWD = np.linspace(0, channel_width, LWD_data.shape[1])
            y_LWD = np.linspace(upstream_water_depths[case_ID-1], 0, LWD_data.shape[0])

        #print the x and y coordinates
        #print(x)
        #print(y_upstream)
        #print(y_downstream)

        #print(upstream72inch_data.shape)
        #print("length of x_upstream36inch: ", len(x_upstream36inch))
        #print("length of y_upstream36inch: ", len(y_upstream36inch))
        #print("length of y_downstream36inch: ", len(y_downstream36inch))
        #print("length of x_upstream72inch: ", len(x_upstream72inch))
        #print("length of y_upstream72inch: ", len(y_upstream72inch))
        #print("length of y_downstream72inch: ", len(y_downstream72inch))

        #create a meshgrid of the x and y coordinates
        X_upstream36inch, Y_upstream36inch = np.meshgrid(x_upstream36inch, y_upstream36inch)
        X_downstream36inch, Y_downstream36inch = np.meshgrid(x_downstream36inch, y_downstream36inch)
        X_upstream72inch, Y_upstream72inch = np.meshgrid(x_upstream72inch, y_upstream72inch)
        X_downstream72inch, Y_downstream72inch = np.meshgrid(x_downstream72inch, y_downstream72inch)

        if case_ID in half_span_cases:
            X_LWD, Y_LWD = np.meshgrid(x_LWD, y_LWD)
        
        # Create contour plots using subplots (1 column, 4 rows)
        if case_ID in half_span_cases:
            fig, axes = plt.subplots(5, 1, figsize=(5, 10), sharex=True, sharey=False)
        else:
            #fig, axes = plt.subplots(4, 1, figsize=(5, 8), sharex=True, sharey=False)
            fig, axes = plt.subplots(5, 1, figsize=(5, 10), sharex=True, sharey=False)

        fig.suptitle(f'Case {case_ID}', fontsize=20)

        #colorbar ticks
        ticks = np.linspace(vmins[case_ID-1], vmaxs[case_ID-1], 6)

        subfig_counter = 0

        # Plot upstream 72 inch (1.83 m) data
        im3 = axes[subfig_counter].contourf(X_upstream72inch, Y_upstream72inch, upstream72inch_data, levels=np.linspace(vmins[case_ID-1], vmaxs[case_ID-1], 21), norm=norm, cmap='viridis', extend='both')
        axes[subfig_counter].set_title('I: 1.83 m Upstream', fontsize=18)
        axes[subfig_counter].set_ylabel('z (m)', fontsize=18)
        axes[subfig_counter].set_ylim(0, 0.5)
        #set yticks to be at 0.1 intervals
        axes[subfig_counter].set_yticks(np.linspace(0, 0.5, 6))
        axes[subfig_counter].set_xticks(np.linspace(0, 1.5, 7))
        axes[subfig_counter].tick_params(axis='both', labelsize=18)

        #colorbar
        cbar = plt.colorbar(im3, ax=axes[subfig_counter], ticks=ticks)
        cbar.set_label('Velocity (m/s)', fontsize=18)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_yticklabels([f'{tick:.2f}' for tick in ticks])

        subfig_counter += 1
        
        # Plot upstream 36 inch (0.91 m) data
        im1 = axes[subfig_counter].contourf(X_upstream36inch, Y_upstream36inch, upstream36inch_data, levels=np.linspace(vmins[case_ID-1], vmaxs[case_ID-1], 21), norm=norm, cmap='viridis', extend='both')
        axes[subfig_counter].set_title('II: 0.91 m Upstream', fontsize=18)
        axes[subfig_counter].set_ylabel('z (m)', fontsize=18)
        axes[subfig_counter].set_ylim(0, 0.5)
        axes[subfig_counter].set_yticks(np.linspace(0, 0.5, 6))
        axes[subfig_counter].set_xticks(np.linspace(0, 1.5, 7))
        axes[subfig_counter].tick_params(axis='both', labelsize=18)
        cbar = plt.colorbar(im1, ax=axes[subfig_counter], ticks=ticks)
        cbar.set_label('Velocity (m/s)', fontsize=18)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_yticklabels([f'{tick:.2f}' for tick in ticks])

        subfig_counter += 1

        if case_ID in half_span_cases:
            im5 = axes[subfig_counter].contourf(X_LWD, Y_LWD, LWD_data, levels=np.linspace(vmins[case_ID-1], vmaxs[case_ID-1], 21), norm=norm, cmap='viridis', extend='both')
            axes[subfig_counter].set_title('III: At LWD', fontsize=18)
            axes[subfig_counter].set_ylabel('z (m)', fontsize=18)
            axes[subfig_counter].set_ylim(0, 0.5)
            axes[subfig_counter].set_yticks(np.linspace(0, 0.5, 6))
            axes[subfig_counter].set_xticks(np.linspace(0, 1.5, 7))
            axes[subfig_counter].tick_params(axis='both', labelsize=18)

            cbar = plt.colorbar(im5, ax=axes[subfig_counter], ticks=ticks)
            cbar.set_label('Velocity (m/s)', fontsize=18)
            cbar.ax.tick_params(labelsize=16)
            cbar.ax.set_yticklabels([f'{tick:.2f}' for tick in ticks])

            subfig_counter += 1
        else:   #for full span cases, no plot, but need to add a blank plot (with a text saying "No LWD")
            axes[subfig_counter].axis('off')
            axes[subfig_counter].text(0.5, 0.5, 'No Cross-Section III at LWD', fontsize=18, ha='center', va='center')
            subfig_counter += 1
        
        # Plot downstream 36 inch (0.91 m) data
        im2 = axes[subfig_counter].contourf(X_downstream36inch, Y_downstream36inch, downstream36inch_data, levels=np.linspace(vmins[case_ID-1], vmaxs[case_ID-1], 21), norm=norm, cmap='viridis', extend='both')
        axes[subfig_counter].set_title('IV: 0.91 m Downstream', fontsize=18)
        axes[subfig_counter].set_ylabel('z (m)', fontsize=18)
        axes[subfig_counter].set_ylim(0, 0.5)
        axes[subfig_counter].set_yticks(np.linspace(0, 0.5, 6))
        axes[subfig_counter].set_xticks(np.linspace(0, 1.5, 7))
        axes[subfig_counter].tick_params(axis='both', labelsize=18)
        cbar = plt.colorbar(im2, ax=axes[subfig_counter], ticks=ticks)
        cbar.set_label('Velocity (m/s)', fontsize=18)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_yticklabels([f'{tick:.2f}' for tick in ticks])

        subfig_counter += 1
        
        # Plot downstream 72 inch (1.83 m) data
        im4 = axes[subfig_counter].contourf(X_downstream72inch, Y_downstream72inch, downstream72inch_data, levels=np.linspace(vmins[case_ID-1], vmaxs[case_ID-1], 21), norm=norm, cmap='viridis', extend='both')
        axes[subfig_counter].set_title('V: 1.83 m Downstream', fontsize=18)
        axes[subfig_counter].set_xlabel('x (m)', fontsize=18)
        axes[subfig_counter].set_ylabel('z (m)', fontsize=18)
        axes[subfig_counter].set_xlim(0, 1.5)
        axes[subfig_counter].set_ylim(0, 0.5)
        axes[subfig_counter].set_yticks(np.linspace(0, 0.5, 6))
        axes[subfig_counter].set_xticks(np.linspace(0, 1.5, 7))
        axes[subfig_counter].tick_params(axis='both', labelsize=18)
        cbar = plt.colorbar(im4, ax=axes[subfig_counter], ticks=ticks)
        cbar.set_label('Velocity (m/s)', fontsize=18)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_yticklabels([f'{tick:.2f}' for tick in ticks])
        
        subfig_counter += 1

        # Adjust layout and save
        plt.tight_layout()
        output_filename = os.path.join(case_dir, f"{os.path.basename(case_dir)}_velocity_contours.png")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight', transparent=True)
        plt.close()

        # Copy the png file to the current directory
        shutil.copy(output_filename, script_dir)
        
        print(f"Contour plot saved as: {output_filename}")

        
        
        
        








