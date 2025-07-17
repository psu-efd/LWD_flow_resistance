import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.colors import Normalize
import os
import glob

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

def load_elevation_data(filename):
    """
    Load elevation data from text file with x, y, z columns
    """
    data = np.loadtxt(filename)
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    
    return x, y, z

def plot_cases_combined(option):
    """
    Plot cases 1 to 4 or cases 5 to 8 in one figure with 4 rows and 1 column, sharing one colorbar
    option = 1 for cases 1 to 4
    option = 2 for cases 5 to 8
    """
    # Find experiment files for cases 1-4
    exp_files = []
    if option == 1:
        range_start = 1 
        range_end = 5
    elif option == 2:
        range_start = 5
        range_end = 9
    
    for i in range(range_start, range_end):
        exp_file = f"exp{i}_modified_terrain_elevation.txt"
        if os.path.exists(exp_file):
            exp_files.append(exp_file)
        else:
            print(f"Warning: {exp_file} not found!")

    
    print(f"Creating combined plot for cases {range_start} to {range_end-1}...")
    
    # Create figure with 4 subplots (4 rows, 1 column) with tight spacing
    fig, axes = plt.subplots(4, 1, figsize=(10, 6), sharex=True, sharey=False,
                            gridspec_kw={'hspace': 0.1, 'wspace': 0.1})
    
    # Specify the color range [0, 0.5]
    norm = Normalize(vmin=0, vmax=0.5)

    contourf_list = []
    
    # Process each case
    for i, exp_file in enumerate(sorted(exp_files)):
        print(f"Processing {exp_file}...")
        
        # Extract case number
        exp_num = exp_file.split('_')[0]
        case_num = exp_num.replace("exp", "")
        
        # Load data
        x, y, z = load_elevation_data(exp_file)
        
        # Clip z to be larger than 0
        z = np.clip(z, 0, None)

        print(f"z min: {np.min(z)}, z max: {np.max(z)}")
        
        # Find unique x and y values
        unique_x = np.unique(x)
        unique_y = np.unique(y)
        
        # Create meshgrid
        X, Y = np.meshgrid(unique_x, unique_y)

        #flip x axis values (negate the values)
        X = -X 
        
        # Reshape z values to match the grid
        Z = z.reshape(len(unique_x), len(unique_y)).T
        
        # Get current axis
        ax = axes[i]
        
        # Create contour plot
        contourf = ax.contourf(X, Y, Z, levels=np.linspace(0, 0.5, 21), cmap='terrain', alpha=0.8, norm=norm)
        contourf_list.append(contourf)

        # Set labels and title
        if i == 3:
            ax.set_xlabel('x (m)', fontsize=16)
        
        ax.set_ylabel('y (m)', fontsize=16)
        ax.set_title(f"Case {case_num}", fontsize=16, fontweight='bold')
        
        # Set axis tick font size
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Set axis ranges
        ax.set_xlim(-5, 10)
        ax.set_ylim(0, 1.5)
        
        # Set axis ticks
        ax.set_xticks([-5, -2.5, 0, 2.5, 5, 7.5, 10])
        ax.set_yticks([0, 0.5, 1, 1.5])
        
        # Set aspect ratio to equal
        ax.set_aspect('equal')
        
        # Add rectangular box centered at x = 0
        box_x_left = -0.05
        box_x_right = 0.05
        box_y_bottom = 0.0
        
        if int(case_num) < 5:
            box_y_top = 1.5
        else:
            box_y_top = 0.75
        
        # Draw the rectangular box with dashed lines
        from matplotlib.patches import Rectangle
        rect = Rectangle((box_x_left, box_y_bottom), 
                        box_x_right - box_x_left, 
                        box_y_top - box_y_bottom,
                        linewidth=2, 
                        edgecolor='red', 
                        facecolor='none', 
                        linestyle='--')
        ax.add_patch(rect)
    
    # Add shared colorbar positioned to the right
    # Create a new axes for the colorbar that doesn't overlap with subplots
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height]

    tick_min = 0
    tick_max = 0.5
    ticks = np.linspace(tick_min, tick_max, 6)

    if option == 1:
        cbar = plt.colorbar(contourf_list[0], cax=cbar_ax, ticks=ticks)
    elif option == 2:
        cbar = plt.colorbar(contourf_list[3], cax=cbar_ax, ticks=ticks)

    cbar.set_label('Elevation (m)', fontsize=14)

    # Set colorbar tick labels based on actual data range
    # Create 6 evenly spaced ticks across the actual data range    
    #cbar.set_ticks(ticks)
    #cbar.set_ticklabels([f'{tick:.1f}' for tick in ticks])
    cbar.ax.tick_params(labelsize=14)
    
    # Adjust layout to make room for colorbar
    plt.subplots_adjust(right=0.88)
    
    # Save the combined plot
    save_filename = f"Cases_{range_start}_to_{range_end-1}_combined.png"
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved as: {save_filename}")
    
    # Show the plot
    #plt.show()


def main():
    """
    Main function to run the plotting script
    """
    print("Bed Elevation Contour Plotting Script")
    print("=" * 40)
    
    # Check if we have experiment files
    exp_files = glob.glob("exp*_modified_terrain_elevation.txt")
    
    if exp_files:
        print(f"Found {len(exp_files)} experiment files.")

        plot_cases_combined(1)
        plot_cases_combined(2)


    else:
        print("No experiment files found in current directory!")
        print("Make sure you have files named 'exp*_modified_terrain_elevation.txt'")

if __name__ == "__main__":
    main()
