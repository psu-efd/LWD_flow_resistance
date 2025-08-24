import numpy as np
import matplotlib.pyplot as plt
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

def extract_longitudinal_profile(x, y, z, y_target=0.75):
    """
    Extract longitudinal profile at a specific y-coordinate
    """
    # Find data points closest to the target y-coordinate
    y_unique = np.unique(y)
    y_closest = y_unique[np.argmin(np.abs(y_unique - y_target))]
    
    # Extract data at this y-coordinate
    mask = np.abs(y - y_closest) < 1e-6
    x_profile = x[mask]
    z_profile = z[mask]
    
    # Sort by x-coordinate
    sort_idx = np.argsort(x_profile)
    x_profile = x_profile[sort_idx]
    z_profile = z_profile[sort_idx]
    
    return x_profile, z_profile, y_closest

def plot_longitudinal_profiles(cases, y_target=0.75, save_filename=None):
    """
    Plot longitudinal profiles for specified cases
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define colors for different cases
    colors = ['blue', 'red', 'green', 'orange']
    styles = ['-', '--', '-', '--']
    
    # Process each case
    for i, case_num in enumerate(cases):
        exp_file = f"exp{case_num}_modified_terrain_elevation.txt"
        
        if not os.path.exists(exp_file):
            print(f"Warning: {exp_file} not found!")
            continue
            
        print(f"Processing {exp_file}...")
        
        # Load data
        x, y, z = load_elevation_data(exp_file)

        #flip x axis values (negate the values)
        x = -x
        
        # Clip z to be larger than 0
        z = np.clip(z, 0, None)
        
        # Extract longitudinal profile
        x_profile, z_profile, y_actual = extract_longitudinal_profile(x, y, z, y_target)
        
        print(f"Case {case_num}: y = {y_actual:.3f}, z range: {np.min(z_profile):.6f} to {np.max(z_profile):.6f}")
        
        # Plot the profile
        ax.plot(x_profile, z_profile, color=colors[i], linewidth=2, linestyle=styles[i],
                label=f'Case {case_num}', marker='o', markersize=3, alpha=0.8)
    
    # Set labels and title
    ax.set_xlabel('x (m)', fontsize=20)
    ax.set_ylabel('Bed Elevation (m)', fontsize=20)

    if cases[0] == 1:
        ax.set_title(f'Bed Longitudinal Profiles for Full-Span Cases', fontsize=20, fontweight='bold')
    else:
        ax.set_title(f'Bed Longitudinal Profiles for Half-Span Cases', fontsize=20, fontweight='bold')
    
    # Set axis tick font size
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Set axis ranges
    ax.set_xlim(-5, 10)
    ax.set_ylim(0, 0.5)
    
    # Set axis ticks
    ax.set_xticks([-5, -2.5, 0, 2.5, 5, 7.5, 10])
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add shaded rectangular box to show obstruction location
    # Box centered at x = 0 with width 0.2 m, height from 0 to 0.5
    from matplotlib.patches import Rectangle
    box_x_left = -0.1  # x = 0 - 0.1 = -0.1
    box_x_right = 0.1  # x = 0 + 0.1 = 0.1
    box_y_bottom = 0.0
    box_y_top = 0.5
    
    rect = Rectangle((box_x_left, box_y_bottom), 
                    box_x_right - box_x_left, 
                    box_y_top - box_y_bottom,
                    linewidth=1, 
                    edgecolor='black', 
                    facecolor='gray', 
                    alpha=0.6)
    ax.add_patch(rect)
    
    # Add "LWD" text at the specified location
    ax.text(0, 0.45, 'LWD', fontsize=18, fontweight='bold', 
            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor='white', edgecolor='black', alpha=0.8))
    
    # Add legend
    ax.legend(fontsize=18, loc='upper right')
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot if filename is provided
    if save_filename:
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"Profile plot saved as: {save_filename}")
    
    # Show the plot
    #plt.show()
    
    return fig, ax

def main():
    """
    Main function to run the longitudinal profile plotting script
    """
    print("Bed Longitudinal Profile Plotting Script")
    print("=" * 50)
    
    # Check if we have experiment files
    exp_files = glob.glob("exp*_modified_terrain_elevation.txt")
    
    if not exp_files:
        print("No experiment files found in current directory!")
        print("Make sure you have files named 'exp*_modified_terrain_elevation.txt'")
        return
    
    print(f"Found {len(exp_files)} experiment files.")
    
    # Plot cases 1-4 on one figure
    print("\n" + "="*30)
    print("Plotting Cases 1-4")
    print("="*30)
    plot_longitudinal_profiles([1, 2, 3, 4], y_target=0.75, 
                              save_filename="Cases_1_to_4_longitudinal_profiles.png")
    
    # Plot cases 5-8 on another figure
    print("\n" + "="*30)
    print("Plotting Cases 5-8")
    print("="*30)
    plot_longitudinal_profiles([5, 6, 7, 8], y_target=0.75, 
                              save_filename="Cases_5_to_8_longitudinal_profiles.png")

if __name__ == "__main__":
    main() 