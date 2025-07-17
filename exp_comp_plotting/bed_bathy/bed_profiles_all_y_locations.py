import numpy as np
import matplotlib.pyplot as plt
import os

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

def extract_all_longitudinal_profiles(x, y, z):
    """
    Extract longitudinal profiles at all y locations
    """
    # Find unique x and y values
    unique_x = np.unique(x)
    unique_y = np.unique(y)
    
    # Sort y values
    unique_y = np.sort(unique_y)

    x_min = np.min(x)
    x_max = np.max(x)
    
    # Create meshgrid
    X, Y = np.meshgrid(unique_x, unique_y)
    
    # Reshape z values to match the grid
    Z = z.reshape(len(unique_x), len(unique_y)).T
    
    # Extract profiles at each y location
    profiles = {}
    for i, y_val in enumerate(unique_y):
        x_profile = unique_x + x_min
        z_profile = Z[i, :]
        profiles[y_val] = (x_profile, z_profile)

    #reverse the order of z_profile
    for i, y_val in enumerate(unique_y):
        z_profile = profiles[y_val][1]
        profiles[y_val] = (profiles[y_val][0], z_profile[::-1])
    
    return profiles, unique_x, unique_y

def plot_case_profiles(case_num, save_filename=None):
    """
    Plot longitudinal profiles at all y locations for a specific case
    """
    exp_file = f"exp{case_num}_modified_terrain_elevation.txt"
    
    if not os.path.exists(exp_file):
        print(f"Warning: {exp_file} not found!")
        return None, None
    
    print(f"Processing {exp_file}...")
    
    # Load data
    x, y, z = load_elevation_data(exp_file)
    
    # Flip x axis values (negate the values)
    x = -x
    
    # Clip z to be larger than 0
    z = np.clip(z, 0, None)
    
    # Extract all profiles
    profiles, unique_x, unique_y = extract_all_longitudinal_profiles(x, y, z)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot profiles at each y location
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_y)))
    
    # Calculate mean profile
    z_mean = np.zeros_like(unique_x)
    for i, x_val in enumerate(unique_x):
        z_values = []
        for y_val in unique_y:
            x_profile, z_profile = profiles[y_val]
            z_values.append(z_profile[i])
        z_mean[i] = np.mean(z_values)
    
    # Plot individual profiles
    for i, y_val in enumerate(unique_y):
        x_profile, z_profile = profiles[y_val]
        ax.plot(unique_x, z_profile, color='gray', linewidth=1, 
                alpha=0.3)
    
    # Plot mean profile
    ax.plot(unique_x, z_mean, color='black', linewidth=3, 
            label='Mean Profile', linestyle='-')
    
    # Set labels and title
    ax.set_xlabel('x (m)', fontsize=20)
    ax.set_ylabel('Bed Elevation (m)', fontsize=20)
    ax.set_title(f'Case {case_num}: Longitudinal Profiles and Mean', fontsize=20, fontweight='bold')
    
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
    from matplotlib.patches import Rectangle
    box_x_left = -0.1
    box_x_right = 0.1
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
    
    # Add "LWD" text
    ax.text(0, 0.45, 'LWD', fontsize=18, fontweight='bold', 
            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor='white', edgecolor='black', alpha=0.8))
    
    # Add legend with only two entries
    legend_elements = [
        plt.Line2D([0], [0], color='gray', linewidth=1, alpha=0.3, label='Individual Profiles'),
        plt.Line2D([0], [0], color='black', linewidth=3, label='Mean Profile')
    ]
    
    ax.legend(handles=legend_elements, fontsize=18, loc='upper right')
    
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
    print("Bed Longitudinal Profile Plotting Script - All Y Locations")
    print("=" * 60)
    
    # Check if we have experiment files
    exp_files = []
    for i in range(1, 5):
        exp_file = f"exp{i}_modified_terrain_elevation.txt"
        if os.path.exists(exp_file):
            exp_files.append(exp_file)
        else:
            print(f"Warning: {exp_file} not found!")
    
    if not exp_files:
        print("No experiment files found in current directory!")
        print("Make sure you have files named 'exp*_modified_terrain_elevation.txt'")
        return
    
    print(f"Found {len(exp_files)} experiment files.")
    
    # Plot each case
    for case_num in range(1, 5):
        print(f"\n" + "="*40)
        print(f"Plotting Case {case_num}")
        print("="*40)
        plot_case_profiles(case_num, 
                          save_filename=f"Case_{case_num}_all_y_profiles.png")

if __name__ == "__main__":
    main() 