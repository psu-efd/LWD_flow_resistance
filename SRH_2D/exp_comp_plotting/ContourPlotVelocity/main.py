import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib import rcParams

# Set the font family to Times New Roman and increase the font size
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 20  # Default font size for all text
rcParams['axes.titlesize'] = 24  # Title font size
rcParams['axes.labelsize'] = 22  # X and Y label font size
rcParams['xtick.labelsize'] = 18  # X tick labels font size
rcParams['ytick.labelsize'] = 18  # Y tick labels font size
rcParams['legend.fontsize'] = 20  # Legend font size

max_elev = 48
row_number = 5

def calculate_total_discharge(velocity_matrix, width_segment, height_segments):
    import numpy as np

    # Convert areas from cm^2 to m^2 (10,000 cm^2 in 1 m^2)
    areas_m2 = (width_segment * height_segments) / 10000

    print("Velocity matrix :", velocity_matrix)
    print("Areas :", areas_m2)
    print("")

    # Calculate discharge for each segment
    discharge_per_cell = velocity_matrix * areas_m2[:, None]  # Broadcast height along rows

    # Total discharge (m^3/s)
    total_discharge = np.sum(discharge_per_cell)
    return total_discharge

# Load data from Excel file
excel_file_path = 'Case13/00inch_obstruction_half.xlsx'  # Replace with your actual file path
data = pd.read_excel(excel_file_path)

# Remove unnecessary columns (if the first column is unnamed and contains NaNs)
cleaned_data = data.drop(columns=[data.columns[0]]).dropna()

# Reshape the data to a 5x5 matrix, assuming the data is structured correctly for this shape
velocity_matrix = cleaned_data.values.reshape((row_number, 5))

# Flip the matrix vertically to match the Excel display
velocity_matrix = np.flipud(velocity_matrix)

# Add zero velocity values at the borders
# Pad left and right columns with zeros
velocity_matrix_padded = np.pad(velocity_matrix, ((0, 0), (1, 1)), mode='constant', constant_values=0)
# Pad the bottom row with zeros (now the top, after flipping)
velocity_matrix_padded = np.pad(velocity_matrix_padded, ((1, 0), (0, 0)), mode='constant', constant_values=0)

# Adjust dimensions for the new padding
x = np.linspace(0, 150, velocity_matrix_padded.shape[1])  # Adapted for the padding
y_segments = [0, 9, 18, 27, 36, max_elev]  # Changed to reflect an increase at the top grid
y = np.array(y_segments)
X, Y = np.meshgrid(x, y)

# Create the contour plot
plt.figure(figsize=(11, 6))  # Slightly larger figure size
contour = plt.contourf(X, Y, velocity_matrix_padded, cmap='viridis', levels=100, vmin=0, vmax=0.5)
cbar = plt.colorbar(contour)
cbar.set_label('Mean Velocity (m/s)')

# Draw an independent grid
grid_x = np.linspace(0, 150, 8)  # 7 columns so 8 lines including boundaries
grid_y = np.linspace(0, 50, 5)  # 6 rows so 7 lines including boundaries
for gx in grid_x:
    plt.axvline(x=gx, color='white', linestyle='--', linewidth=0.5)
for gy in grid_y:
    plt.axhline(y=gy, color='white', linestyle='--', linewidth=0.5)

# Annotate each cell with the velocity value
for i in range(velocity_matrix_padded.shape[0]):
    for j in range(velocity_matrix_padded.shape[1]):
        # All text in white for consistency and visibility
        color = 'white'

        # Adjust the text position for edge cells
        ha = 'center'
        va = 'center'
        if j == 0:  # Shift right for left edge
            ha = 'left'
        elif j == velocity_matrix_padded.shape[1] - 1:  # Shift left for right edge
            ha = 'right'
        if i == 0:  # Shift up for bottom edge
            va = 'bottom'
        elif i == velocity_matrix_padded.shape[0] - 1:  # Shift down for top edge
            va = 'top'

        text = plt.text(X[i, j], Y[i, j], f'{velocity_matrix_padded[i, j]:.3f}',
                 color=color, ha=ha, va=va, fontsize=14, path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
        # Adding black stroke around the text for better visibility

plt.title('Contour Plot of Mean Water Velocity in a Flume')
plt.xlabel('Width (cm)')
plt.ylabel('Water Surface Elevation (cm)')
plt.tight_layout(pad=0.1)  # Add this line to adjust layout and remove extra whitespace
plt.savefig('contour_plot.png', bbox_inches='tight', pad_inches=0.1)  # Save the figure with tight layout
plt.show()

width_segment = 30  # cm
height_segments = np.array([9, 9, 9, 9, (max_elev - 9*(row_number - 1))])  # cm

# print("Velocity matrix :", velocity_matrix)
total_discharge = calculate_total_discharge(velocity_matrix, width_segment, height_segments)
print("Total Discharge across the flume cross-section: {:.4f} m^3/s".format(total_discharge))
