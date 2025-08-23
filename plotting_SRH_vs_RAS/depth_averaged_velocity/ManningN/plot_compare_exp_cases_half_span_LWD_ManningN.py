#Plot results of flume experiment cases for visualization
#plot SRH-2D simulation results + measurement velocity profiles

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import griddata, interp2d

import pandas as pd

import os
import vtk
from vtk.util import numpy_support

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

def extract_data_from_vtk(vtkFileName, SRH_or_RAS):
    """
    extract data from a vtk file
    """

    # Load the VTK file
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtkFileName)  # Replace with your VTK file path
    reader.ReadAllScalarsOn()  # Ensure all scalar fields are read
    reader.ReadAllVectorsOn()  # Ensure all vector fields are read
    reader.Update()

    # Get the unstructured grid data
    data = reader.GetOutput()

    #print("data = ", data)

    # Extract cell centers and scalar data
    cell_centers = vtk.vtkCellCenters()
    cell_centers.SetInputData(data)
    cell_centers.Update()

    # Get points and scalar values from the cell centers
    points = np.array(
        [cell_centers.GetOutput().GetPoint(i)[:2] for i in range(cell_centers.GetOutput().GetNumberOfPoints())])

    #Flip x coordinates
    points[:, 0] = - points[:, 0]

    # Extract values from cell data (assuming the scalar field is at cell centers)
    if SRH_or_RAS == "SRH_2D":
        Velocity_m_p_s = data.GetCellData().GetArray("Velocity_m_p_s")
        #Vel_Mag_m_p_s = data.GetCellData().GetArray("Vel_Mag_m_p_s")
        Water_Depth_m = data.GetCellData().GetArray("Water_Depth_m")

    elif SRH_or_RAS == "HEC_RAS_2D":
        Velocity_m_p_s = data.GetCellData().GetArray("Velocity_cell_m_p_s")
        #Vel_Mag_m_p_s = data.GetCellData().GetArray("Vel_Mag_m_p_s")
        Water_Depth_m = data.GetCellData().GetArray("Water_Depth_m")

    if Velocity_m_p_s is None:
        raise ValueError("No Velocity_m_p_s data found at cell centers. Please check your VTK file.")

    # Convert data to a numpy array
    Vel_x_np = np.array([-Velocity_m_p_s.GetTuple3(i)[0] for i in range(Velocity_m_p_s.GetNumberOfTuples())])  #flip x velocity
    Vel_y_np = np.array([ Velocity_m_p_s.GetTuple3(i)[1] for i in range(Velocity_m_p_s.GetNumberOfTuples())])  # flip x velocity

    #compute velocity magnitude
    Vel_mag_np = np.sqrt(Vel_x_np**2 + Vel_y_np**2)

    # Check if points and scalars have compatible shapes
    if len(points) != len(Vel_mag_np):
        raise ValueError("Mismatch between number of cell centers and scalar values.")
    
    return points, Vel_x_np, Vel_y_np, Vel_mag_np


def plot_contour_from_vtk(case_ID, Fr, beta, ManningN_SRH_2D, ManningN_HEC_RAS_2D, alpha_exp, alpha_simulation_SRH_2D, alpha_simulation_HEC_RAS_2D, rect_x, rect_y, rect_width, rect_height, vtkFileName_SRH_2D, vtkFileName_HEC_RAS_2D, U_all, V_all, x_positions, y_positions):

    #load data from vtk file: water depht and velocity

    if not os.path.exists(vtkFileName_SRH_2D):
        return None

    points_SRH_2D, Vel_x_np_SRH_2D, Vel_y_np_SRH_2D, Vel_mag_np_SRH_2D = extract_data_from_vtk(vtkFileName_SRH_2D, "SRH_2D")
    points_HEC_RAS_2D, Vel_x_np_HEC_RAS_2D, Vel_y_np_HEC_RAS_2D, Vel_mag_np_HEC_RAS_2D = extract_data_from_vtk(vtkFileName_HEC_RAS_2D, "HEC_RAS_2D")

    # Check if points and scalars have compatible shapes
    if len(points_SRH_2D) != len(Vel_mag_np_SRH_2D):
        raise ValueError("Mismatch between number of cell centers and scalar values.")

    # Create a grid for contour plotting
    x = points_SRH_2D[:, 0]
    y = points_SRH_2D[:, 1]
    z = Vel_mag_np_SRH_2D

    x_RAS_2D = points_HEC_RAS_2D[:, 0]
    y_RAS_2D = points_HEC_RAS_2D[:, 1]
    z_RAS_2D = Vel_mag_np_HEC_RAS_2D

    # Create a regular grid interpolated from the scattered data
    xi = np.linspace(x.min()+0.01, x.max()-0.01, 420)
    yi = np.linspace(y.min()+0.01, y.max()-0.01, 60)
    X, Y = np.meshgrid(xi, yi)

    # Interpolate the scalar field onto the grid
    Z = griddata(points_SRH_2D, z, (X, Y), method="linear")

    vmin = Z.min()
    vmax = Z.max()

    # Plot the contour
    fig, ax = plt.subplots(figsize=(42, 6))

    #contour = ax.contourf(X, Y, Z, levels=20, cmap="coolwarm", vmin=vmin, vmax=vmax)
    contour = ax.contourf(X, Y, Z, levels=20, cmap="viridis", vmin=vmin, vmax=vmax)
    
    cbar = plt.colorbar(contour, pad=0.02) #, shrink=1.0, fraction=0.1, aspect=40)

    cbar.set_label(label="Velocity (m/s)", fontsize=48)

    tick_positions = np.linspace(vmin, vmax, 5)

    cbar.set_ticks(tick_positions)  # Apply custom ticks
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar.ax.tick_params(labelsize=40)  # Set tick font size

    #plot velocity vectors from experiments

    for iProfile in range(5):
        #How many points of velocity measurement on the profile
        if iProfile == 2:   #the profile at LWD only has 3 points; others have 5.
            nPoints = 3
        else:
            nPoints = 5

        coords_at_points = np.zeros((nPoints,2))

        vel_x_SRH_2D_at_points = np.zeros(nPoints)
        vel_y_SRH_2D_at_points = np.zeros(nPoints)

        for iPoint in range(nPoints):
            coords_at_points[iPoint,0] = x_positions[iProfile]
            coords_at_points[iPoint,1] = y_positions[iProfile][iPoint]

        interpolated_SRH_2D_velocity_x = griddata((x, y), Vel_x_np_SRH_2D, coords_at_points, method='linear')
        interpolated_SRH_2D_velocity_y = griddata((x, y), Vel_y_np_SRH_2D, coords_at_points, method='linear')

        interpolated_RAS_2D_velocity_x = griddata((x_RAS_2D, y_RAS_2D), Vel_x_np_HEC_RAS_2D, coords_at_points, method='linear')
        interpolated_RAS_2D_velocity_y = griddata((x_RAS_2D, y_RAS_2D), Vel_y_np_HEC_RAS_2D, coords_at_points, method='linear')

        # Plot the profile
        for iPoint in range(nPoints):
            #plot velocity from experiment
            plt.arrow(coords_at_points[iPoint, 0], coords_at_points[iPoint, 1], U_all[iProfile][iPoint], -V_all[iProfile][iPoint], linewidth=2, head_width=0.04, head_length=0.02, fc='blue', ec='blue', length_includes_head=True)
            plt.plot(coords_at_points[iPoint, 0] + U_all[iProfile][iPoint] + 0.01, coords_at_points[iPoint, 1]-V_all[iProfile][iPoint], 'ko', markersize=4)  # Smaller dot (size 4) to the right of the arrow tip

            #plot SRH-2D velocity
            plt.arrow(coords_at_points[iPoint, 0], coords_at_points[iPoint, 1],
                      interpolated_SRH_2D_velocity_x[iPoint], interpolated_SRH_2D_velocity_y[iPoint],
                      head_width=0.04, head_length=0.02,fc='yellow', ec='yellow',
                      linestyle="--", linewidth=2, length_includes_head=True)
            
            #plot HEC-RAS 2D velocity
            plt.arrow(coords_at_points[iPoint, 0], coords_at_points[iPoint, 1],
                      interpolated_RAS_2D_velocity_x[iPoint], interpolated_RAS_2D_velocity_y[iPoint],
                      head_width=0.04, head_length=0.02,fc='black', ec='black', alpha=0.5,
                      linestyle="--", linewidth=2, length_includes_head=True)

        #plot a velocity vector scale
        plt.arrow(-4, 0.75, 0.5, 0.0,
                  head_width=0.04, head_length=0.02, fc='black', ec='black',
                  linewidth=2, length_includes_head=True)

        plt.text(-3.75, 0.8, "0.5 m/s", color="black", fontsize=36, horizontalalignment='center',)

        dot_x_positions1 = [x_positions[iProfile] + value + 0.01 for value in U_all[iProfile]]

        dot_y_positions1 = []

        for y_i, value in enumerate(V_all[iProfile]):
            dot_y_positions1.append(y_positions[iProfile][y_i] - value)

        plt.plot(dot_x_positions1, dot_y_positions1, 'k-', linewidth=1)  # Curve connecting the dots for profile
        plt.plot([x_positions[iProfile], x_positions[iProfile]], [y_positions[iProfile][0], y_positions[iProfile][-1]], 'k--',
                 linewidth=1)  # Dotted vertical line for profile

    #rectangle = plt.Rectangle((rect_x, rect_y), rect_width, rect_height, linewidth=2, edgecolor="white", facecolor="none")
    rectangle = plt.Rectangle((rect_x, rect_y), rect_width, rect_height, linewidth=2, linestyle='--', edgecolor="red", facecolor="none")
    ax.add_patch(rectangle)

    #ax.set_xlim(-50.1+0.02, 25.1-0.02)
    #ax.set_ylim(0+0.02, 10-0.02)
    #ax.set_xlim(xi.min(), xi.max())
    #ax.set_ylim(yi.min(), yi.max())
    ax.set_xlim(-5, 10)
    ax.set_ylim(0, 1.5)

    ax.set_xlabel("x (m)", fontsize=48)
    ax.set_ylabel("y (m)", fontsize=48)

    #set tick label
    ax.set_xticks(np.arange(-5, 11, 1))  # Set x ticks from -5 to 10 in 1 m increments
    ax.set_xticklabels([str(int(x)) for x in np.arange(-5, 11, 1)])  # Set x tick labels

    ax.tick_params(axis='x', labelsize=48)
    ax.tick_params(axis='y', labelsize=48)

    #ax.set_title(rf"$Fr$ = {Fr:.2f}, $\beta$ = {beta:.2f}, $C_d$ = {Cd:.2f}, $\alpha_{{SRH-2D}}$ = {alpha_SRH_2D:.2f}, $\alpha_{{simple}}$ = {alpha_simple:.2f}", fontsize=52)
    if case_ID in [5, 6, 7, 8]:
        ax.set_title(rf"Case {case_ID}: $Fr$ = {Fr:.3f}, half-span, with sediment, Manning's $n$ = {ManningN_SRH_2D:.2f} (SRH-2D), {ManningN_HEC_RAS_2D:.2f} (HEC-RAS)", fontsize=52)
    else: 
        ax.set_title(rf"Case {case_ID}: $Fr$ = {Fr:.3f}, half-span, without sediment, Manning's $n$ = {ManningN_SRH_2D:.2f} (SRH-2D), {ManningN_HEC_RAS_2D:.2f} (HEC-RAS)", fontsize=52)

    
    #add text for flow partition
    plt.text(7, 1.0, 
                f"$\\alpha_{{\\mathrm{{exp.}}}}$ = {alpha_exp:.3f}\n"
                f"$\\alpha_{{\\mathrm{{SRH-2D}}}}$ = {alpha_simulation_SRH_2D:.3f}\n"
                f"$\\alpha_{{\\mathrm{{HEC-RAS\\ 2D}}}}$ = {alpha_simulation_HEC_RAS_2D:.3f}", 
                fontsize=52,          
                color='white',
                ha='left',  # horizontal alignment
                va='center')  # vertical alignment
        
    # Show the customized plot
    plt.tight_layout()

    fig.savefig("exp_vel_mag_contour_ManningN_"+ str(case_ID).zfill(4) +".png", dpi=300, bbox_inches='tight', pad_inches=0)

    #plt.show()
    plt.close()

if __name__ == "__main__":

    case_IDs = [5, 6, 7, 8, 13, 14, 15, 16]
    #case_IDs = [1]

    Frs = [0.077, 0.097, 0.127,	0.153, 0.076, 0.097, 0.126, 0.163]
    ManningN_SRH_2D = [1.26, 1.17, 1.26, 1.5,2.63,2.37,1.78,1.62]
    ManningN_HEC_RAS_2D = [1.31, 1.18, 1.35, 1.49, 2.84,2.34,1.98,1.84]

    betas = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    alphas_exp = [0.779, 0.775,	0.824, 0.832, 0.868, 0.857, 0.846, 0.836]
    alphas_simulation_SRH_2D = [0.779,	0.777,	0.823,	0.831,	0.868,	0.856,	0.846, 0.837]
    alphas_simulation_HEC_RAS_2D = [0.777,	0.7746,	0.82457, 0.8302, 0.869, 0.857, 0.8486, 0.8364]

    #Velocity data from flume experiments
    # First set of values (upstream 72 inches = - 1.83 m)
    U1_case5 = [0.14858,0.18368,0.19842,0.15466,0.1451]    # Experiment 5
    U1_case6 = [0.19166,	0.2339,	0.23624,	0.20236,	0.17762]    # Experiment 6
    U1_case7 = [0.20015,	0.256725,	0.248475,	0.233875,	0.206875]    # Experiment 7
    U1_case8 = [0.252575,	0.318225,	0.30025,	0.29035,	0.26505]    # Experiment 8

    U1_case13 = [0.159, 0.15828, 0.14064, 0.17036, 0.1974]    # Experiment 13
    U1_case14 = [0.2058, 0.197, 0.17086, 0.22468, 0.23558]    # Experiment 14
    U1_case15 = [0.260867, 0.200533, 0.174933, 0.211233, 0.202233]    # Experiment 15
    U1_case16 = [0.346067, 0.2741, 0.2496, 0.2393, 0.216467]  # Experiment 16

    V1_case5 = [0.00524,	0.00378,	0.00052,	-0.00056,	0.00796]    # Experiment 5
    V1_case6 = [0.007,	0.00504,	0.00068,	-0.00076,	0.01062]    # Experiment 6
    V1_case7 = [0.0112,	0.00845,	0.007,	0.005775,	0.0072]    # Experiment 7
    V1_case8 = [-0.0022,	-0.0075,	-0.010025,	-0.008375,	-0.004975]    # Experiment 8

    V1_case13 = [0.01822, 0.0734914, 0.0115, 0.01358, 0.02114]    # Experiment 13
    V1_case14 = [0.01636, 0.01504, 0.01362, 0.0178, 0.02352]    # Experiment 14
    V1_case15 = [0.027733333, 0.013133333, 0.012066667, 0.020966667, 0.0282]    # Experiment 15
    V1_case16 = [0.038433333, 0.025133333, 0.014466667, 0.018266667, 0.0255]  # Experiment 16

    y_positions1 = [0.15, 0.45, 0.75, 1.05, 1.35]  # Adjusted y-positions for 0.3 m spacing

    # Second set of values (upstream 36 inches = - 0.91 m)
    U2_case5 = [0.1211,	0.15958,	0.19286,	0.18032,	0.17058]    # Experiment 5
    U2_case6 = [0.1725,	0.21732,	0.23152,	0.20802,	0.18754]    # Experiment 6
    U2_case7 = [0.1708,	0.240625,	0.255625,	0.252475,	0.229975]    # Experiment 7
    U2_case8 = [0.214825,	0.299425,	0.320125,	0.309875,	0.28145]  # Experiment 8

    U2_case13 = [0.1392, 0.14738, 0.1504, 0.1738, 0.20766]    # Experiment 13
    U2_case14 = [0.18392, 0.18488, 0.18582, 0.22472, 0.25718]    # Experiment 14
    U2_case15 = [0.212133, 0.1949, 0.193467, 0.217833, 0.246067]    # Experiment 15
    U2_case16 = [0.2819, 0.266133, 0.253, 0.2675, 0.270133]  # Experiment 16

    V2_case5 = [-0.00044, -0.00306, -0.00774, -0.00516, 0.00224]    # Experiment 5
    V2_case6 = [-0.00058, -0.00412, -0.01036, -0.0069, 0.00302]    # Experiment 6
    V2_case7 = [-0.003425, -0.020075, -0.0171, -0.0046, 0.0057]    # Experiment 7
    V2_case8 = [-0.01975, -0.04335, -0.035725, -0.016875, -0.005025]  # Experiment 8

    V2_case13 = [0.00802, -0.0048, -0.0066, 0.00118, 0.01444]    # Experiment 13
    V2_case14 = [0.00904, -0.00302, -0.00404, 0.0015, 0.02062]    # Experiment 14
    V2_case15 = [0.0092, -0.007466667, -0.015166667, -0.002433333, 0.016266667]    # Experiment 15
    V2_case16 = [0.0149, -0.014666667, -0.0213, -0.012433333, 0.0179]  # Experiment 16

    y_positions2 = y_positions1  # Same y-positions for consistency

    # Third set of values (only three values, with the first two removed, obstruction 00 inches = 0 m)
    U3_case5 = [0.17144,	0.29882,	0.26494]    # Experiment 5
    U3_case6 = [0.18946,	0.3189,	0.27858]    # Experiment 6
    U3_case7 = [0.20425,	0.327775,	0.27895]    # Experiment 7
    U3_case8 = [0.23555,	0.385775,	0.325625]  # Experiment 8

    U3_case13 = [0.1997, 0.30588, 0.30756]    # Experiment 13
    U3_case14 = [0.2494, 0.38214, 0.39144]    # Experiment 14
    U3_case15 = [0.200967, 0.418833, 0.405733]    # Experiment 15
    U3_case16 = [0.244933, 0.538833, 0.4994]  # Experiment 16

    V3_case5 = [-0.08736, -0.05438, -0.00452]    # Experiment 5
    V3_case6 = [-0.09714, -0.06046, -0.00506]    # Experiment 6
    V3_case7 = [-0.09175, -0.04815, 0.0135]    # Experiment 7
    V3_case8 = [-0.139225, -0.084975, 0.022875]  # Experiment 8

    V3_case13 = [-0.09452, -0.05096, 0.00928]    # Experiment 13
    V3_case14 = [-0.11236, -0.07786, 0.00976]    # Experiment 14
    V3_case15 = [-0.109333333, -0.097666667, 0.006533333]    # Experiment 15
    V3_case16 = [-0.1177, -0.1269, 0.002366667]  # Experiment 16

    y_positions3 = y_positions1[2:]  # Keeping the positions for the remaining three arrows

    # Fourth set of values (downstream 36 inches = 0.91 m)
    U4_case5 = [0.01638,	0.00598,	0.05182,	0.3639,	0.34354]     # Experiment 5
    U4_case6 = [0.0541,	0.05082,	0.10292,	0.39268,	0.36722]     # Experiment 6
    U4_case7 = [0.03425,	0.05055,	0.03485,	0.3822,	0.3297]     # Experiment 7
    U4_case8 = [0.038225,	0.051525,	0.0288,	0.468625,	0.3767]  # Experiment 8

    U4_case13 = [0.03402, 0.03252, 0.01858, 0.36968, 0.38428]     # Experiment 13
    U4_case14 = [0.04048, 0.04312, 0.02374, 0.46876, 0.48606]     # Experiment 14
    U4_case15 = [0.0391, 0.042566667, 0.021133, 0.5168, 0.522]     # Experiment 15
    U4_case16 = [0.062, 0.063933, 0.028133, 0.671933, 0.672167]  # Experiment 16

    V4_case5 = [0.00096, 0.0018, 0.00846, 0.03556, 0.02356]     # Experiment 5
    V4_case6 = [0.00394, 0.0143, 0.01692, 0.04244, 0.02356]     # Experiment 6
    V4_case7 = [-0.002025, 0.016825, 0.0006, 0.062, 0.037]     # Experiment 7
    V4_case8 = [0.004625, -0.0065, -0.004875, 0.0381, 0.036675]  # Experiment 8

    V4_case13 = [-0.00276, -0.01066, -0.00168, 0.03852, 0.04328]     # Experiment 13
    V4_case14 = [-0.0019, -0.01324, -0.0074, 0.0425, 0.05222]     # Experiment 14
    V4_case15 = [-0.0051, -0.0165, -0.008466667, 0.059166667, 0.059833333]     # Experiment 15
    V4_case16 = [-0.009, -0.024666667, -0.010666667, 0.0726, 0.069333333]  # Experiment 16

    y_positions4 = y_positions1  # Same y-positions for consistency

    # Fifth set of values (downstream 72 inches = 1.83 m)
    U5_case5 = [0.03948,	0.00172,	0.07552,	0.35296,	0.36134]     # Experiment 5
    U5_case6 = [0.0658,	0.08538,	0.1802,	0.3842,	0.3612]     # Experiment 6
    U5_case7 = [0.11195,	0.0852,	0.285725,	0.36485,	0.3222]     # Experiment 7
    U5_case8 = [0.1408,	0.095475,	0.313675,	0.442975,	0.375925]  # Experiment 8

    U5_case13 = [0.01384, 0.01244, 0.07918, 0.33488, 0.38092]     # Experiment 13
    U5_case14 = [0.0285, 0.02416, 0.09756, 0.42988, 0.48924]     # Experiment 14
    U5_case15 = [0.0326, 0.033467, 0.111133, 0.4688, 0.529433]     # Experiment 15
    U5_case16 = [0.0514, 0.04067, 0.128533, 0.6009, 0.6785]  # Experiment 16

    V5_case5 = [-0.00306, -0.00348, -0.00192, 0.0148, 0.00072]     # Experiment 5
    V5_case6 = [-0.00706, 0.00814, 0.02328, 0.02978, 0.01842]     # Experiment 6
    V5_case7 = [0.02535, 0.026325, 0.08505, 0.061275, 0.032]     # Experiment 7
    V5_case8 = [0.0439, 0.02375, 0.076675, 0.05305, 0.0179]  # Experiment 8

    V5_case13 = [-0.01034, -0.015, 0.00512, 0.04126, 0.04626]     # Experiment 13
    V5_case14 = [-0.00714, -0.02282, 0.00286, 0.06278, 0.0606]     # Experiment 14
    V5_case15 = [-0.0046, -0.012266667, 0.0052, 0.067833333, 0.0703]     # Experiment 15
    V5_case16 = [-0.003866667, -0.015866667, -0.004066667, 0.0838, 0.090966667]  # Experiment 16

    y_positions5 = y_positions1  # Same y-positions for consistency

    #put all the exp. data together
    U1s = [U1_case5, U1_case6, U1_case7, U1_case8, U1_case13, U1_case14, U1_case15, U1_case16]
    U2s = [U2_case5, U2_case6, U2_case7, U2_case8, U2_case13, U2_case14, U2_case15, U2_case16]
    U3s = [U3_case5, U3_case6, U3_case7, U3_case8, U3_case13, U3_case14, U3_case15, U3_case16]
    U4s = [U4_case5, U4_case6, U4_case7, U4_case8, U4_case13, U4_case14, U4_case15, U4_case16]
    U5s = [U5_case5, U5_case6, U5_case7, U5_case8, U5_case13, U5_case14, U5_case15, U5_case16]

    V1s = [V1_case5, V1_case6, V1_case7, V1_case8, V1_case13, V1_case14, V1_case15, V1_case16]
    V2s = [V2_case5, V2_case6, V2_case7, V2_case8, V2_case13, V2_case14, V2_case15, V2_case16]
    V3s = [V3_case5, V3_case6, V3_case7, V3_case8, V3_case13, V3_case14, V3_case15, V3_case16]
    V4s = [V4_case5, V4_case6, V4_case7, V4_case8, V4_case13, V4_case14, V4_case15, V4_case16]
    V5s = [V5_case5, V5_case6, V5_case7, V5_case8, V5_case13, V5_case14, V5_case15, V5_case16]

    x_positions = [-1.83, -0.91, 0, 0.91, 1.83]
    y_positions = [y_positions1, y_positions2, y_positions3, y_positions4, y_positions5]

    #dimensions of the LWD
    rect_x, rect_y = -0.1, 0  # Bottom-left corner of the rectangle
    rect_width =  0.2 # Width of the rectangle
    rect_height = 0.75 # Height of the rectangle

    for i, case_ID in enumerate(case_IDs):
        print("plotting case_ID = ", case_ID)

        vtkFileName_SRH_2D = "Exp_"+str(case_ID)+"_n/SRH2D_Exp_"+str(case_ID)+"_ManningN_C_0003.vtk"
        vtkFileName_HEC_RAS_2D = "Exp_"+str(case_ID)+"_n/RAS_2D_"+str(case_ID).zfill(2)+".vtk"
        plot_contour_from_vtk(case_ID, Frs[i], betas[i], ManningN_SRH_2D[i], ManningN_HEC_RAS_2D[i], alphas_exp[i], alphas_simulation_SRH_2D[i], alphas_simulation_HEC_RAS_2D[i], rect_x, rect_y, rect_width, rect_height, vtkFileName_SRH_2D, vtkFileName_HEC_RAS_2D, [U1s[i], U2s[i], U3s[i], U4s[i], U5s[i]], [V1s[i], V2s[i], V3s[i], V4s[i], V5s[i]], x_positions, y_positions)

    print("All done!")
