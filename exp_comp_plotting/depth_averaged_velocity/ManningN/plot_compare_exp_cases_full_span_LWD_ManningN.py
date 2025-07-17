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

def plot_contour_from_vtk(case_ID, Fr, beta, ManningN, alpha_exp, alpha_simulation, rect_x, rect_y, rect_width, rect_height, vtkFileName,
                          U_all, x_positions, y_positions):

    #load data from vtk file: water depht and velocity

    if not os.path.exists(vtkFileName):
        return None

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
    Velocity_m_p_s = data.GetCellData().GetArray("Velocity_m_p_s")
    Vel_Mag_m_p_s = data.GetCellData().GetArray("Vel_Mag_m_p_s")
    Water_Depth_m = data.GetCellData().GetArray("Water_Depth_m")

    if Velocity_m_p_s is None:
        raise ValueError("No Velocity_m_p_s data found at cell centers. Please check your VTK file.")

    # Convert data to a numpy array
    Vel_x_np = np.array([-Velocity_m_p_s.GetTuple3(i)[0] for i in range(Velocity_m_p_s.GetNumberOfTuples())])  #flip x velocity
    Vel_y_np = np.array([ Velocity_m_p_s.GetTuple3(i)[1] for i in range(Velocity_m_p_s.GetNumberOfTuples())])  # flip x velocity

    Vel_mag_np = np.array([Vel_Mag_m_p_s.GetTuple1(i) for i in range(Vel_Mag_m_p_s.GetNumberOfTuples())])

    # Check if points and scalars have compatible shapes
    if len(points) != len(Vel_mag_np):
        raise ValueError("Mismatch between number of cell centers and scalar values.")

    # Create a grid for contour plotting
    x = points[:, 0]
    y = points[:, 1]
    z = Vel_mag_np

    # Create a regular grid interpolated from the scattered data
    xi = np.linspace(x.min()+0.01, x.max()-0.01, 420)
    yi = np.linspace(y.min()+0.01, y.max()-0.01, 60)
    X, Y = np.meshgrid(xi, yi)

    # Interpolate the scalar field onto the grid
    Z = griddata(points, z, (X, Y), method="linear")

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
        #full span cases, no LWD profile
        if iProfile == 2:
            continue

        #How many points of velocity measurement on the profile        
        nPoints = 5

        coords_at_points = np.zeros((nPoints,2))

        vel_x_SRH_2D_at_points = np.zeros(nPoints)
        vel_y_SRH_2D_at_points = np.zeros(nPoints)

        for iPoint in range(nPoints):
            coords_at_points[iPoint,0] = x_positions[iProfile]
            coords_at_points[iPoint,1] = y_positions[iProfile][iPoint]

        interpolated_SRH_2D_velocity_x = griddata((x, y), Vel_x_np, coords_at_points, method='linear')
        interpolated_SRH_2D_velocity_y = griddata((x, y), Vel_y_np, coords_at_points, method='linear')

        # Plot the profile
        for iPoint in range(nPoints):
            #plot velocity from experiment
            plt.arrow(coords_at_points[iPoint, 0], coords_at_points[iPoint, 1], U_all[iProfile][iPoint], 0, linewidth=2, head_width=0.04, head_length=0.02, fc='blue', ec='blue', length_includes_head=True)
            plt.plot(coords_at_points[iPoint, 0] + U_all[iProfile][iPoint] + 0.01, coords_at_points[iPoint, 1], 'ko', markersize=4)  # Smaller dot (size 4) to the right of the arrow tip

            #plot SRH-2D velocity
            plt.arrow(coords_at_points[iPoint, 0], coords_at_points[iPoint, 1],
                      interpolated_SRH_2D_velocity_x[iPoint], interpolated_SRH_2D_velocity_y[iPoint],
                      head_width=0.04, head_length=0.02,fc='yellow', ec='yellow',
                      linestyle="--", linewidth=2, length_includes_head=True)

        #plot a velocity vector scale
        plt.arrow(-4, 0.75, 0.5, 0.0,
                  head_width=0.04, head_length=0.02, fc='black', ec='black',
                  linewidth=2, length_includes_head=True)

        plt.text(-3.75, 0.8, "0.5 m/s", color="black", fontsize=36, horizontalalignment='center',)

        dot_x_positions1 = [x_positions[iProfile] + value + 0.01 for value in U_all[iProfile]]
        plt.plot(dot_x_positions1, y_positions[iProfile], 'k-', linewidth=1)  # Curve connecting the dots for profile
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
    if case_ID in [1, 2, 3, 4]:
        ax.set_title(rf"Case {case_ID}: $Fr$ = {Fr:.3f}, full-span, with sediment, Manning's $n$ = {ManningN:.1f}", fontsize=52)
    else: 
        ax.set_title(rf"Case {case_ID}: $Fr$ = {Fr:.3f}, full-span, without sediment, Manning's $n$ = {ManningN:.1f}", fontsize=52)

    # Show the customized plot
    plt.tight_layout()

    fig.savefig("exp_vel_mag_contour_ManningN_"+ str(case_ID).zfill(4) +".png", dpi=300, bbox_inches='tight', pad_inches=0)

    #plt.show()
    plt.close()

if __name__ == "__main__":

    case_IDs = [1, 2, 3, 4, 9, 10, 11, 12]
    #case_IDs = [1]

    Frs = [0.070, 0.097, 0.109, 0.137, 0.064, 0.077, 0.090, 0.099]
    ManningN = [2.1,2.2,2.5,2.6,2.4,2.36,1.82,1.97]
    betas = [1, 1, 1, 1, 1, 1, 1, 1]
    alphas_exp = [1, 1, 1, 1, 1, 1, 1, 1]
    alphas_simulation = [1, 1, 1, 1, 1, 1, 1, 1]

    #Velocity data from flume experiments
    # First set of values (upstream 72 inches = - 1.83 m)
    U1_case1 = [0.14261,	0.167212,	0.171524,	0.13854,	0.12671]    # Experiment 1
    U1_case2 = [0.166,	0.207026,	0.182372,	0.15782,	0.163836]    # Experiment 2
    U1_case3 = [0.191728,	0.231102,	0.231772,	0.20023,	0.17826]    # Experiment 3
    U1_case4 = [0.241426,	0.28685,	0.28455,	0.247868,	0.235218]    # Experiment 4

    U1_case9 = [0.11944,	0.13152,	0.14338,	0.16584,	0.17798]    # Experiment 9
    U1_case10 = [0.15754,	0.16082,	0.1663,	0.1906,	0.2099]    # Experiment 10
    U1_case11 = [0.14965,	0.2024,	0.17525,	0.170575,	0.171925]    # Experiment 11
    U1_case12 = [0.21215,	0.223175,	0.176525,	0.195125,	0.2293]  # Experiment 12
    y_positions1 = [0.15, 0.45, 0.75, 1.05, 1.35]  # Adjusted y-positions for 0.3 m spacing

    # Second set of values (upstream 36 inches = - 0.91 m)
    U2_case1 = [0.151618,	0.165568,	0.165634,	0.140464,	0.141536]    # Experiment 1
    U2_case2 = [0.163952,	0.188316,	0.1946724,	0.16495,	0.176212]    # Experiment 2
    U2_case3 = [0.185718,	0.22262,	0.228336,	0.198382,	0.182808]    # Experiment 3
    U2_case4 = [0.236266,	0.280256,	0.277282,	0.248532,	0.233428]    # Experiment 4

    U2_case9 = [0.12958,	0.1318,	0.143,	0.1581,	0.17256]    # Experiment 9
    U2_case10 = [0.16134,	0.16494,	0.16844,	0.18912,	0.20066]    # Experiment 10
    U2_case11 = [0.152675,	0.18415,	0.187375,	0.17725,	0.1705]    # Experiment 11
    U2_case12 = [0.207875,	0.217025,	0.18825,	0.19465,	0.219]  # Experiment 12
    y_positions2 = y_positions1  # Same y-positions for consistency

    # No values (obstruction 00 inches = 0 m)
    U3_case1 = []    # Experiment 1
    U3_case2 = []    # Experiment 2
    U3_case3 = []    # Experiment 3
    U3_case4 = []    # Experiment 4

    U3_case9 = []     # Experiment 9
    U3_case10 = []    # Experiment 10
    U3_case11 = []    # Experiment 11
    U3_case12 = []    # Experiment 12
    y_positions3 = y_positions1[2:]  # Keeping the positions for the remaining three arrows

    # Fourth set of values (downstream 36 inches = 0.91 m)
    U4_case1 = [0.227514,	0.122426,	0.152292,	0.147178,	0.12971]     # Experiment 1
    U4_case2 = [0.188426,	0.191488,	0.226714,	0.20305,	0.210904]     # Experiment 2
    U4_case3 = [0.388133333,	0.330696667,	0.30288,	0.35405,	0.406216667]     # Experiment 3
    U4_case4 = [0.387103333,	0.374716667,	0.338936667,	0.391886667,	0.41705]  # Experiment 4

    U4_case9 = [0.19575,	0.17175,	0.3411,	0.041975,	0.1479]     # Experiment 9
    U4_case10 = [0.261425,	0.2304,	0.407325,	0.097225,	0.190125]     # Experiment 10
    U4_case11 = [0.3185,	0.1935,	0.4699,	0.06495,	0.29185]     # Experiment 11
    U4_case12 = [0.36245,	0.3552,	0.6269,	0.1068,	0.37975]  # Experiment 12
    y_positions4 = y_positions1  # Same y-positions for consistency

    # Fifth set of values (downstream 72 inches = 1.83 m)
    U5_case1 = [0.21029,	0.134754,	0.135922,	0.13806,	0.13243]     # Experiment 1
    U5_case2 = [0.23897,	0.187722,	0.204544,	0.225228,	0.175986]     # Experiment 2
    U5_case3 = [0.2953275,	0.3168725,	0.29459,	0.2684375,	0.2377875]     # Experiment 3
    U5_case4 = [0.435363333,	0.442693333,	0.431893333,	0.4314,	0.46438]  # Experiment 4

    U5_case9 = [0.184125,	0.19195,	0.263975,	0.080025,	0.151175]     # Experiment 9
    U5_case10 = [0.243375,	0.260175,	0.2987,	0.1542,	0.2273]     # Experiment 10
    U5_case11 = [0.4537,	0.33265,	0.3663,	0.1012,	0.2166]     # Experiment 11
    U5_case12 = [0.53205,	0.39485,	0.49465,	0.1158,	0.3152]  # Experiment 12
    y_positions5 = y_positions1  # Same y-positions for consistency

    #put all the exp. data together
    U1s = [U1_case1, U1_case2, U1_case3, U1_case4, U1_case9, U1_case10, U1_case11, U1_case12]
    U2s = [U2_case1, U2_case2, U2_case3, U2_case4, U2_case9, U2_case10, U2_case11, U2_case12]
    U3s = [U3_case1, U3_case2, U3_case3, U3_case4, U3_case9, U3_case10, U3_case11, U3_case12]
    U4s = [U4_case1, U4_case2, U4_case3, U4_case4, U4_case9, U4_case10, U4_case11, U4_case12]
    U5s = [U5_case1, U5_case2, U5_case3, U5_case4, U5_case9, U5_case10, U5_case11, U5_case12]
    x_positions = [-1.83, -0.91, 0, 0.91, 1.83]
    y_positions = [y_positions1, y_positions2, y_positions3, y_positions4, y_positions5]

    #dimensions of the LWD
    rect_x, rect_y = -0.1, 0  # Bottom-left corner of the rectangle
    rect_width =  0.2 # Width of the rectangle
    rect_height = 1.5 # Height of the rectangle

    for i, case_ID in enumerate(case_IDs):
        print("plotting case_ID = ", case_ID)

        vtkFileName = "Exp_"+str(case_ID)+"_n/SRH2D_Exp_"+str(case_ID)+"_ManningN_C_0003"".vtk"
        plot_contour_from_vtk(case_ID, Frs[i], betas[i], ManningN[i], alphas_exp[i], alphas_simulation[i], rect_x, rect_y, rect_width, rect_height, vtkFileName,
                              [U1s[i], U2s[i], U3s[i], U4s[i], U5s[i]], x_positions, y_positions)

    print("All done!")
