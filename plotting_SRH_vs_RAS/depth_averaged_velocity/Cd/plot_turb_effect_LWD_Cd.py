#Plot the effect of turbulence model on the result of Cd approach

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import griddata, interp2d

import matplotlib.ticker as tick
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
        WSE_m = data.GetCellData().GetArray("Water_Elev_m")

    elif SRH_or_RAS == "HEC_RAS_2D":
        Velocity_m_p_s = data.GetCellData().GetArray("Velocity_cell_m_p_s")
        #Vel_Mag_m_p_s = data.GetCellData().GetArray("Vel_Mag_m_p_s")
        Water_Depth_m = data.GetCellData().GetArray("Water_Depth_m")
        WSE_m = data.GetCellData().GetArray("Water_Elev_m")

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
    
    return points, Vel_x_np, Vel_y_np, Vel_mag_np, Water_Depth_m, WSE_m

def plot_contour_from_vtk(SRH_or_RAS, case_ID, Fr, beta, rect_x, rect_y, rect_width, rect_height, vtkFileName_w_turb, vtkFileName_wo_turb):
    """
    The plot has 3 rows and 3 columns: 
      - First column is to plot the contours of U, V, and WSE simulated by SRH-2D with turbulence model
      - Second column is to plot the contours of U, V, and WSE simulated by SRH-2D without turbulence model
      - Third column is to plot the difference of U, V, and WSE between SRH-2D with and without turbulence model
    """

    cmap = "viridis"

    #load data from vtk file: water depht and velocity

    if not os.path.exists(vtkFileName_w_turb) or not os.path.exists(vtkFileName_wo_turb):
        print(f"File {vtkFileName_w_turb} or {vtkFileName_wo_turb} does not exist.")
        return None

    points_w_turb, Vel_x_np_w_turb, Vel_y_np_w_turb, Vel_mag_np_w_turb, Water_Depth_m_w_turb, WSE_m_w_turb = extract_data_from_vtk(vtkFileName_w_turb, SRH_or_RAS)
    points_wo_turb, Vel_x_np_wo_turb, Vel_y_np_wo_turb, Vel_mag_np_wo_turb, Water_Depth_m_wo_turb, WSE_m_wo_turb = extract_data_from_vtk(vtkFileName_wo_turb, SRH_or_RAS)

     # Debug: Check the data before interpolation
    print(f"points_w_turb shape: {points_w_turb.shape if points_w_turb is not None else 'None'}")
    print(f"Vel_x_np_w_turb shape: {Vel_x_np_w_turb.shape if Vel_x_np_w_turb is not None else 'None'}")
    print(f"points_w_turb range: X[{points_w_turb[:, 0].min():.3f}, {points_w_turb[:, 0].max():.3f}], Y[{points_w_turb[:, 1].min():.3f}, {points_w_turb[:, 1].max():.3f}]")
    print(f"Vel_x_np_w_turb range: [{Vel_x_np_w_turb.min():.3f}, {Vel_x_np_w_turb.max():.3f}]")
    print(f"Any NaN in points_w_turb: {np.any(np.isnan(points_w_turb))}")
    print(f"Any NaN in Vel_x_np_w_turb: {np.any(np.isnan(Vel_x_np_w_turb))}")


    # Create a grid for contour plotting
    x = points_w_turb[:, 0]
    y = points_w_turb[:, 1]
    
    x_wo_turb = points_wo_turb[:, 0]
    y_wo_turb = points_wo_turb[:, 1]    

    # Create a regular grid interpolated from the scattered data
    xi = np.linspace(x.min()+0.01, x.max()-0.01, 420)
    yi = np.linspace(y.min()+0.01, y.max()-0.01, 60)
    X, Y = np.meshgrid(xi, yi)

     # Debug: Check the grid
    print(f"Grid range: X[{X.min():.3f}, {X.max():.3f}], Y[{Y.min():.3f}, {Y.max():.3f}]")
    print(f"Grid shape: {X.shape}")

    # Interpolate the scalar field onto the grid
    vel_x_w_turb = griddata(points_w_turb, Vel_x_np_w_turb, (X, Y), method="nearest")
    vel_y_w_turb = griddata(points_w_turb, Vel_y_np_w_turb, (X, Y), method="nearest")
    wse_w_turb = griddata(points_w_turb, WSE_m_w_turb, (X, Y), method="nearest")

    vel_x_wo_turb = griddata(points_wo_turb, Vel_x_np_wo_turb, (X, Y), method="nearest")
    vel_y_wo_turb = griddata(points_wo_turb, Vel_y_np_wo_turb, (X, Y), method="nearest")
    wse_wo_turb = griddata(points_wo_turb, WSE_m_wo_turb, (X, Y), method="nearest")

     # Debug: Check the result
    print(f"vel_x_w_turb shape: {vel_x_w_turb.shape}")
    print(f"vel_x_w_turb range: [{vel_x_w_turb.min():.3f}, {vel_x_w_turb.max():.3f}]")
    print(f"NaN count in vel_x_w_turb: {np.sum(np.isnan(vel_x_w_turb))}")
    print(f"Total points in vel_x_w_turb: {vel_x_w_turb.size}")

    #compute the difference of vel_x, vel_y, and wse between w_turb and wo_turb
    vel_x_diff = vel_x_w_turb - vel_x_wo_turb
    vel_y_diff = vel_y_w_turb - vel_y_wo_turb
    wse_diff = wse_w_turb - wse_wo_turb

    #compute the range of vel_x, vel_y, and wse
    vmin_vel_x_w_turb = vel_x_w_turb.min()
    vmax_vel_x_w_turb = vel_x_w_turb.max()

    vmin_vel_y_w_turb = vel_y_w_turb.min()
    vmax_vel_y_w_turb = vel_y_w_turb.max()

    vmin_wse_w_turb = wse_w_turb.min()
    vmax_wse_w_turb = wse_w_turb.max()
    
    vmin_vel_x_wo_turb = vel_x_wo_turb.min()
    vmax_vel_x_wo_turb = vel_x_wo_turb.max()

    vmin_vel_y_wo_turb = vel_y_wo_turb.min()
    vmax_vel_y_wo_turb = vel_y_wo_turb.max()

    vmin_wse_wo_turb = wse_wo_turb.min()
    vmax_wse_wo_turb = wse_wo_turb.max()

    vmin_vel_x_diff = vel_x_diff.min()
    vmax_vel_x_diff = vel_x_diff.max()

    vmin_vel_y_diff = vel_y_diff.min()
    vmax_vel_y_diff = vel_y_diff.max()

    vmin_wse_diff = wse_diff.min()
    vmax_wse_diff = wse_diff.max()  

    #print the difference of vel_x, vel_y, and wse
    print(f"vmin_vel_x_diff = {vmin_vel_x_diff}, vmax_vel_x_diff = {vmax_vel_x_diff}")
    print(f"vmin_vel_y_diff = {vmin_vel_y_diff}, vmax_vel_y_diff = {vmax_vel_y_diff}")
    print(f"vmin_wse_diff = {vmin_wse_diff}, vmax_wse_diff = {vmax_wse_diff}")

    #plot the contours
    fig, axs = plt.subplots(3, 3, figsize=(18, 4))
    
    # Row and column labels
    row_labels = ['u (m/s)', 'v (m/s)', 'WSE (m)']
    col_labels = ['With Turbulence', 'Without Turbulence', 'Difference (With - Without)']
    
    # Plot with turbulence (first column)
    # U-velocity with turbulence
    im1 = axs[0, 0].contourf(X, Y, vel_x_w_turb, levels=20, cmap=cmap, 
                               vmin=vmin_vel_x_w_turb, vmax=vmax_vel_x_w_turb)
    axs[0, 0].set_title(f'{row_labels[0]} - {col_labels[0]}', fontsize=14)
    axs[0, 0].set_ylabel('Y (m)', fontsize=14)
    axs[0, 0].set_xticks([])
    axs[0, 0].tick_params(axis='y', which='major', labelsize=14)
    #plt.colorbar(im1, ax=axs[0, 0], format='%.3f', shrink=0.8, aspect=20)

    divider = make_axes_locatable(axs[0, 0])
    cax = divider.append_axes("right", size="3%", pad=0.2)
    clb = fig.colorbar(im1, ticks=np.linspace(vmin_vel_x_w_turb, vmax_vel_x_w_turb, 5), cax=cax, shrink=1.2)
    clb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb.ax.tick_params(labelsize=12)

    # Get the subplot's position in figure coordinates
    # bbox = axs[0, 0].get_position()

    # # Make a colorbar axes that’s 120% as tall, nudged down a bit
    # cax = fig.add_axes([
    #     bbox.x1 + 0.01,           # left
    #     bbox.y0 - 0.10*bbox.height,  # bottom (shift down)
    #     0.02,                     # width
    #     1.20*bbox.height          # height (120% of the subplot)
    # ])
    # clb = fig.colorbar(im1, ticks=np.linspace(vmin_vel_x_w_turb, vmax_vel_x_w_turb, 5), cax=cax)
    # clb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    # clb.ax.tick_params(labelsize=12)

    # Make colorbar taller by adjusting its position
    cax_pos = cax.get_position()
    cax.set_position([cax_pos.x0, cax_pos.y0 - 0.5, cax_pos.width, cax_pos.height + 0.5])
    
    # V-velocity with turbulence
    im2 = axs[1, 0].contourf(X, Y, vel_y_w_turb, levels=20, cmap=cmap,
                               vmin=vmin_vel_y_w_turb, vmax=vmax_vel_y_w_turb)
    axs[1, 0].set_title(f'{row_labels[1]} - {col_labels[0]}', fontsize=14)
    axs[1, 0].set_ylabel('Y (m)', fontsize=14)
    axs[1, 0].set_xticks([])
    axs[1, 0].tick_params(axis='y', which='major', labelsize=14)
    #plt.colorbar(im2, ax=axs[1, 0], format='%.3f', shrink=0.8, aspect=20)

    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes("right", size="3%", pad=0.2)
    clb = fig.colorbar(im2, ticks=np.linspace(vmin_vel_y_w_turb, vmax_vel_y_w_turb, 5), cax=cax)
    clb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb.ax.tick_params(labelsize=12)
    
    # WSE with turbulence
    im3 = axs[2, 0].contourf(X, Y, wse_w_turb, levels=20, cmap=cmap,
                               vmin=vmin_wse_w_turb, vmax=vmax_wse_w_turb)
    axs[2, 0].set_title(f'{row_labels[2]} - {col_labels[0]}', fontsize=14)
    axs[2, 0].set_xlabel('X (m)', fontsize=14)
    axs[2, 0].set_ylabel('Y (m)', fontsize=14)
    axs[2, 0].tick_params(axis='both', which='major', labelsize=14)
    #plt.colorbar(im3, ax=axs[2, 0], format='%.3f', shrink=0.8, aspect=20)
    
    divider = make_axes_locatable(axs[2, 0])
    cax = divider.append_axes("right", size="3%", pad=0.2)
    clb = fig.colorbar(im3, ticks=np.linspace(vmin_wse_w_turb, vmax_wse_w_turb, 5), cax=cax)
    clb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb.ax.tick_params(labelsize=12)
    
    # Plot without turbulence (second column)
    # U-velocity without turbulence
    im4 = axs[0, 1].contourf(X, Y, vel_x_wo_turb, levels=20, cmap=cmap,
                               vmin=vmin_vel_x_wo_turb, vmax=vmax_vel_x_wo_turb)
    axs[0, 1].set_title(f'{row_labels[0]} - {col_labels[1]}', fontsize=14)
    #axs[0, 1].tick_params(axis='both', which='major', labelsize=10)
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    #plt.colorbar(im4, ax=axs[0, 1], format='%.3f', shrink=0.8, aspect=20)

    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes("right", size="3%", pad=0.2)
    clb = fig.colorbar(im4, ticks=np.linspace(vmin_vel_x_wo_turb, vmax_vel_x_wo_turb, 5), cax=cax)
    clb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb.ax.tick_params(labelsize=12)
    
    # V-velocity without turbulence
    im5 = axs[1, 1].contourf(X, Y, vel_y_wo_turb, levels=20, cmap=cmap,
                               vmin=vmin_vel_y_wo_turb, vmax=vmax_vel_y_wo_turb)
    axs[1, 1].set_title(f'{row_labels[1]} - {col_labels[1]}', fontsize=14)
    #axs[1, 1].tick_params(axis='both', which='major', labelsize=10)
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    #plt.colorbar(im5, ax=axs[1, 1], format='%.3f', shrink=0.8, aspect=20)

    divider = make_axes_locatable(axs[1, 1])
    cax = divider.append_axes("right", size="3%", pad=0.2)
    clb = fig.colorbar(im5, ticks=np.linspace(vmin_vel_y_wo_turb, vmax_vel_y_wo_turb, 5), cax=cax)
    clb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb.ax.tick_params(labelsize=12)
    
    # WSE without turbulence
    im6 = axs[2, 1].contourf(X, Y, wse_wo_turb, levels=20, cmap=cmap,
                               vmin=vmin_wse_wo_turb, vmax=vmax_wse_wo_turb)
    axs[2, 1].set_title(f'{row_labels[2]} - {col_labels[1]}', fontsize=10)
    axs[2, 1].set_xlabel('X (m)', fontsize=14)
    axs[2, 1].set_yticks([])
    axs[2, 1].tick_params(axis='x', which='major', labelsize=14)
    #plt.colorbar(im6, ax=axs[2, 1], format='%.3f', shrink=0.8, aspect=20)

    divider = make_axes_locatable(axs[2, 1])
    cax = divider.append_axes("right", size="3%", pad=0.2)
    clb = fig.colorbar(im6, ticks=np.linspace(vmin_wse_wo_turb, vmax_wse_wo_turb, 5), cax=cax)
    clb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb.ax.tick_params(labelsize=12)
    
    # Plot differences (third column)
    # U-velocity difference
    im7 = axs[0, 2].contourf(X, Y, vel_x_diff, levels=20, cmap=cmap,
                               vmin=vmin_vel_x_diff, vmax=vmax_vel_x_diff)
    axs[0, 2].set_title(f'{row_labels[0]} - {col_labels[2]}', fontsize=14)
    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])
    #plt.colorbar(im7, ax=axs[0, 2], format='%.3f', shrink=0.8, aspect=20)

    divider = make_axes_locatable(axs[0, 2])
    cax = divider.append_axes("right", size="3%", pad=0.2)
    clb = fig.colorbar(im7, ticks=np.linspace(vmin_vel_x_diff, vmax_vel_x_diff, 5), cax=cax)
    clb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
    clb.ax.tick_params(labelsize=12)
    
    # V-velocity difference
    im8 = axs[1, 2].contourf(X, Y, vel_y_diff, levels=20, cmap=cmap,
                               vmin=vmin_vel_y_diff, vmax=vmax_vel_y_diff)
    axs[1, 2].set_title(f'{row_labels[1]} - {col_labels[2]}', fontsize=14)
    axs[1, 2].set_xticks([])
    axs[1, 2].set_yticks([])
    #plt.colorbar(im8, ax=axs[1, 2], format='%.3f', shrink=0.8, aspect=20)

    divider = make_axes_locatable(axs[1, 2])
    cax = divider.append_axes("right", size="3%", pad=0.2)
    clb = fig.colorbar(im8, ticks=np.linspace(vmin_vel_y_diff, vmax_vel_y_diff, 5), cax=cax)
    clb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
    clb.ax.tick_params(labelsize=12)
    
    # WSE difference
    im9 = axs[2, 2].contourf(X, Y, wse_diff, levels=20, cmap=cmap,
                               vmin=vmin_wse_diff, vmax=vmax_wse_diff)
    axs[2, 2].set_title(f'{row_labels[2]} - {col_labels[2]}', fontsize=14)
    axs[2, 2].set_xlabel('X (m)', fontsize=14)
    axs[2, 2].set_yticks([])
    axs[2, 2].tick_params(axis='x', which='major', labelsize=14)
    #plt.colorbar(im9, ax=axs[2, 2], format='%.3f', shrink=0.8, aspect=20)

    divider = make_axes_locatable(axs[2, 2])
    cax = divider.append_axes("right", size="3%", pad=0.2)
    clb = fig.colorbar(im9, ticks=np.linspace(vmin_wse_diff, vmax_wse_diff, 5), cax=cax)
    clb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
    clb.ax.tick_params(labelsize=12)
    
    # Add LWD rectangle to all plots
    for i in range(3):
        for j in range(3):            
            rect = plt.Rectangle((rect_x, rect_y), rect_width, rect_height, 
                                linewidth=1, linestyle='--', edgecolor='red', facecolor='none', alpha=0.8)
            axs[i, j].add_patch(rect)
            axs[i, j].set_aspect('equal')
            axs[i, j].grid(True, alpha=0.3)
    
    # Add overall title
    if SRH_or_RAS == "SRH_2D":
        if case_ID in [1, 2, 3, 4, 5, 6, 7, 8]: #with sediment
            if case_ID in [1, 2, 3, 4]: #full span
                fig.suptitle(f'Case {case_ID} with SRH-2D (Drag approach): Fr = {Fr:.3f}, with sediment, full span', 
                            fontsize=14, y=0.98)
            else: #half span
                fig.suptitle(f'Case {case_ID} with SRH-2D (Drag approach): Fr = {Fr:.3f}, with sediment, half span', 
                            fontsize=14, y=0.98)
        else: #no sediment
            if case_ID in [9, 10, 11, 12]: #full span
                fig.suptitle(f'Case {case_ID} with SRH-2D (Drag approach): Fr = {Fr:.3f}, no sediment, full span', 
                            fontsize=14, y=0.98)
            else: #half span
                fig.suptitle(f'Case {case_ID} with SRH-2D (Drag approach): Fr = {Fr:.3f}, no sediment, half span', 
                            fontsize=14, y=0.98)
    elif SRH_or_RAS == "HEC_RAS_2D":
        if case_ID in [1, 2, 3, 4, 5, 6, 7, 8]: #with sediment
            if case_ID in [1, 2, 3, 4]: #full span
                fig.suptitle(f'Case {case_ID} with HEC-RAS 2D (Drag approach): Fr = {Fr:.3f}, with sediment, full span', 
                            fontsize=14, y=0.98)
            else: #half span
                fig.suptitle(f'Case {case_ID} with HEC-RAS 2D (Drag approach): Fr = {Fr:.3f}, with sediment, half span', 
                            fontsize=14, y=0.98)
        else: #no sediment
            if case_ID in [9, 10, 11, 12]: #full span
                fig.suptitle(f'Case {case_ID} with HEC-RAS 2D (Drag approach): Fr = {Fr:.3f}, no sediment, full span', 
                            fontsize=14, y=0.98)
            else: #half span
                fig.suptitle(f'Case {case_ID} with HEC-RAS 2D (Drag approach): Fr = {Fr:.3f}, no sediment, half span', 
                            fontsize=14, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_filename = f'turb_effect_case_{case_ID}_Cd_{SRH_or_RAS}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_filename}")
    
    # Show the plot
    #plt.show()

    plt.close()
    
    #return fig

if __name__ == "__main__":

    #define data 
    case_IDs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    #case_IDs = [9]

    Frs = [0.070, 0.097, 0.109, 0.137, 0.077, 0.097, 0.127,	0.153, 0.064, 0.077, 0.090, 0.099, 0.076, 0.097, 0.126, 0.163]
    
    #ManningN_SRH_2D = [2.13, 2.16, 2.66, 2.72, 1.26, 1.17, 1.26, 1.5, 2.42, 2.36, 1.92, 1.98, 2.63,2.37,1.78,1.62]
    #ManningN_HEC_RAS_2D = [2.24, 2.24, 2.66, 2.72, 1.31, 1.18, 1.35, 1.49, 2.56, 2.42, 1.91, 1.93, 2.84,2.34,1.98,1.84]
    
    betas = [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5]
    alphas_exp = [1, 1, 1, 1, 0.779, 0.775, 0.824, 0.832, 1, 1, 1, 1, 0.868, 0.857, 0.846, 0.836]
    alphas_simulation_SRH_2D = [1, 1, 1, 1, 0.779, 0.777, 0.823, 0.831, 1, 1, 1, 1, 0.868, 0.856, 0.846, 0.837]
    alphas_simulation_HEC_RAS_2D = [1, 1, 1, 1, 0.777, 0.7746, 0.82457, 0.8302, 1, 1, 1, 1, 0.869, 0.857, 0.8486, 0.8364]

    
    #dimensions of the LWD
    rect_x, rect_y = -0.1, 0  # Bottom-left corner of the rectangle
    rect_width =  0.2 # Width of the rectangle    

    for i, case_ID in enumerate(case_IDs):
        print("plotting case_ID = ", case_ID)

        if case_ID in [1, 2, 3, 4, 9, 10, 11, 12]: #full span
            rect_height = 1.5 # Height of the rectangle
        else: #half span
            rect_height = 0.75 # Height of the rectangle

        #plot SRH-2D results
        vtkFileName_w_turb = "Exp_"+str(case_ID)+"_Cd/SRH2D_Exp_"+str(case_ID)+"_Cd_C_0003.vtk"
        vtkFileName_wo_turb = "Exp_"+str(case_ID)+"_Cd/SRH2D_Exp_"+str(case_ID)+"_Cd_C_0003_wo_turb.vtk"

        SRH_or_RAS = "SRH_2D"
        plot_contour_from_vtk(SRH_or_RAS, case_ID, Frs[i], betas[i], rect_x, rect_y, rect_width, rect_height, vtkFileName_w_turb, vtkFileName_wo_turb)

        #plot HEC-RAS 2D results
        vtkFileName_w_turb = "Exp_"+str(case_ID)+"_Cd/RAS_2D_"+str(case_ID).zfill(2)+".vtk"
        vtkFileName_wo_turb = "Exp_"+str(case_ID)+"_Cd/RAS_2D_"+str(case_ID).zfill(2)+"_wo_turb.vtk"
        SRH_or_RAS = "HEC_RAS_2D"
        plot_contour_from_vtk(SRH_or_RAS, case_ID, Frs[i], betas[i], rect_x, rect_y, rect_width, rect_height, vtkFileName_w_turb, vtkFileName_wo_turb)

    print("All done!")
