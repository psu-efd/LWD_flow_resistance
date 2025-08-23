#Plot the effect of mesh resolution on the result


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

def plot_contour_from_vtk(SRH_or_RAS, case_ID, Fr, beta, rect_x, rect_y, rect_width, rect_height, vtkFileName_mesh_1, vtkFileName_mesh_2, vtkFileName_mesh_3, vtkFileName_mesh_4, full_view_or_zoom_view):
    """
    The plot has 3 rows and 4 columns: 
      - First column is to plot the contours of U, V, and WSE with mesh 1
      - Second column is to plot the contours of U, V, and WSE with mesh 2
      - Third column is to plot the contours of U, V, and WSE with mesh 3
      - Fourth column is to plot the contours of U, V, and WSE with mesh 4
    """

    cmap = "viridis"

    if full_view_or_zoom_view == "zoom_view":
        x_min = -2.0
        x_max = 4.0

    #load data from vtk file: water depht and velocity

    if not os.path.exists(vtkFileName_mesh_1) or not os.path.exists(vtkFileName_mesh_2) or not os.path.exists(vtkFileName_mesh_3) or not os.path.exists(vtkFileName_mesh_4):
        print(f"File {vtkFileName_mesh_1} or {vtkFileName_mesh_2} or {vtkFileName_mesh_3} or {vtkFileName_mesh_4} does not exist.")
        return None

    points_mesh_1, Vel_x_np_mesh_1, Vel_y_np_mesh_1, Vel_mag_np_mesh_1, Water_Depth_m_mesh_1, WSE_m_mesh_1 = extract_data_from_vtk(vtkFileName_mesh_1, SRH_or_RAS)
    points_mesh_2, Vel_x_np_mesh_2, Vel_y_np_mesh_2, Vel_mag_np_mesh_2, Water_Depth_m_mesh_2, WSE_m_mesh_2 = extract_data_from_vtk(vtkFileName_mesh_2, SRH_or_RAS)
    points_mesh_3, Vel_x_np_mesh_3, Vel_y_np_mesh_3, Vel_mag_np_mesh_3, Water_Depth_m_mesh_3, WSE_m_mesh_3 = extract_data_from_vtk(vtkFileName_mesh_3, SRH_or_RAS)
    points_mesh_4, Vel_x_np_mesh_4, Vel_y_np_mesh_4, Vel_mag_np_mesh_4, Water_Depth_m_mesh_4, WSE_m_mesh_4 = extract_data_from_vtk(vtkFileName_mesh_4, SRH_or_RAS)

     # Debug: Check the data before interpolation
    print(f"points_mesh_1 shape: {points_mesh_1.shape if points_mesh_1 is not None else 'None'}")
    print(f"Vel_x_np_mesh_1 shape: {Vel_x_np_mesh_1.shape if Vel_x_np_mesh_1 is not None else 'None'}")
    print(f"points_mesh_1 range: X[{points_mesh_1[:, 0].min():.3f}, {points_mesh_1[:, 0].max():.3f}], Y[{points_mesh_1[:, 1].min():.3f}, {points_mesh_1[:, 1].max():.3f}]")
    print(f"Vel_x_np_mesh_1 range: [{Vel_x_np_mesh_1.min():.3f}, {Vel_x_np_mesh_1.max():.3f}]")
    print(f"Any NaN in points_mesh_1: {np.any(np.isnan(points_mesh_1))}")
    print(f"Any NaN in Vel_x_np_mesh_1: {np.any(np.isnan(Vel_x_np_mesh_1))}")


    # Create a grid for contour plotting
    x = points_mesh_1[:, 0]
    y = points_mesh_1[:, 1]
    
    x_mesh_2 = points_mesh_2[:, 0]
    y_mesh_2 = points_mesh_2[:, 1]    

    x_mesh_3 = points_mesh_3[:, 0]
    y_mesh_3 = points_mesh_3[:, 1]

    x_mesh_4 = points_mesh_4[:, 0]
    y_mesh_4 = points_mesh_4[:, 1]

    # Create a regular grid interpolated from the scattered data
    xi = np.linspace(x.min()+0.01, x.max()-0.01, 420)
    yi = np.linspace(y.min()+0.01, y.max()-0.01, 60)
    X, Y = np.meshgrid(xi, yi)

     # Debug: Check the grid
    print(f"Grid range: X[{X.min():.3f}, {X.max():.3f}], Y[{Y.min():.3f}, {Y.max():.3f}]")
    print(f"Grid shape: {X.shape}")

    # Interpolate the scalar field onto the grid
    vel_x_mesh_1 = griddata(points_mesh_1, Vel_x_np_mesh_1, (X, Y), method="nearest")
    vel_y_mesh_1 = griddata(points_mesh_1, Vel_y_np_mesh_1, (X, Y), method="nearest")
    wse_mesh_1 = griddata(points_mesh_1, WSE_m_mesh_1, (X, Y), method="nearest")

    vel_x_mesh_2 = griddata(points_mesh_2, Vel_x_np_mesh_2, (X, Y), method="nearest")
    vel_y_mesh_2 = griddata(points_mesh_2, Vel_y_np_mesh_2, (X, Y), method="nearest")
    wse_mesh_2 = griddata(points_mesh_2, WSE_m_mesh_2, (X, Y), method="nearest")

    vel_x_mesh_3 = griddata(points_mesh_3, Vel_x_np_mesh_3, (X, Y), method="nearest")
    vel_y_mesh_3 = griddata(points_mesh_3, Vel_y_np_mesh_3, (X, Y), method="nearest")
    wse_mesh_3 = griddata(points_mesh_3, WSE_m_mesh_3, (X, Y), method="nearest")

    vel_x_mesh_4 = griddata(points_mesh_4, Vel_x_np_mesh_4, (X, Y), method="nearest")
    vel_y_mesh_4 = griddata(points_mesh_4, Vel_y_np_mesh_4, (X, Y), method="nearest")
    wse_mesh_4 = griddata(points_mesh_4, WSE_m_mesh_4, (X, Y), method="nearest")

     # Debug: Check the result
    print(f"vel_x_mesh_1 shape: {vel_x_mesh_1.shape}")
    print(f"vel_x_mesh_1 range: [{vel_x_mesh_1.min():.3f}, {vel_x_mesh_1.max():.3f}]")
    print(f"NaN count in vel_x_mesh_1: {np.sum(np.isnan(vel_x_mesh_1))}")
    print(f"Total points in vel_x_mesh_1: {vel_x_mesh_1.size}")

    #compute the difference of vel_x, vel_y, and wse between w_turb and wo_turb
    vel_x_diff = vel_x_mesh_1 - vel_x_mesh_2
    vel_y_diff = vel_y_mesh_1 - vel_y_mesh_2
    wse_diff = wse_mesh_1 - wse_mesh_2

    #compute the range of vel_x, vel_y, and wse
    vmin_vel_x_mesh_1 = vel_x_mesh_1.min()
    vmax_vel_x_mesh_1 = vel_x_mesh_1.max()

    vmin_vel_y_mesh_1 = vel_y_mesh_1.min()
    vmax_vel_y_mesh_1 = vel_y_mesh_1.max()

    vmin_wse_mesh_1 = wse_mesh_1.min()
    vmax_wse_mesh_1 = wse_mesh_1.max()
    
    vmin_vel_x_mesh_2 = vel_x_mesh_2.min()
    vmax_vel_x_mesh_2 = vel_x_mesh_2.max()

    vmin_vel_y_mesh_2 = vel_y_mesh_2.min()
    vmax_vel_y_mesh_2 = vel_y_mesh_2.max()

    vmin_wse_mesh_2 = wse_mesh_2.min()
    vmax_wse_mesh_2 = wse_mesh_2.max()

    vmin_vel_x_mesh_3 = vel_x_mesh_3.min()
    vmax_vel_x_mesh_3 = vel_x_mesh_3.max()

    vmin_vel_y_mesh_3 = vel_y_mesh_3.min()
    vmax_vel_y_mesh_3 = vel_y_mesh_3.max()

    vmin_wse_mesh_3 = wse_mesh_3.min()
    vmax_wse_mesh_3 = wse_mesh_3.max()

    vmin_vel_x_mesh_4 = vel_x_mesh_4.min()
    vmax_vel_x_mesh_4 = vel_x_mesh_4.max()

    vmin_vel_y_mesh_4 = vel_y_mesh_4.min()
    vmax_vel_y_mesh_4 = vel_y_mesh_4.max()

    vmin_wse_mesh_4 = wse_mesh_4.min()
    vmax_wse_mesh_4 = wse_mesh_4.max()

    vmin_vel_x_diff = vel_x_mesh_1 - vel_x_mesh_2.min()
    vmax_vel_x_diff = vel_x_mesh_1 - vel_x_mesh_2.max()

    vmin_vel_y_diff = vel_y_mesh_1 - vel_y_mesh_2.min()
    vmax_vel_y_diff = vel_y_mesh_1 - vel_y_mesh_2.max()

    vmin_wse_diff = wse_diff.min()
    vmax_wse_diff = wse_diff.max()  

    #compute the range of vel_x, vel_y, and wse across the four meshes
    vmin_vel_x_all = np.min([vmin_vel_x_mesh_1, vmin_vel_x_mesh_2, vmin_vel_x_mesh_3, vmin_vel_x_mesh_4])
    vmax_vel_x_all = np.max([vmax_vel_x_mesh_1, vmax_vel_x_mesh_2, vmax_vel_x_mesh_3, vmax_vel_x_mesh_4])
    vmin_vel_y_all = np.min([vmin_vel_y_mesh_1, vmin_vel_y_mesh_2, vmin_vel_y_mesh_3, vmin_vel_y_mesh_4])
    vmax_vel_y_all = np.max([vmax_vel_y_mesh_1, vmax_vel_y_mesh_2, vmax_vel_y_mesh_3, vmax_vel_y_mesh_4])
    vmin_wse_all = np.min([vmin_wse_mesh_1, vmin_wse_mesh_2, vmin_wse_mesh_3, vmin_wse_mesh_4])
    vmax_wse_all = np.max([vmax_wse_mesh_1, vmax_wse_mesh_2, vmax_wse_mesh_3, vmax_wse_mesh_4])

    print(f"vmin_vel_x_all = {vmin_vel_x_all}, vmax_vel_x_all = {vmax_vel_x_all}")
    print(f"vmin_vel_y_all = {vmin_vel_y_all}, vmax_vel_y_all = {vmax_vel_y_all}")
    print(f"vmin_wse_all = {vmin_wse_all}, vmax_wse_all = {vmax_wse_all}")

    #plot the contours
    if full_view_or_zoom_view == "zoom_view":
        fig, axs = plt.subplots(3, 4, figsize=(14, 4))
    else:
        fig, axs = plt.subplots(3, 4, figsize=(20, 4))
    
    # Row and column labels
    row_labels = ['u (m/s)', 'v (m/s)', 'WSE (m)']
    col_labels = ['Mesh 1', 'Mesh 2', 'Mesh 3', 'Mesh 4']

    levels_u = np.linspace(vmin_vel_x_all, vmax_vel_x_all, 21)  # 21 levels for 20 contours
    levels_v = np.linspace(vmin_vel_y_all, vmax_vel_y_all, 21)  # 21 levels for 20 contours
    levels_wse = np.linspace(vmin_wse_all, vmax_wse_all, 21)  # 21 levels for 20 contours
    
    # Plot with mesh 1 (first column)
    # U-velocity with mesh 1
    
    im1 = axs[0, 0].contourf(X, Y, vel_x_mesh_1, levels=levels_u, cmap=cmap, 
                               vmin=vmin_vel_x_all, vmax=vmax_vel_x_all)
    axs[0, 0].set_title(f'{row_labels[0]} - {col_labels[0]}', fontsize=14)
    axs[0, 0].set_ylabel('Y (m)', fontsize=14)
    axs[0, 0].set_xticks([])
    axs[0, 0].tick_params(axis='y', which='major', labelsize=14)
    if full_view_or_zoom_view == "zoom_view":
        axs[0, 0].set_xlim(x_min, x_max)
    #plt.colorbar(im1, ax=axs[0, 0], format='%.3f', shrink=0.8, aspect=20)

    divider = make_axes_locatable(axs[0, 0])
    cax = divider.append_axes("right", size="3%", pad=0.2)
    clb = fig.colorbar(im1, ticks=np.linspace(vmin_vel_x_all, vmax_vel_x_all, 5), cax=cax, shrink=1.2)
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
    
    # V-velocity with mesh 1
    im2 = axs[1, 0].contourf(X, Y, vel_y_mesh_1, levels=levels_v, cmap=cmap,
                               vmin=vmin_vel_y_all, vmax=vmax_vel_y_all)
    axs[1, 0].set_title(f'{row_labels[1]} - {col_labels[0]}', fontsize=14)
    axs[1, 0].set_ylabel('Y (m)', fontsize=14)
    axs[1, 0].set_xticks([])
    axs[1, 0].tick_params(axis='y', which='major', labelsize=14)
    if full_view_or_zoom_view == "zoom_view":
        axs[1, 0].set_xlim(x_min, x_max)
    #plt.colorbar(im2, ax=axs[1, 0], format='%.3f', shrink=0.8, aspect=20)

    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes("right", size="3%", pad=0.2)
    clb = fig.colorbar(im2, ticks=np.linspace(vmin_vel_y_all, vmax_vel_y_all, 5), cax=cax)
    clb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb.ax.tick_params(labelsize=12)
    
    # WSE with mesh 1
    im3 = axs[2, 0].contourf(X, Y, wse_mesh_1, levels=levels_wse, cmap=cmap,
                               vmin=vmin_wse_all, vmax=vmax_wse_all)
    axs[2, 0].set_title(f'{row_labels[2]} - {col_labels[0]}', fontsize=14)
    axs[2, 0].set_xlabel('X (m)', fontsize=14)
    axs[2, 0].set_ylabel('Y (m)', fontsize=14)
    axs[2, 0].tick_params(axis='both', which='major', labelsize=14)
    if full_view_or_zoom_view == "zoom_view":
        axs[2, 0].set_xlim(x_min, x_max)
    #plt.colorbar(im3, ax=axs[2, 0], format='%.3f', shrink=0.8, aspect=20)
    
    divider = make_axes_locatable(axs[2, 0])
    cax = divider.append_axes("right", size="3%", pad=0.2)
    clb = fig.colorbar(im3, ticks=np.linspace(vmin_wse_all, vmax_wse_all, 5), cax=cax)
    clb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb.ax.tick_params(labelsize=12)
    
    # Plot mesh 2 (second column)
    # U-velocity with mesh 2
    im4 = axs[0, 1].contourf(X, Y, vel_x_mesh_2, levels=levels_u, cmap=cmap,
                               vmin=vmin_vel_x_all, vmax=vmax_vel_x_all)
    axs[0, 1].set_title(f'{row_labels[0]} - {col_labels[1]}', fontsize=14)
    if full_view_or_zoom_view == "zoom_view":
        axs[0, 1].set_xlim(x_min, x_max)
    #axs[0, 1].tick_params(axis='both', which='major', labelsize=10)
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    #plt.colorbar(im4, ax=axs[0, 1], format='%.3f', shrink=0.8, aspect=20)

    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes("right", size="3%", pad=0.2)
    clb = fig.colorbar(im4, ticks=np.linspace(vmin_vel_x_all, vmax_vel_x_all, 5), cax=cax)
    clb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb.ax.tick_params(labelsize=12)
    
    # V-velocity with mesh 2
    im5 = axs[1, 1].contourf(X, Y, vel_y_mesh_2, levels=levels_v, cmap=cmap,
                               vmin=vmin_vel_y_all, vmax=vmax_vel_y_all)
    axs[1, 1].set_title(f'{row_labels[1]} - {col_labels[1]}', fontsize=14)
    if full_view_or_zoom_view == "zoom_view":
        axs[1, 1].set_xlim(x_min, x_max)
    #axs[1, 1].tick_params(axis='both', which='major', labelsize=10)
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    #plt.colorbar(im5, ax=axs[1, 1], format='%.3f', shrink=0.8, aspect=20)

    divider = make_axes_locatable(axs[1, 1])
    cax = divider.append_axes("right", size="3%", pad=0.2)
    clb = fig.colorbar(im5, ticks=np.linspace(vmin_vel_y_all, vmax_vel_y_all, 5), cax=cax)
    clb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb.ax.tick_params(labelsize=12)
    
    # WSE with mesh 2
    im6 = axs[2, 1].contourf(X, Y, wse_mesh_2, levels=levels_wse, cmap=cmap,
                               vmin=vmin_wse_all, vmax=vmax_wse_all)
    axs[2, 1].set_title(f'{row_labels[2]} - {col_labels[1]}', fontsize=14)
    axs[2, 1].set_xlabel('X (m)', fontsize=14)
    axs[2, 1].set_yticks([])
    axs[2, 1].tick_params(axis='x', which='major', labelsize=14)
    if full_view_or_zoom_view == "zoom_view":
        axs[2, 1].set_xlim(x_min, x_max)
    #plt.colorbar(im6, ax=axs[2, 1], format='%.3f', shrink=0.8, aspect=20)

    divider = make_axes_locatable(axs[2, 1])
    cax = divider.append_axes("right", size="3%", pad=0.2)
    clb = fig.colorbar(im6, ticks=np.linspace(vmin_wse_all, vmax_wse_all, 5), cax=cax)
    clb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb.ax.tick_params(labelsize=12)
    
    # Plot mesh 3 (third column)
    # U-velocity with mesh 3
    im7 = axs[0, 2].contourf(X, Y, vel_x_mesh_3, levels=levels_u, cmap=cmap,
                               vmin=vmin_vel_x_all, vmax=vmax_vel_x_all)
    axs[0, 2].set_title(f'{row_labels[0]} - {col_labels[2]}', fontsize=14)
    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])
    if full_view_or_zoom_view == "zoom_view":
        axs[0, 2].set_xlim(x_min, x_max)
    #plt.colorbar(im7, ax=axs[0, 2], format='%.3f', shrink=0.8, aspect=20)

    divider = make_axes_locatable(axs[0, 2])
    cax = divider.append_axes("right", size="3%", pad=0.2)
    clb = fig.colorbar(im7, ticks=np.linspace(vmin_vel_x_all, vmax_vel_x_all, 5), cax=cax)
    clb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb.ax.tick_params(labelsize=12)
    
    # V-velocity with mesh 3
    im8 = axs[1, 2].contourf(X, Y, vel_y_mesh_3, levels=levels_v, cmap=cmap,
                               vmin=vmin_vel_y_all, vmax=vmax_vel_y_all)
    axs[1, 2].set_title(f'{row_labels[1]} - {col_labels[2]}', fontsize=14)
    axs[1, 2].set_xticks([])
    axs[1, 2].set_yticks([])
    if full_view_or_zoom_view == "zoom_view":
        axs[1, 2].set_xlim(x_min, x_max)
    #plt.colorbar(im8, ax=axs[1, 2], format='%.3f', shrink=0.8, aspect=20)

    divider = make_axes_locatable(axs[1, 2])
    cax = divider.append_axes("right", size="3%", pad=0.2)
    clb = fig.colorbar(im8, ticks=np.linspace(vmin_vel_y_all, vmax_vel_y_all, 5), cax=cax)
    clb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb.ax.tick_params(labelsize=12)
    
    # WSE with mesh 3
    im9 = axs[2, 2].contourf(X, Y, wse_mesh_3, levels=levels_wse, cmap=cmap,
                               vmin=vmin_wse_all, vmax=vmax_wse_all)
    axs[2, 2].set_title(f'{row_labels[2]} - {col_labels[2]}', fontsize=14)
    axs[2, 2].set_xlabel('X (m)', fontsize=14)
    axs[2, 2].set_yticks([])
    axs[2, 2].tick_params(axis='x', which='major', labelsize=14)
    if full_view_or_zoom_view == "zoom_view":
        axs[2, 2].set_xlim(x_min, x_max)
    #plt.colorbar(im9, ax=axs[2, 2], format='%.3f', shrink=0.8, aspect=20)

    divider = make_axes_locatable(axs[2, 2])
    cax = divider.append_axes("right", size="3%", pad=0.2)
    clb = fig.colorbar(im9, ticks=np.linspace(vmin_wse_all, vmax_wse_all, 5), cax=cax)
    clb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb.ax.tick_params(labelsize=12)
    
    # Plot mesh 4 (fourth column)
    # U-velocity with mesh 4
    im7 = axs[0, 3].contourf(X, Y, vel_x_mesh_4, levels=levels_u, cmap=cmap,
                               vmin=vmin_vel_x_all, vmax=vmax_vel_x_all)
    axs[0, 3].set_title(f'{row_labels[0]} - {col_labels[3]}', fontsize=14)
    axs[0, 3].set_xticks([])
    axs[0, 3].set_yticks([])
    if full_view_or_zoom_view == "zoom_view":
        axs[0, 3].set_xlim(x_min, x_max)
    #plt.colorbar(im7, ax=axs[0, 3], format='%.3f', shrink=0.8, aspect=20)

    divider = make_axes_locatable(axs[0, 3])
    cax = divider.append_axes("right", size="3%", pad=0.2)
    clb = fig.colorbar(im7, ticks=np.linspace(vmin_vel_x_all, vmax_vel_x_all, 5), cax=cax)
    clb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb.ax.tick_params(labelsize=12)
    
    # V-velocity with mesh 3
    im8 = axs[1, 3].contourf(X, Y, vel_y_mesh_4, levels=levels_v, cmap=cmap,
                               vmin=vmin_vel_y_all, vmax=vmax_vel_y_all)
    axs[1, 3].set_title(f'{row_labels[1]} - {col_labels[3]}', fontsize=14)
    axs[1, 3].set_xticks([])
    axs[1, 3].set_yticks([])
    if full_view_or_zoom_view == "zoom_view":
        axs[1, 3].set_xlim(x_min, x_max)
    #plt.colorbar(im8, ax=axs[1, 3], format='%.3f', shrink=0.8, aspect=20)

    divider = make_axes_locatable(axs[1, 3])
    cax = divider.append_axes("right", size="3%", pad=0.2)
    clb = fig.colorbar(im8, ticks=np.linspace(vmin_vel_y_all, vmax_vel_y_all, 5), cax=cax)
    clb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb.ax.tick_params(labelsize=12)
    
    # WSE with mesh 3
    im9 = axs[2, 3].contourf(X, Y, wse_mesh_4, levels=levels_wse, cmap=cmap,
                               vmin=vmin_wse_all, vmax=vmax_wse_all)
    axs[2, 3].set_title(f'{row_labels[2]} - {col_labels[3]}', fontsize=14)
    axs[2, 3].set_xlabel('X (m)', fontsize=14)
    axs[2, 3].set_yticks([])
    axs[2, 3].tick_params(axis='x', which='major', labelsize=14)
    if full_view_or_zoom_view == "zoom_view":
        axs[2, 3].set_xlim(x_min, x_max)
    #plt.colorbar(im9, ax=axs[2, 2], format='%.3f', shrink=0.8, aspect=20)

    divider = make_axes_locatable(axs[2, 3])
    cax = divider.append_axes("right", size="3%", pad=0.2)
    clb = fig.colorbar(im9, ticks=np.linspace(vmin_wse_all, vmax_wse_all, 5), cax=cax)
    clb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    clb.ax.tick_params(labelsize=12)    
    
    
    # Add LWD rectangle to all plots
    for i in range(3):
        for j in range(4):            
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
    if full_view_or_zoom_view == "zoom_view":
        output_filename = f'mesh_effect_case_{case_ID}_Cd_{SRH_or_RAS}_zoom_view.png'
    else:
        output_filename = f'mesh_effect_case_{case_ID}_Cd_{SRH_or_RAS}.png'

    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_filename}")
    
    # Show the plot
    #plt.show()

    plt.close()
    
    #return fig

if __name__ == "__main__":

    #define data 

    #whether to plot the full view or the zoom view
    full_view_or_zoom_view = "zoom_view"
    #full_view_or_zoom_view = "full_view"

    case_IDs = [4, 8, 12, 16]
    #case_IDs = [8]

    Frs = [0.137, 0.153,0.099, 0.163]
    
    betas = [1, 0.5, 1, 0.5]
    alphas_exp = [1, 0.832, 1, 0.836]
    alphas_simulation_SRH_2D = [1, 0.831, 1, 0.837]
    alphas_simulation_HEC_RAS_2D = [1, 0.8302, 1, 0.8364]
    
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
        vtkFileName_mesh_1 = "Exp_"+str(case_ID)+"_Cd/SRH2D_Exp_"+str(case_ID)+"_Cd_C_0003.vtk"
        vtkFileName_mesh_2 = "Exp_"+str(case_ID)+"_Cd/SRH2D_Exp_"+str(case_ID)+"_Cd_C_0003_coarse_1.vtk"
        vtkFileName_mesh_3 = "Exp_"+str(case_ID)+"_Cd/SRH2D_Exp_"+str(case_ID)+"_Cd_C_0003_coarse_2.vtk"
        vtkFileName_mesh_4 = "Exp_"+str(case_ID)+"_Cd/SRH2D_Exp_"+str(case_ID)+"_Cd_C_0003_coarse_3.vtk"

        SRH_or_RAS = "SRH_2D"
        plot_contour_from_vtk(SRH_or_RAS, case_ID, Frs[i], betas[i], rect_x, rect_y, rect_width, rect_height, vtkFileName_mesh_1, vtkFileName_mesh_2, vtkFileName_mesh_3, vtkFileName_mesh_4, full_view_or_zoom_view)

        #plot HEC-RAS 2D results
        vtkFileName_mesh_1 = "Exp_"+str(case_ID)+"_Cd/RAS_2D_"+str(case_ID).zfill(2)+".vtk"
        vtkFileName_mesh_2 = "Exp_"+str(case_ID)+"_Cd/RAS_2D_"+str(case_ID).zfill(2)+"_coarse_1.vtk"
        vtkFileName_mesh_3 = "Exp_"+str(case_ID)+"_Cd/RAS_2D_"+str(case_ID).zfill(2)+"_coarse_2.vtk"
        vtkFileName_mesh_4 = "Exp_"+str(case_ID)+"_Cd/RAS_2D_"+str(case_ID).zfill(2)+"_coarse_3.vtk"
        SRH_or_RAS = "HEC_RAS_2D"
        plot_contour_from_vtk(SRH_or_RAS, case_ID, Frs[i], betas[i], rect_x, rect_y, rect_width, rect_height, vtkFileName_mesh_1, vtkFileName_mesh_2, vtkFileName_mesh_3, vtkFileName_mesh_4, full_view_or_zoom_view)

    print("All done!")
