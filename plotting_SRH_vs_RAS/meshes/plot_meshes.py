"""
Plot the meshes for the different cases. The meshes are in the SRH-2D and RAS-2D folders in the format of vtk files.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import patches

import meshio

import os


plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

def plot_mesh_from_vtk(vtk_file_path, bZoomIn):
    """
    Plot the mesh from the vtk file, which is in legacy UNSTRUCTURED_GRID VTK format. The mesh is a 2D mesh.

    Args:
        vtk_file_path (str): The path to the vtk file.
        bZoomIn (bool): Whether to zoom in on the mesh.
    Returns:
        None
    """
    mesh = meshio.read(vtk_file_path)

    # Extract the points and cells from the mesh
    points = mesh.points
    cells = mesh.cells

    #flip the sign of the x-coordinates
    points[:, 0] = -points[:, 0]

    # Extract the cell types and the number of cells for each cell type
    cell_types = [cell.type for cell in cells]
    num_cells = [len(cell.data) for cell in cells]

    # Extract the points for each cell
    cell_data = [cell.data for cell in cells]

    # Plot the mesh
    if not bZoomIn:
        plt.figure(figsize=(15.3, 1.5))
    else:
        plt.figure(figsize=(2, 1.5))
    
    # Plot points
    #plt.scatter(points[:, 0], points[:, 1], c='red', s=20, alpha=0.7, label='Nodes')
    
    # Plot cells based on their type
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    color_idx = 0
    
    for i, (cell_type, cell_indices) in enumerate(zip(cell_types, cell_data)):
        #color = colors[color_idx % len(colors)]
        color = 'blue'
        
        if cell_type == "triangle":
            # Plot triangular cells
            for cell in cell_indices:
                # Extract the three vertices of the triangle
                triangle_points = points[cell]


                # Plot the triangle edges
                plt.plot([triangle_points[0, 0], triangle_points[1, 0]], 
                        [triangle_points[0, 1], triangle_points[1, 1]], 
                        color=color, linewidth=0.5, alpha=0.6)
                plt.plot([triangle_points[1, 0], triangle_points[2, 0]], 
                        [triangle_points[1, 1], triangle_points[2, 1]], 
                        color=color, linewidth=0.5, alpha=0.6)
                plt.plot([triangle_points[2, 0], triangle_points[0, 0]], 
                        [triangle_points[2, 1], triangle_points[0, 1]], 
                        color=color, linewidth=0.5, alpha=0.6)
        
        elif cell_type == "quad":
            # Plot quadrilateral cells
            for cell in cell_indices:
                # Extract the four vertices of the quad
                quad_points = points[cell]
                # Plot the quad edges
                plt.plot([quad_points[0, 0], quad_points[1, 0]], 
                        [quad_points[0, 1], quad_points[1, 1]], 
                        color=color, linewidth=0.5, alpha=0.6)
                plt.plot([quad_points[1, 0], quad_points[2, 0]], 
                        [quad_points[1, 1], quad_points[2, 1]], 
                        color=color, linewidth=0.5, alpha=0.6)
                plt.plot([quad_points[2, 0], quad_points[3, 0]], 
                        [quad_points[2, 1], quad_points[3, 1]], 
                        color=color, linewidth=0.5, alpha=0.6)
                plt.plot([quad_points[3, 0], quad_points[0, 0]], 
                        [quad_points[3, 1], quad_points[0, 1]], 
                        color=color, linewidth=0.5, alpha=0.6)
        
        elif cell_type == "line":
            # Plot line cells (1D elements, might be boundaries)
            for cell in cell_indices:
                line_points = points[cell]
                if not bZoomIn:   #only plot the line when not zoomed in
                    plt.plot([line_points[0, 0], line_points[1, 0]], 
                            [line_points[0, 1], line_points[1, 1]], 
                            color=color, linewidth=0.5, alpha=0.8)
                
        
        else:
            # Handle other cell types generically
            #print(f"Warning: Unsupported cell type '{cell_type}' with {len(cell_indices[0])} vertices")
            for cell in cell_indices:
                cell_points = points[cell]
                # Plot edges between consecutive vertices
                for j in range(len(cell_points)):
                    next_j = (j + 1) % len(cell_points)
                    plt.plot([cell_points[j, 0], cell_points[next_j, 0]], 
                            [cell_points[j, 1], cell_points[next_j, 1]], 
                            color=color, linewidth=0.5, alpha=0.6)
        
        color_idx += 1
    

    # Draw a rectangle to denote the location of the large woody debris
    # Rectangle parameters: (x, y), width, height
    rect = patches.Rectangle(
        (-0.1, 0.0),   # bottom-left corner
        0.2,        # width
        1.5,        # height
        linewidth=2,
        edgecolor='red',
        facecolor='none'  # No fill
    )
    plt.gca().add_patch(rect)

    # Add legend
    #legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
    #                              markersize=8, label='Nodes')]
    
    #for i, cell_type in enumerate(set(cell_types)):
    #    color = colors[i % len(colors)]
    #    legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=2, 
    #                                    label=f'{cell_type.capitalize()} cells'))
    
    #plt.legend(handles=legend_elements, loc='upper right')
    
    # Set equal aspect ratio for proper mesh visualization
    #plt.axis('equal')
    #plt.grid(True, alpha=0.3)

    # Set the x and y limits
    if bZoomIn:
        plt.xlim(-1.0, 1.0)
        plt.ylim(0.0, 1.5)
    else:
        plt.xlim(-5.1, 10.2)
        plt.ylim(0.0, 1.5)

    #format the x and y ticks and labels
    if not bZoomIn:
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('x (m)', fontsize=18)
        plt.ylabel('y (m)', fontsize=18)
    else:
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('')
        plt.ylabel('')

        #turn off the axis border
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
    
    # Show the plot
    plt.tight_layout()

    # Save the plot
    if bZoomIn:
        plt.savefig(vtk_file_path.replace(".vtk", "_zoomed.png"), dpi=300, bbox_inches='tight')
    else:
        plt.savefig(vtk_file_path.replace(".vtk", ".png"), dpi=300, bbox_inches='tight')
    #plt.show()
    
    # Print mesh information
    #print(f"Mesh loaded from: {vtk_file_path}")
    #print(f"Number of nodes: {len(points)}")
    #print(f"Number of cell types: {len(cell_types)}")
    #for cell_type, num_cell in zip(cell_types, num_cells):
    #    print(f"  {cell_type}: {num_cell} cells")


def plot_all_meshes_in_directory(directory_path, file_pattern="*.vtk"):
    """
    Plot all VTK mesh files in a directory.
    
    Args:
        directory_path (str): Path to the directory containing VTK files
        file_pattern (str): File pattern to match (default: "*.vtk")
    
    Returns:
        None
    """
    import glob
    
    # Find all VTK files in the directory
    vtk_files = glob.glob(os.path.join(directory_path, file_pattern))
    
    if not vtk_files:
        print(f"No VTK files found in {directory_path}")
        return
    
    print(f"Found {len(vtk_files)} VTK files in {directory_path}")
    
    # Plot each mesh
    for vtk_file in vtk_files:
        print(f"\nPlotting mesh: {os.path.basename(vtk_file)}")
        try:
            plot_mesh_from_vtk(vtk_file)
        except Exception as e:
            print(f"Error plotting {vtk_file}: {e}")


if __name__ == "__main__":

    #plot SRH-2D mesh    
    # plot_mesh_from_vtk("SRH-2D/mesh.vtk", bZoomIn=False)
    # plot_mesh_from_vtk("SRH-2D/mesh.vtk", bZoomIn=True)

    # plot_mesh_from_vtk("SRH-2D/coarse_1.vtk", bZoomIn=False)
    # plot_mesh_from_vtk("SRH-2D/coarse_1.vtk", bZoomIn=True)

    # plot_mesh_from_vtk("SRH-2D/coarse_2.vtk", bZoomIn=False)
    # plot_mesh_from_vtk("SRH-2D/coarse_2.vtk", bZoomIn=True)

    # plot_mesh_from_vtk("SRH-2D/coarse_3.vtk", bZoomIn=False)
    # plot_mesh_from_vtk("SRH-2D/coarse_3.vtk", bZoomIn=True)

    #plot RAS-2D mesh
    #plot_mesh_from_vtk("RAS-2D/mesh.vtk", bZoomIn=False)
    #plot_mesh_from_vtk("RAS-2D/mesh.vtk", bZoomIn=True)

    plot_mesh_from_vtk("RAS-2D/coarse_1.vtk", bZoomIn=False)
    plot_mesh_from_vtk("RAS-2D/coarse_1.vtk", bZoomIn=True)

    plot_mesh_from_vtk("RAS-2D/coarse_2.vtk", bZoomIn=False)
    plot_mesh_from_vtk("RAS-2D/coarse_2.vtk", bZoomIn=True)

    plot_mesh_from_vtk("RAS-2D/coarse_3.vtk", bZoomIn=False)
    plot_mesh_from_vtk("RAS-2D/coarse_3.vtk", bZoomIn=True)


    
    # Example 2: Plot all meshes in a directory
    # plot_all_meshes_in_directory("SRH_2D/Cd_approach_w_turb/Exp_1_Cd")
    
  
    
    print("Mesh plotting functions loaded. Use plot_mesh_from_vtk() to plot individual meshes.")
    print("Use plot_all_meshes_in_directory() to plot all VTK files in a directory.")
  