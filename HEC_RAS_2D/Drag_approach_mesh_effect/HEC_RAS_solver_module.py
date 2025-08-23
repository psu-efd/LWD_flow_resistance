"""
A module with the function to call HEC-RAS. The main purpose of this module is to run the calibration for all the cases: The b_coefficient is calibrated for each case.
"""

import multiprocessing
import os
import shutil
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from skopt import gp_minimize, dump, load
from skopt.plots import plot_gaussian_process
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern

from pathlib import Path

import pandas as pd

import numpy as np

import vtk

from vtk.util import numpy_support as VN

import pyHMT2D

#measurement noise level
noise_level = 0.05  #5%

#GP parameters
n_calls=10 #10  # the number of evaluations of f
n_initial_points=5 #5  # the number of random initialization points

# Gloabl lists to store the history of b_coefficient values, probe values, and objective function values
b_Coefficient_history = []
Objective_function_history = []  #total error
Objective_function_history_Water_Depth = []  #error due to water depth
Objective_function_history_Q_split = []      #error due to flow split
WSE_upstream_history = []
WSE_downstream_history = []
H_upstream_history = []
H_downstream_history = []
Q_split_history = []

#global variables
current_case_ID = 1  #current case ID (needed for the objective function because objective function only takes one parameter)

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

def clear_global_variables():
    """
    Clear the global variables such as the history of b_coefficient values, probe values, and objective function values.
    """

    b_Coefficient_history.clear()
    Objective_function_history.clear()
    Objective_function_history_Water_Depth.clear()
    Objective_function_history_Q_split.clear()
    WSE_upstream_history.clear()
    WSE_downstream_history.clear()
    H_upstream_history.clear()
    H_downstream_history.clear()
    Q_split_history.clear()

def run_one_HEC_RAS_case_PorosityDragCoefficient(case_ID, PorosityDrag_MaterialID, PorosityDrag_b_Coefficient, PorosityDrag_MaterialName, faceless=True):
    """
    Run a single HEC-RAS case with the specified case_ID and PorosityDrag_b_Coefficient value. The PorosityDrag_b_Coefficient value is used to set the PorosityDrag_b_Coefficient for the LWD zone.

    In HEC-RAS, the porosity, a, and b coefficients are used to compute the drag induced by vegetation. In this function, we only vary the b coefficient.

    Parameters
    ----------
    case_ID : int
        ID of the case to run
    PorosityDrag_MaterialID : int
        ID of the PorosityDrag zone. Note: In HEC-RAS, the ID is 1-based.
    PorosityDrag_b_Coefficient : float
        PorosityDrag_b_Coefficient value for this case
    PorosityDrag_MaterialName : str
        Name of the PorosityDrag zone

    Returns
    -------

    """

    print(f"Running HEC-RAS case {case_ID} with PorosityDrag_b_Coefficient = {PorosityDrag_b_Coefficient}")

    #base HEC-RAS case directory
    base_case_dir = 'RAS_case'

    #go into the base case directory
    os.chdir(base_case_dir)

    #HEC-RAS plan name: case_1, case_2, etc. (without the extension)
    plan_name = 'case_'+str(case_ID)
    
    #create a HEC-RAS model instance
    my_hec_ras_model = pyHMT2D.RAS_2D.HEC_RAS_Model(version="6.6", faceless=faceless)

    #initialize the HEC-RAS model
    my_hec_ras_model.init_model()

    #print("Hydraulic model name: ", my_hec_ras_model.getName())
    #print("Hydraulic model version: ", my_hec_ras_model.getVersion())

    #open a HEC-RAS project (This is hard-code; needs to be changed for a specific case)
    my_hec_ras_model.open_project("lwd.prj")

    #set the current plan file name
    my_hec_ras_model.set_current_plan(plan_name)
    my_hec_ras_model.save_project()  #save the project to make sure the current plan is set correctly

    print("Current plan name: ", plan_name)

    #plan file name
    plan_file_name = my_hec_ras_model.get_current_planFile()
    #print("Plan file name: ", plan_file_name)

    #get the HEC-RAS case data
    my_hec_ras_data = my_hec_ras_model.get_simulation_case()

    #modify the PorosityDrag_b_Coefficient. In this example, we only modify one zone: LWD.
    PorosityDrag_MaterialIDs = [PorosityDrag_MaterialID] 
    PorosityDrag_b_Coefficients = [PorosityDrag_b_Coefficient]
    PorosityDrag_MaterialNames = [PorosityDrag_MaterialName]
    my_hec_ras_data.modify_PorosityDrag_b_Coefficient(PorosityDrag_MaterialIDs, PorosityDrag_b_Coefficients, PorosityDrag_MaterialNames)

    # update the time stamp of the PorosityDrag_b_Coefficient GeoTiff file (to force HEC-RAS to re-compute 2D flow area's
    # properties table. (No need? The above PorosityDrag_b_Coefficient modification already updated the time stamp.)
    if os.path.dirname(my_hec_ras_data.hdf_filename) == '':
        fileBase = b''
    else:
        fileBase = str.encode(os.path.dirname(my_hec_ras_data.hdf_filename) + '/')

    full_landcover_filename = (fileBase + my_hec_ras_data.landcover_filename).decode("ASCII")
    #print("Landcover filename: ", full_landcover_filename)

    Path(full_landcover_filename).touch()

    # save the current project before run it
    my_hec_ras_model.save_project()

    #run the HEC-RAS model's current project
    bRunSucessful = my_hec_ras_model.run_model()

    #close the HEC-RAS project
    my_hec_ras_model.close_project()

    #quit HEC-RAS
    my_hec_ras_model.exit_model()

    #convert HEC-RAS result to VTK (This is hard-coded; needs to be changes for a specific case)
    #result hdf file name: lwd.p01.hdf, lwd.p02.hdf, etc.
    result_hdf_file_name = plan_file_name + '.hdf'
    #print("Result HDF file name: ", result_hdf_file_name)
    #my_ras_2d_data = pyHMT2D.RAS_2D.RAS_2D_Data(result_hdf_file_name, "Terrain/Terrain_flat.tif")
    my_ras_2d_data = pyHMT2D.RAS_2D.RAS_2D_Data(result_hdf_file_name)

    vtkFileNameList = my_ras_2d_data.saveHEC_RAS2D_results_to_VTK(lastTimeStep=True)
    #print("VTK file name: ", vtkFileNameList[-1])

    #copy the vtk result file (only the last time step) to "cases" directory (one level above)
    shutil.copy(vtkFileNameList[-1], "../calibration_results/"+"case_"+str(case_ID)+"/result.vtk")

    # go back to the root
    os.chdir("..")

    #if successful, return case_ID; otherwise, return -case_ID
    if bRunSucessful:
        return  case_ID
    else:
        return -case_ID
    

def sample_HEC_RAS_results(case_ID):
    """
    Sample the HEC-RAS results in the 2D vtk file.

    - Get the WSE at upstream and downstream points.
    - Sample the velocity at two cross-sections and compute the discharge.

    """

    #vtk file name
    vtk_file_name = "calibration_results/case_"+str(case_ID)+"/result.vtk"

    #sampling points at upstream and downstream
    upstream_point = [2.642, 0.75]
    downstream_point = [-4.673, 0.75]

    #monitoring lines definition: each line is defined by two points
    monitoring_line_1 = [[0, 0], [0, 0.75]]
    monitoring_line_2 = [[0, 0.75], [0, 1.5]]

    #length of the monitoring lines
    length_monitoring_line_1 = 0.75
    length_monitoring_line_2 = 0.75

    #number of points on the monitoring lines
    nPoints_monitoring_line_1 = 40
    nPoints_monitoring_line_2 = 40

    #compute the points on the monitoring lines
    monitoring_line_1_points = np.linspace(monitoring_line_1[0], monitoring_line_1[1], nPoints_monitoring_line_1)
    monitoring_line_2_points = np.linspace(monitoring_line_2[0], monitoring_line_2[1], nPoints_monitoring_line_2)

    #print("Monitoring line 1 points: ", monitoring_line_1_points)
    #print("Monitoring line 2 points: ", monitoring_line_2_points)

    #create vtkPoints for the monitoring points at the upstream and downstream
    monitoring_points = vtk.vtkPoints()

    monitoring_points.InsertNextPoint(upstream_point[0], upstream_point[1], 0.0)
    monitoring_points.InsertNextPoint(downstream_point[0], downstream_point[1], 0.0)

    vtk_handler = pyHMT2D.Misc.vtkHandler()

    vtkUnstructuredGridReader = vtk_handler.readVTK_UnstructuredGrid(vtk_file_name)

    # sample on the sampling points
    points, WSE_at_monitoring_points, bed_elev_at_monitoring_points = vtk_handler.probeUnstructuredGridVTKOverLine(
                                            monitoring_points, vtkUnstructuredGridReader,
                                            'Water_Elev_m')
    
    #print("WSE at monitoring points: ", WSE_at_monitoring_points)

    #create vtkPoints for the points on each monitoring line
    monitoring_line_1_vtkPoints = vtk.vtkPoints()
    monitoring_line_2_vtkPoints = vtk.vtkPoints()

    for pointI in range(nPoints_monitoring_line_1):
        monitoring_line_1_vtkPoints.InsertNextPoint(monitoring_line_1_points[pointI][0], monitoring_line_1_points[pointI][1], 0.0)

    for pointI in range(nPoints_monitoring_line_2):
        monitoring_line_2_vtkPoints.InsertNextPoint(monitoring_line_2_points[pointI][0], monitoring_line_2_points[pointI][1], 0.0)

    # sample the velocity and water depth on the monitoring lines
    points, velocity_at_monitoring_line_1, bed_elev = vtk_handler.probeUnstructuredGridVTKOverLine(
                                            monitoring_line_1_vtkPoints, vtkUnstructuredGridReader,
                                            'Velocity_cell_m_p_s')

    points, velocity_at_monitoring_line_2, bed_elev = vtk_handler.probeUnstructuredGridVTKOverLine(
                                            monitoring_line_2_vtkPoints, vtkUnstructuredGridReader,
                                            'Velocity_cell_m_p_s')
    
    points, water_depth_at_monitoring_line_1, bed_elev = vtk_handler.probeUnstructuredGridVTKOverLine(
                                            monitoring_line_1_vtkPoints, vtkUnstructuredGridReader,
                                            'Water_Depth_m')
    
    points, water_depth_at_monitoring_line_2, bed_elev = vtk_handler.probeUnstructuredGridVTKOverLine(
                                            monitoring_line_2_vtkPoints, vtkUnstructuredGridReader,
                                            'Water_Depth_m')

    #print("Velocity at monitoring line 1: ", velocity_at_monitoring_line_1)
    #print("Velocity at monitoring line 2: ", velocity_at_monitoring_line_2)

    #print("Water depth at monitoring line 1: ", water_depth_at_monitoring_line_1)
    #print("Water depth at monitoring line 2: ", water_depth_at_monitoring_line_2)

    #compute the discharge at the two cross-sections (only using the x-component of the velocity): summation of the discharge at each point
    discharge_at_monitoring_line_1 = abs(np.sum(velocity_at_monitoring_line_1[:,0] * water_depth_at_monitoring_line_1 * length_monitoring_line_1 / nPoints_monitoring_line_1))
    discharge_at_monitoring_line_2 = abs(np.sum(velocity_at_monitoring_line_2[:,0] * water_depth_at_monitoring_line_2 * length_monitoring_line_2 / nPoints_monitoring_line_2))

    #print("Discharge at monitoring line 1: ", discharge_at_monitoring_line_1)
    #print("Discharge at monitoring line 2: ", discharge_at_monitoring_line_2)

    #compute the total discharge
    total_discharge = discharge_at_monitoring_line_1 + discharge_at_monitoring_line_2

    #print("Total discharge: ", total_discharge)

    #compute the flow partition: 
    flow_partition = discharge_at_monitoring_line_2 / total_discharge

    #print("Flow partition: ", flow_partition)

    return WSE_at_monitoring_points, bed_elev_at_monitoring_points, discharge_at_monitoring_line_1, discharge_at_monitoring_line_2, total_discharge, flow_partition


def optimize_model_parameter(parameter_bounds):

    # call the GP optimizer
    result = gp_minimize(
        objective_wrapper_function,  # the function to minimize (only takes one parameter; that is why we need a wrapper function to include other parameters)
        parameter_bounds,  # the bounds on each dimension of x
        acq_func="EI",  # the acquisition function
        n_calls=n_calls,  # the number of evaluations of f
        n_initial_points=n_initial_points,  # the number of random initialization points
        noise= (noise_level) ** 2,  # the noise variance of observation (5% estimated)
        random_state=1234)  # the random seed

    #print("result = ", result)

    # save gp_minimize result
    dump(result, 'calibration_results/case_'+str(current_case_ID)+'/gp_result.pkl')

    # save simulation results to files: 'simulation_results_'+str(current_case_ID)+'.dat' and 'calibrated_b_Coefficient_'+str(current_case_ID)+'.dat'
    save_calibration_results(result.x[0])
    
# Define a wrapper function that only takes the continuous variable and uses the fixed boolean
def objective_wrapper_function(x):
    # Whether it is full width or half-width (it is passed to the objective function)
    if current_case_ID in [4, 41, 42, 43, 12, 121, 122, 123]:
        bFullWidth = True  #half width
    elif current_case_ID in [8, 81, 82, 83, 16, 161, 162, 163]:
        bFullWidth = False  #full width
    else:
        raise ValueError(f"Invalid case ID: {current_case_ID}")
    
    # The following two parameters are fixed for all cases
    PorosityDrag_MaterialID = 1
    PorosityDrag_MaterialName = 'LWD'

    # Call the objective function
    return objective_function_b_Coefficient(x, current_case_ID, PorosityDrag_MaterialID, PorosityDrag_MaterialName, bFullWidth)

# Define the objective function to be minimized (for b_coefficient)
def objective_function_b_Coefficient(b_Coefficient, case_ID, PorosityDrag_MaterialID, PorosityDrag_MaterialName, bFullWidth):

    print("b_Coefficient = ", b_Coefficient)

    # Append the current parameter value to the history
    b_Coefficient_history.append(b_Coefficient[0])

    # Run the case with the current b_Coefficient value. If the simulation is successful, the case_ID is returned; otherwise, -case_ID is returned.
    simulation_result = run_one_HEC_RAS_case_PorosityDragCoefficient(case_ID, PorosityDrag_MaterialID, b_Coefficient[0], PorosityDrag_MaterialName, faceless=False)

    if simulation_result > 0:
        #read the HEC-RAS results
        WSE_at_monitoring_points, bed_elev_at_monitoring_points, discharge_at_monitoring_line_1, discharge_at_monitoring_line_2, total_discharge, flow_partition = \
            sample_HEC_RAS_results(case_ID)       
      
    else:
        raise ValueError(f"Simulation failed for case {case_ID}")

    WSE_upstream_history.append(WSE_at_monitoring_points[0])
    WSE_downstream_history.append(WSE_at_monitoring_points[1])
    H_upstream_history.append(WSE_at_monitoring_points[0] - bed_elev_at_monitoring_points[0])
    H_downstream_history.append(WSE_at_monitoring_points[1] - bed_elev_at_monitoring_points[1])
    Q_split_history.append(flow_partition)

    # read the measurement data
    measurement_file_name = '../measurement_data/measurement_results_case_'+str(current_case_ID)+'.dat'
    #print("Measurement data file name: ", measurement_file_name)
    df_experiment = pd.read_csv(measurement_file_name, sep='\s+')

    # Access the measurement values
    WSE_upstream = df_experiment['WSE_upstream'].values[0]
    WSE_downstream = df_experiment['WSE_downstream'].values[0]

    H_upstream = WSE_upstream - bed_elev_at_monitoring_points[0]       #upstream monitoring point
    H_downstream = WSE_downstream - bed_elev_at_monitoring_points[1]   #downstream monitoring point

    Q_split_exp = df_experiment['Flow_Fraction_open'].values[0]

    objective_function_value_Water_Depth, objective_function_value_Q_split = \
        compute_error(bFullWidth, H_upstream, H_downstream, H_upstream_history[-1], H_downstream_history[-1], Q_split_exp, Q_split_history[-1])

    objective_function_value = objective_function_value_Water_Depth + objective_function_value_Q_split

    Objective_function_history.append(objective_function_value)
    Objective_function_history_Water_Depth.append(objective_function_value_Water_Depth)
    Objective_function_history_Q_split.append(objective_function_value_Q_split)

    return objective_function_value

def compute_error(bFullWidth, H_upstream_exp, H_downstream_exp, H_upstream_sim, H_downstream_sim, Q_split_exp, Q_split_sim):
    """
    A function to compute the error between simulation and measurement.

    Parameters
    ----------
    bFullWidth: Whether the case is full width or not (half width)
    H_upstream_exp: Water depth of the upstream from measurement
    H_downstream_exp: Water depth of the downstream from measurement
    H_upstream_sim: Water depth of the upstream from simulation
    H_downstream_sim: Water depth of the downstream from simulation
    Q_split_exp: Flow discharge split from measurement
    Q_split_sim: Flow discharge split from simulation

    Returns
    -------

    """


    objective_function_value_Water_Depth = abs(H_upstream_exp - H_upstream_sim) / H_upstream_exp + \
                                           abs(H_downstream_exp - H_downstream_sim) / H_downstream_exp

    objective_function_value_Q_split = 0.0
    if not bFullWidth:
        objective_function_value_Q_split = abs(Q_split_sim - Q_split_exp)

    return objective_function_value_Water_Depth, objective_function_value_Q_split


def save_calibration_results(calibrated_value):
    """
    Save the calibration results to files: simulation_results.dat, calibrated_b_Coefficient.dat
    Returns
    -------
    calibrated_value: float, the calibrated value for b_coefficient

    """

    

    with open('calibration_results/case_'+str(current_case_ID)+'/simulation_results.dat', 'w') as f:
        f.write('b_Coefficient error_total error_H error_Q_split H_upstream H_downstream Q_split\n')

        for b_Coefficient, error_total, error_H, error_Q_split, H_upstream, H_downstream, Q_split in zip(
                    b_Coefficient_history,
                    Objective_function_history,
                    Objective_function_history_Water_Depth,
                    Objective_function_history_Q_split,
                    H_upstream_history,
                    H_downstream_history,
                    Q_split_history):

            sublist = [b_Coefficient, error_total, error_H, error_Q_split, H_upstream, H_downstream, Q_split]

            #print(f"sublist = {sublist}")

            # Join each sublist into a string with spaces separating elements
            f.write(' '.join(map(str, sublist)) + '\n')

    # write the calibrated b_coefficient to result file
    with open('calibration_results/case_'+str(current_case_ID)+'/calibrated_b_Coefficient.dat', 'w') as file:
        file.write(str(calibrated_value))

def plot_calibration_history():
    """
    Plot the calibration history for current case by reading the simulation_results.dat file.
    """

    print("Plotting the calibration history for case ", current_case_ID)

    #read the simulation_results.dat file
    simulation_results_file = 'calibration_results/case_'+str(current_case_ID)+'/simulation_results.dat'
    df_simulation_results = pd.read_csv(simulation_results_file, sep='\s+')

    b_Coefficient_history = df_simulation_results['b_Coefficient'].values
    Objective_function_history = df_simulation_results['error_total'].values
    Objective_function_history_Water_Depth = df_simulation_results['error_H'].values
    Objective_function_history_Q_split = df_simulation_results['error_Q_split'].values
    H_upstream_history = df_simulation_results['H_upstream'].values
    H_downstream_history = df_simulation_results['H_downstream'].values
    Q_split_history = df_simulation_results['Q_split'].values

    #plot the calibration history using scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(b_Coefficient_history, Objective_function_history, label='Total error')

    #add labels and title
    plt.xlabel('b_Coefficient [1/m]', fontsize=16)
    plt.ylabel('Error', fontsize=16)
    plt.title('Calibration History', fontsize=16)

    plt.tick_params(axis='both', which='major', labelsize=12)

    #save the figure to file
    plt.savefig("calibration_results/case_"+str(current_case_ID)+"/calibration_history.png", dpi=300, bbox_inches='tight', pad_inches=0.02)

    #show the plot
    #plt.show()

    plt.close()

def plot_optimization_results(parameter_bounds):
    """
    plot the Gaussain optimization result.

    :return:
    """

    #read in 'gp_result_'+str(current_case_ID)+'.pkl' data
    result = load('calibration_results/case_'+str(current_case_ID)+'/gp_result.pkl')

    #print(result)
    #exit()

    # Plot the model, sampling points, and confidence interval
    #plot_gaussian_process(result)
    #plt.show()

    # Get the evaluated points (X) and corresponding function values (Y)
    X = np.array(result.x_iters)
    Y = np.array(result.func_vals)

    # Fit a GaussianProcessRegressor on the data to get the prediction
    gp = GaussianProcessRegressor(kernel=1**2 * Matern(length_scale=1, nu=2.5),
                                  n_restarts_optimizer=2, noise=(noise_level) ** 2,
                                  normalize_y=True, random_state=822569775)
    gp.fit(X, Y)

    # Define points to plot the model's prediction
    x_vals = np.linspace(parameter_bounds[0][0], parameter_bounds[0][1], 100).reshape(-1, 1)
    y_pred, sigma = gp.predict(x_vals, return_std=True)

    # Plot the model (prediction) and confidence interval
    plt.figure(figsize=(6, 4))

    # Plot the Gaussian process predicted mean (model)
    plt.plot(x_vals, y_pred, 'g--', label='Predicted model error mean')

    # Plot the confidence interval (±1.96 standard deviations)
    plt.fill_between(x_vals.ravel(),
                     y_pred - 1.96 * sigma,
                     y_pred + 1.96 * sigma,
                     alpha=0.2, color='green',
                     edgecolor='none',
                     label='95\% confidence interval')

    # Plot the actual sampled points
    plt.plot(X, Y, 'ro', label="Sampled $b$ points")

    # Add labels and title
    #plt.title("Gaussian Process Model and Confidence Interval")
    plt.xlabel("$b$ value [1/m]", fontsize=16)

    plt.ylabel("Simulation model error", fontsize=16)

    # show the ticks on both axes and set the font size
    plt.tick_params(axis='both', which='major', labelsize=12)

    # show legend, set its location, font size, and turn off the frame
    plt.legend(loc='upper left', fontsize=14, frameon=False)

    # save the figure to file
    plt.savefig("calibration_results/case_"+str(current_case_ID)+"/calibration.png", dpi=300, bbox_inches='tight', pad_inches=0.02)

    # Show the plot
    # plt.show()

def run_case_with_optimized_b_Coefficient(case_ID, faceless=True):
    """
    run the case with the optimized b_coefficient
    """

    #read the calibrated parameter (b_coefficient)
    calibrated_b_Coefficient_file = 'calibration_results/case_'+str(case_ID)+'/calibrated_b_Coefficient.dat'

    # Open the file and read the single float value
    with open(calibrated_b_Coefficient_file, 'r') as file:
        b_Coefficient = float(file.read().strip())  # Read and convert to float

        print("The calibrated b_coefficient is {}".format(b_Coefficient))

    #run the case with the optimized b_coefficient
    PorosityDrag_MaterialID = 1
    PorosityDrag_MaterialName = 'LWD'
    simulation_result = run_one_HEC_RAS_case_PorosityDragCoefficient(case_ID, PorosityDrag_MaterialID, b_Coefficient, PorosityDrag_MaterialName, faceless=faceless)

    if simulation_result > 0:
        print("The simulation is successful")
    else:
        print("The simulation is failed")

    

