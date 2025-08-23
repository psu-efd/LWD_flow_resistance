import pyHMT2D

import numpy as np
from scipy.stats import truncnorm
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

import vtk
from vtk.numpy_interface import dataset_adapter as dsa

import os
import json
import shutil

from HEC_RAS_solver_module import run_one_HEC_RAS_case_ManningN, sample_HEC_RAS_results

plt.rc('text', usetex=True)  #allow the use of Latex for math expressions and equations
plt.rc('font', family='serif') #specify the default font family to be "serif"

def run_sample_HEC_RAS_results():
    case_ID = 8
    WSE_at_monitoring_points, discharge_at_monitoring_line_1, discharge_at_monitoring_line_2, total_discharge, flow_partition = sample_HEC_RAS_results(case_ID)

    print("WSE at monitoring points: ", WSE_at_monitoring_points)
    print("Discharge at monitoring line 1: ", discharge_at_monitoring_line_1)
    print("Discharge at monitoring line 2: ", discharge_at_monitoring_line_2)
    print("Total discharge: ", total_discharge)
    print("Flow partition: ", flow_partition)

def run_test_RAS_2D_Data():
    #go into the case directory
    os.chdir("RAS_case")

    case_ID = 8

    my_ras_2d_data = pyHMT2D.RAS_2D.RAS_2D_Data("lwd.p08.hdf")

    print("RAS2D_mesh_data class self-dump:")
    print("    hdf_filename = ", my_ras_2d_data.hdf_filename)
    print("    geometry_file_name = ", my_ras_2d_data.geometry_file_name)
    print("    terrain_hdf_filename = ", my_ras_2d_data.terrain_hdf_filename)
    print("    terrain_tiff_filename = ", my_ras_2d_data.terrain_tiff_filename)

    vtkFileNameList = my_ras_2d_data.saveHEC_RAS2D_results_to_VTK(lastTimeStep=True)
    print("VTK file name: ", vtkFileNameList[-1])

    #copy the vtk result file (only the last time step) to "cases" directory (one level above)
    shutil.copy(vtkFileNameList[-1], "../result_vtks/"+"case_"+str(case_ID).zfill(2)+"/result.vtk")

    #go back to the parent directory
    os.chdir("..")

def run_a_case():
     #define parameters
    case_ID = 8
    ManningN_MaterialID = 1
    ManningN = 1.15
    ManningN_MaterialName = 'LWD'

    #run the HEC-RAS case
    run_one_HEC_RAS_case_ManningN(case_ID, ManningN_MaterialID, ManningN, ManningN_MaterialName, bDeleteCaseDir=True)

def run_all_plans():
    #loop over all plans in the RAS project and run them
    #base HEC-RAS case directory
    base_case_dir = 'RAS_case'

    #go into the base case directory
    os.chdir(base_case_dir)
    
    #create a HEC-RAS model instance
    my_hec_ras_model = pyHMT2D.RAS_2D.HEC_RAS_Model(version="6.6", faceless=False)

    #initialize the HEC-RAS model
    my_hec_ras_model.init_model()

    #print("Hydraulic model name: ", my_hec_ras_model.getName())
    print("Hydraulic model version: ", my_hec_ras_model.getVersion())

    #open a HEC-RAS project (This is hard-code; needs to be changed for a specific case)
    my_hec_ras_model.open_project("lwd.prj")

    #get the list of plan names
    plan_count, plan_names = my_hec_ras_model.get_plan_names(IncludeOnlyPlansInBaseDirectory=False)

    print("There are ", plan_count, "plan(s) in current project.")
    print('Plan names: ', plan_names)  # returns plan names

    #loop over all plans and run them
    for plan_name in plan_names:
        print("Running plan ", plan_name)

        #set the current plan file name
        my_hec_ras_model.set_current_plan(plan_name)        

        #run the HEC-RAS model's current project
        bRunSucessful = my_hec_ras_model.run_model()

        if bRunSucessful:
            print("Run of plan ", plan_name, " is successful.")
        else:
            print("Run of plan ", plan_name, " is failed.")

    #save the project
    my_hec_ras_model.save_project()  #save the project to make sure the current plan is set correctly

    #close the HEC-RAS project
    my_hec_ras_model.close_project()

    #quit HEC-RAS
    my_hec_ras_model.exit_model()

    #go back to the parent directory
    os.chdir("..")

def kill_all_hec_ras():
    pyHMT2D.RAS_2D.helpers.kill_all_hec_ras()

if __name__ == "__main__":

    #available_versions = pyHMT2D.RAS_2D.get_installed_hec_ras_versions()
    #print(available_versions)

    #exit()

    run_all_plans()

    #run_a_case()

    #run_test_RAS_2D_Data()

    #run_sample_HEC_RAS_results()

    #kill_all_hec_ras()

    print('done')