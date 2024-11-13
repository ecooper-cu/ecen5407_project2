# %% Import each module in the order that it will be executed
import PySAM.Pvwattsv8 as pv_model
import PySAM.Windpower as wind_model
import PySAM.Battery as battery_model
import pandas as pd
import os
from PySAM.Hybrids.HybridSystem import HybridSystem

import json # To load inputs from SAM


# %% Load the inputs from the JSON file
# Note that for the Hybrid System, we use a single JSON file rather than a file per module.
# The JSON file referenced here is from SAM code generator for a PVWatts Wind Battery sytem with a
# Single Owner financial model
inputs_file = 'data/PySam_Inputs/Hybrid_System_Demo/Hybrid_Demo.json'
with open(inputs_file, 'r') as f:
        inputs = json.load(f)['input']

# %% Create the hybrid system model using performance model names as defined by the import 
# statements above. The string 'singleowner' indicates the financial model ('hostdeveloper' would
# be another option).
m = HybridSystem([pv_model, wind_model, battery_model], 'singleowner')
m.new()

# %% Load the inputs from the JSON file into the main module
unassigned = m.assign(inputs) # returns a list of unassigned variables if any
print(unassigned)

#%% Run a simulation
m.execute()

# %% Store some outputs
# Be careful to use the correct module names as defined by the HybridSystem() function:
#     pv, pvwatts, wind, gensys, battery, fuelcell
#     _grid, singleowner, utilityrate5, host_developer
pvannualenergy = m.pvwatts.Outputs.annual_energy
windannualenergy = m.wind.Outputs.annual_energy
battrountripefficiency = m.battery.Outputs.average_battery_roundtrip_efficiency
gridannualenergy = m._grid.SystemOutput.annual_energy
npv = m.singleowner.Outputs.project_return_aftertax_npv
windenergytime = m.wind.Outputs.annual_energy_distribution_time # time series of wind production
columns = windenergytime[0][1:] 
row_indices = [row[0] for row in windenergytime[1:]]  
values = [row[1:] for row in windenergytime[1:]]      
windseries_df = pd.DataFrame(values, columns=columns, index=row_indices) # df of windseries

# print outputs
print(pvannualenergy)
print(windannualenergy)
print(battrountripefficiency)
print(gridannualenergy)
print(npv)
# %% create same model with leap year data
m = HybridSystem([pv_model, wind_model, battery_model], 'singleowner')
m.new()
inputs['windpower']['wind_resource_filename'] = os.path.join('data', 'wind_speeds', 'sd_2012_5m_leap.csv')
unassigned = m.assign(inputs) # returns a list of unassigned variables if any
print(unassigned)
m.execute()
# %% pull leap year data & combine into single dataframe
windenergytime_1 = m.wind.Outputs.annual_energy_distribution_time # time series of wind production
columns = windenergytime[0][1:] 
row_indices = [row[0] for row in windenergytime[1:]]  
values = [row[1:] for row in windenergytime[1:]]      
windseries_df_1 = pd.DataFrame(values, columns=columns, index=row_indices) # df of windseries
cols = range(1, 367)

# %%
windseries_df.insert(31+28, 368, windseries_df_1[31+28].values)
windseries_df.columns = cols
#%%