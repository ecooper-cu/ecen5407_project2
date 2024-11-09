# %% Import each module in the order that it will be executed
import PySAM.Pvwattsv8 as pv_model
import PySAM.Windpower as wind_model
import PySAM.Battery as battery_model
from PySAM.Hybrids.HybridSystem import HybridSystem

import json # To load inputs from SAM

# To organize and plot outputs from simulation
import pysam_helpers

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

# print outputs
print(f'The annual generation from PV is: {pvannualenergy:.2e} kWh')
print(f'The annual generation from wind is: {windannualenergy:.2e} kWh')
print(f'The round trip efficiency of the battery is: {battrountripefficiency:.2f}%')
print(f"Annual System AC Energy in Year 1 was: {gridannualenergy:.2e} kWh")
print(f"The net present value (NPV) of the system is: ${npv:.2e}")

# %% Create a dictionary of DataFrames with the outputs from each model
pv_model_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m.pvwatts)
wind_model_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m.wind)
battery_model_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m.battery)
grid_model_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m._grid)
single_owner_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m.singleowner)

# %% Re-run the model using data with the Feb 29 values subbed in for Feb 28 values.
# We are using input files which represent the resource availability from the year 2012. However, 
# since 2012 is a leap year, SAM won't manage the leap day entries well. As such, the default 
# data (used above) has been modified to eliminate all entries from Feb 29, 2012. 
# Using that data provided us with a baseline for the expected plant output over the course of a 
# non-leap year. We now wish to use data where Feb 28 entries are replaced by values from Feb 29.
# The model outputs using that data can provide information on the expected output during a leap 
# day.

m_leap = HybridSystem([pv_model, wind_model, battery_model], 'singleowner')
m_leap.new()

unassigned = m_leap.assign(inputs)
print(unassigned)

m_leap.pvwatts.SolarResource.solar_resource_file = 'data/222628_32.73_-117.18_2022_interpolated_LEAP.csv'
m_leap.wind.Resource.wind_resource_filename = 'data/wind_speeds/sd_2012_5m_LEAP.csv'

m_leap.execute()

pv_model_leap_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m_leap.pvwatts)
wind_model_leap_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m_leap.wind)
battery_model_leap_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m_leap.battery)
grid_model_leap_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m_leap._grid)
single_owner_leap_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m_leap.singleowner)