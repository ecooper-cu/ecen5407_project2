# %% Module imports
### PySAM stuff
# Get performance model for each subsystem
import PySAM.Pvsamv1 as pv_model
import PySAM.Windpower as wind_model
import PySAM.Battery as battery_model

# Get function for managing hybrid variables and simulations
from PySAM.Hybrids.HybridSystem import HybridSystem

# To load inputs from SAM
import json

# To organize and plot outputs from PySAM simulation
import pandas as pd
import pysam_helpers

### Load stuff
import load_inspection_helpers

# %% Execute PySAM model
# Load the inputs from the JSON file
# Note that for the Hybrid System, we use a single JSON file rather than a file per module.
# The JSON file referenced here is from SAM code generator for a PV Wind Battery sytem with a
# Single Owner financial model
print("Loading inputs...")
inputs_file = 'data/PySam_Inputs/Hybrid_Project/Hybrid.json'
with open(inputs_file, 'r') as f:
        inputs = json.load(f)['input']

# Create the hybrid system model using performance model names as defined by the import 
# statements above. The string 'singleowner' indicates the financial model ('hostdeveloper' would
# be another option).
print("Creating hybrid system...")
m = HybridSystem([pv_model, wind_model, battery_model], 'singleowner')
m.new()

# Load the inputs from the JSON file into the main module
print("Assigning inputs to hybrid system....")
unassigned = m.assign(inputs) # returns a list of unassigned variables if any
print(unassigned)

# Forbid the battery from charging from the system – now it can't participate
m.battery.value("batt_dispatch_auto_can_charge", 0)

# Run a simulation
print("Running PySAM simulation...")

m.execute()

# Create a dictionary of DataFrames with the outputs from each model
pv_model_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m.pv)
wind_model_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m.wind)
battery_model_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m.battery)
grid_model_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m._grid)
single_owner_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m.singleowner)

# %% Compare load to generation
# Build a dataframe of load data
load = load_inspection_helpers.format_load_data(load_filepath='data/Project 2 - Load Profile.xlsx')
load['Load (kW)'] = load['Load (MW)'] * 1000

# Build a dataframe of generation data
pv_df = pv_model_outputs['Lifetime 5 Minute Data'].reset_index()
pv_gen = pv_df[['Datetime', 'subarray1_dc_gross', 'gen']]
pv_gen.rename(columns={'subarray1_dc_gross':'PV DC Gross (kW)', 'gen': 'PV Generation (kW)'}, inplace=True)
wind_df = wind_model_outputs['5 Minute Data'].reset_index()
wind_gen = wind_df[['Datetime', 'gen']]
wind_gen.rename(columns={'gen':'Wind Generation (kW)'}, inplace=True)
gen = pd.merge(pv_gen, wind_gen, on='Datetime', how='inner')
gen['Datetime'] = pd.to_datetime(gen['Datetime'])
gen['System Power (kW)'] = gen['PV DC Gross (kW)'] + gen['Wind Generation (kW)']

# Merge the two dataframes
merged = pd.merge(gen, load, on='Datetime', how='inner')

# Identify where generation exceeds load 
merged['Power Available for Battery (kW)'] =  merged['System Power (kW)'] - merged['Load (kW)']

# Set the battery power target equal to the difference
# (negative for charging, positive for discharging)
merged['Battery Power Target (kW)'] = merged['Power Available for Battery (kW)'] * -1

# %% Generate some plots
date_start = '2012-07-27 00:00:00'
date_end = '2012-07-28 00:00:00'
gen.set_index('Datetime', inplace=True)
pysam_helpers.plot_values_by_time_range(df=gen, start_time=date_start, end_time=date_end, y_columns=['PV Generation (kW)', 'Wind Generation (kW)'])

# %% Generate a csv with the dispatch target
merged['Battery Power Target (kW)'].to_csv('dispatch_target.csv', index=False)

# %% Generate a csv of system power output without the battery
gen.to_csv('data/PySAM_Outputs/baseline_system_output.csv')
# %%
