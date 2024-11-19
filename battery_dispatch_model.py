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
gen['RE Power (kW)'] = gen['PV Generation (kW)'] + gen['Wind Generation (kW)']

#Add geothermal generation, assuming flat generation on 77MW capacity with 95% capacity factor
gen['Net Power (kW)'] = gen['RE Power (kW)'] + 77 * 1000 * 0.95

# Merge the two dataframes
merged = pd.merge(gen, load, on='Datetime', how='inner')

# Identify where generation exceeds load 
merged['Power Available for Battery (kW)'] =  merged['Net Power (kW)'] - merged['Load (kW)']

# Set the battery power target equal to the difference
# (negative for charging, positive for discharging)
merged['Battery Power Target (kW)'] = merged['Power Available for Battery (kW)'] * -1

# %% Generate some plots
date_start = '2012-12-21 00:00:00'
date_end = '2012-12-22 00:00:00'
#merged.set_index('Datetime', inplace=True)
pysam_helpers.plot_values_by_time_range(df=merged, start_time=date_start, end_time=date_end, y_columns=['PV Generation (kW)', 'Wind Generation (kW)', 'Load (kW)', 'Power Available for Battery (kW)'])

# %% Generate a csv with the dispatch target
merged['Battery Power Target (kW)'].to_csv('data/PySam_Inputs/Hybrid_Project/dispatch_target_5min.csv', index=False)

# %% Generate a csv of system power output without the battery
merged.to_csv('data/PySAM_Outputs/baseline_system_output.csv')
# %% From initial analysis, the battery is highly underutilized:
# Grab discharging targets from the battery dispatch time series
merged['Battery Discharge Target (kW)'] = merged['Battery Power Target (kW)'][merged['Battery Power Target (kW)'] > 0]

# Calculate energy discharged in year 1 based on these targets
merged['Battery Energy Discharged (kWh)'] = merged['Battery Discharge Target (kW)'] * (5/60)
year_1_mask = merged.index < pd.Timestamp(f'2013-01-01 00:00:00')
year_1_battery_generation_kwh = merged.loc[year_1_mask, 'Battery Energy Discharged (kWh)'].sum()

# We need a baseline against which to evaluate the utilization of the battery. We will choose an 
# 'ideal' dispatch of 100% of the battery's capacity per day as this baseline.
battery_capacity_kWhdc = m.battery.BatterySystem.batt_computed_bank_capacity
ideal_annual_battery_generation_kwh = battery_capacity_kWhdc * 365

# Compare actual energy discharged in year 1 against ideal energy discharged in year 1
battery_capacity_factor = year_1_battery_generation_kwh / ideal_annual_battery_generation_kwh
print(f"Battery utilization is {battery_capacity_factor * 100:.2f}% of the ideal utilization.")

# %%
