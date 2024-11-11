# %% Module imports
# Get performance model for each subsystem
import PySAM.Pvsamv1 as pv_model
import PySAM.Windpower as wind_model
import PySAM.Battery as battery_model

# Get function for managing hybrid variables and simulations
from PySAM.Hybrids.HybridSystem import HybridSystem

# To load inputs from SAM
import json

# To organize and plot outputs from simulation
import pandas as pd
import pysam_helpers
import load_inspection_helpers

# %% Load the inputs from the JSON file
# Note that for the Hybrid System, we use a single JSON file rather than a file per module.
# The JSON file referenced here is from SAM code generator for a PV Wind Battery sytem with a
# Single Owner financial model
inputs_file = 'data/PySam_Inputs/Hybrid_Project/Hybrid.json'
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

# Change the minimum battery SoC to 10%
m.battery.value("batt_minimum_SOC", 10)

#%% Run a simulation
m.execute()

# %% Store some outputs
# Be careful to use the correct module names as defined by the HybridSystem() function:
#     pv, pvwatts, wind, gensys, battery, fuelcell
#     _grid, singleowner, utilityrate5, host_developer
pvannualenergy = m.pv.Outputs.annual_energy
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

# Create a dictionary of DataFrames with the outputs from each model
pv_model_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m.pv)
wind_model_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m.wind)
battery_model_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m.battery)
grid_model_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m._grid)
single_owner_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m.singleowner)

# Generate some plots
date_start = '2012-07-27 00:00:00'
date_end = '2012-07-28 00:00:00'

pysam_helpers.plot_values_by_time_range(df=pv_model_outputs['Lifetime 5 Minute Data'], start_time=date_start, end_time=date_end, y_columns=['gen'])
pysam_helpers.plot_values_by_time_range(df=wind_model_outputs['5 Minute Data'], start_time=date_start, end_time=date_end, y_columns=['gen'])
pysam_helpers.plot_values_by_time_range(df=battery_model_outputs['Lifetime 5 Minute Data'], start_time=date_start, end_time=date_end, y_columns=['batt_SOC'])
pysam_helpers.plot_values_by_time_range(df=battery_model_outputs['Lifetime 5 Minute Data'], start_time=date_start, end_time=date_end, y_columns=['batt_to_grid', 'system_to_batt', 'system_to_grid'])

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

m_leap.pv.SolarResource.solar_resource_file = 'data/222628_32.73_-117.18_2022_interpolated_LEAP.csv'
m_leap.wind.Resource.wind_resource_filename = 'data/wind_speeds/sd_2012_5m_LEAP.csv'

# Change the minimum battery SoC to 10%
m_leap.battery.value("batt_minimum_SOC", 10)

m_leap.execute()

pv_model_leap_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m_leap.pv)
wind_model_leap_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m_leap.wind)
battery_model_leap_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m_leap.battery)
grid_model_leap_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m_leap._grid)
single_owner_leap_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m_leap.singleowner)

# %% Find the Feb 29 energy production as a share of annual energy production
# First, pull the relevant columns from each module output DataFrame into one DataFrame for the 
# system output
pv_df = pv_model_leap_outputs['Lifetime 5 Minute Data'].reset_index()
pv_df = pv_df.loc[:,['Datetime', 'subarray1_dc_gross', 'ac_gross']]
pv_df.rename(columns={'subarray1_dc_gross':'PV DC Gross (kW)', 'ac_gross': 'PV AC Gross (kW)'}, inplace=True)
batt_df = battery_model_leap_outputs['Lifetime 5 Minute Data'].reset_index()
batt_df = batt_df.loc[:,['Datetime', 'batt_to_grid', 'system_to_batt', 'system_to_grid']]
wind_df = wind_model_leap_outputs['5 Minute Data'].reset_index()
wind_df = wind_df.loc[:,['Datetime', 'gen']]
wind_df.rename(columns={'gen':'Wind AC Gross (kW)'}, inplace=True)
system_df = pd.merge(pv_df, batt_df, on='Datetime', how='inner')
system_df = pd.merge(system_df, wind_df, on='Datetime', how='inner')

# Pull together each source of generation to show the total
system_df['PV + Battery Generation (kW)'] = system_df['system_to_batt'] + system_df['system_to_grid'] + system_df['batt_to_grid']
system_df['Generation to Grid (kW)'] = system_df['system_to_grid'] + system_df['Wind AC Gross (kW)'] + system_df['batt_to_grid']
system_df['Total Generation (kW)'] = system_df['PV + Battery Generation (kW)'] + system_df['Wind AC Gross (kW)']

# This plot demonstrates that the 'ac_gross' variable in the PV model output DataFrame is 
# accounting for both energy sent to the battery and energy sent to the grid. 
# To break up the generation by subsystem completely, we need to isolate the variables for the PV 
# production that goes to the grid (not the battery), the wind production that goes to the grid 
# (not) the battery, and the battery energy that goes to the grid.
# In practice, the wind production here is so small that we can just assume that it always goes 
# straight to the grid.
date_start = '2012-07-27 00:00:00'
date_end = '2012-07-28 00:00:00'
system_df.set_index('Datetime', inplace=True)
pysam_helpers.plot_values_by_time_range(df=system_df, start_time=date_start, end_time=date_end, y_columns=['PV AC Gross (kW)', 'Generation to Grid (kW)', 'Total Generation (kW)'])

# Find the leap day generation
system_df['Power to Grid (kW)'] = system_df['batt_to_grid'] + system_df['system_to_grid'] + system_df['Wind AC Gross (kW)']
system_df['Energy to Grid (kWh)'] = system_df['Power to Grid (kW)'] * (5/60)
# Note that the indexing is a little wonky because we're spoofing Feb 29 data as Feb 28 when 
# running the SAM simulation
feb_29_start = pd.Timestamp(f'2012-02-28 00:00:00')
feb_29_end = pd.Timestamp(f'2012-03-01 00:00:00')
system_df.reset_index(inplace=True)
system_df['Datetime'] = pd.to_datetime(system_df['Datetime'])
year_1_mask = system_df['Datetime'] < pd.Timestamp(f'2013-01-01 00:00:00')
feb_29_mask = (system_df['Datetime'] >= feb_29_start) & (system_df['Datetime'] < feb_29_end)
leap_day_generation_kwh = system_df.loc[feb_29_mask, 'Energy to Grid (kWh)'].sum()
year_1_generation_kwh = system_df.loc[year_1_mask, 'Energy to Grid (kWh)'].sum()
leap_day_generation_percent = (leap_day_generation_kwh / year_1_generation_kwh) * 100

# Print the results
print(f"The generation on leap day is: {leap_day_generation_kwh:.2f} kWh.")
print(f"As a fraction of annual generation for year 1, the generation on leap day is: {leap_day_generation_percent:.2f}%")

# %% Compare load to generation
# Build a dataframe of load data
load = load_inspection_helpers.format_load_data(load_filepath='data/Project 2 - Load Profile.xlsx')
load['Load (kW)'] = load['Load (MW)'] * 1000

# Merge the two dataframes
merged = pd.merge(system_df, load, on='Datetime', how='inner')

merged['Excess Generation (kW)'] = merged['Generation to Grid (kW)'] - merged['Load (kW)']
merged['Unmet Load (kW)'] = merged['Load (kW)'] - merged['Generation to Grid (kW)']

# Calculate some metrics on the reliability
threshold = 1e-3 # This is 0.1% of peak load
unmet_threshold = (merged['Load (kW)'].max()) * threshold
unmet_load_mask = merged['Unmet Load (kW)'] > unmet_threshold
unmet_load_instances = unmet_load_mask.sum()
unmet_load_magnitude = merged['Unmet Load (kW)'][unmet_load_mask].sum()
unmet_load_percent = (unmet_load_magnitude / (merged['Load (kW)'].sum())) * 100
merged[unmet_load_mask].plot(x='Datetime', y='Unmet Load (kW)')

# Define some metrics on the cost
lcoe_nom = m.singleowner.Outputs.lcoe_nom
net_cost = m.singleowner.Outputs.cost_installed

# Calculate some metrics on the excess generation
curtailment_mask = merged['Excess Generation (kW)'] > unmet_threshold
curtailment_instances = curtailment_mask.sum()
curtailment_magnitude = merged['Excess Generation (kW)'][curtailment_mask].sum()
curtailment_percent = (curtailment_magnitude * (5/60) / year_1_generation_kwh) * 100
merged[curtailment_mask].plot(x='Datetime', y='Excess Generation (kW)')

# Print some relevant outputs
print(f"The number of instances where the load exceeds the generation by {threshold*100}% is: {unmet_load_instances:d}")
print(f"{unmet_load_percent:.2f}% of annual load was not met by generation")
print(f"{curtailment_percent:.2f}% of annual energy production in Year 1 was curtailed")
print(f"The installed cost of the system is: ${net_cost:.2e}")
print(f"The calculated nominal LCOE is: {(lcoe_nom/100):.2f} $/kWh")

# Plot some results
date_start = '2012-07-27 00:00:00'
date_end = '2012-07-28 00:00:00'
merged.set_index('Datetime', inplace=True)
pysam_helpers.plot_values_by_time_range(df=merged, start_time=date_start, end_time=date_end, y_columns=['Load (kW)', 'Generation to Grid (kW)', 'Total Generation (kW)'])
date_start = '2012-12-19 00:00:00'
date_end = '2012-12-20 00:00:00'
pysam_helpers.plot_values_by_time_range(df=merged, start_time=date_start, end_time=date_end, y_columns=['Load (kW)', 'Generation to Grid (kW)', 'Total Generation (kW)'])
# %%
