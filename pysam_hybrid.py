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

#%% Run a simulation
m.execute()

# %% Store some outputs
# Be careful to use the correct module names as defined by the HybridSystem() function:
#     pv, pvwatts, wind, gensys, battery, fuelcell
#     _grid, singleowner, utilityrate5, host_developer
# Relationships between dependent variables can be found in runtime/ui files:
# https://github.com/NREL/SAM/blob/develop/deploy/runtime/ui/

# PV Stuff
pv_capacity_kWdc = m.pv.SystemDesign.system_capacity
pv_land_area = m.pv.HybridCosts.land_area * 4046.86 # square meters
inverter_model = int(m.pv.Inverter.inverter_model)
inv_snl_paco = m.pv.Inverter.inv_snl_paco # Inverter Sandia Maximum AC Power [Wac]
inv_ds_paco = m.pv.Inverter.inv_ds_paco # Inverter Datasheet Maximum AC Power [Wac]
inv_pd_paco = m.pv.Inverter.inv_pd_paco # Inverter Partload Maximum AC Power [Wac]
inv_cec_cg_paco = m.pv.Inverter.inv_cec_cg_paco # Inverter Coefficient Generator Max AC Power [Wac]
inverter_power = [inv_snl_paco, inv_ds_paco, inv_pd_paco, inv_cec_cg_paco][inverter_model]
inverter_count = m.pv.SystemDesign.inverter_count
total_inverter_capacity = inverter_power * inverter_count / 1000 # [kWac]
pv_dcac_ratio = pv_capacity_kWdc / total_inverter_capacity
pv_cost = m.pv.HybridCosts.total_installed_cost

# Wind Stuff
wind_capacity_kWac = m.wind.Farm.system_capacity
wind_size_x = max(m.wind.Farm.wind_farm_xCoordinates) # meters
wind_size_y = max(m.wind.Farm.wind_farm_yCoordinates) # meters
wind_land_area = wind_size_x * wind_size_y # square meters
wind_cost = m.wind.HybridCosts.total_installed_cost


# Battery Stuff
battery_power_kWdc = m.battery.BatterySystem.batt_power_discharge_max_kwdc 
battery_capacity_kWhdc = m.battery.BatterySystem.batt_computed_bank_capacity
battery_dc_ac_efficiency = m.battery.BatterySystem.batt_dc_ac_efficiency
battery_time_at_max_discharge = battery_capacity_kWhdc / battery_power_kWdc
if m.battery.BatterySystem.batt_ac_or_dc == 0:
        battery_connection_type = 'DC'
        battery_charge_efficiency = m.battery.BatterySystem.batt_dc_dc_efficiency
        battery_discharge_efficiency = m.battery.BatterySystem.batt_dc_ac_efficiency
elif m.battery.BatterySystem.batt_ac_or_dc == 1:
        battery_connection_type = 'AC'
        battery_charge_efficiency = m.battery.BatterySystem.batt_ac_dc_efficiency
        battery_discharge_efficiency = m.battery.BatterySystem.batt_dc_ac_efficiency
else:
        battery_connection_type = 'dis-'
        battery_charge_efficiency = 0
        battery_discharge_efficiency = 0
battery_cost = m.battery.HybridCosts.total_installed_cost

system_cost = pv_cost + wind_cost + battery_cost

# print outputs
print(f'The total installed cost for the system is: ${system_cost:.2f}')
print('\n')
print(f'The PV system size is: {pv_capacity_kWdc:.2f} kW (DC)')
print(f'The PV AC capacity is: {total_inverter_capacity:.2f} kW (AC)')
print(f'The PV system DC:AC ratio is: {pv_dcac_ratio:.2f}')
print(f'The PV system spans {pv_land_area:.2f} square meters')
print(f'The total installed cost for the PV system is: ${pv_cost:.2f}')
print('\n')
print(f'The wind system size is: {wind_capacity_kWac:.2f} kW (AC)')
print(f'The wind system spans {wind_land_area:.2f} square meters')
print(f'The total installed cost for the wind system is: ${wind_cost:.2f}')
print('\n')
print(f'The battery nominal power is: {battery_power_kWdc:.2f} kW (DC)')
print(f'The battery capacity is: {battery_capacity_kWhdc:.2f} kWh (DC)')
print(f'The battery can discharge for {battery_time_at_max_discharge} hours at rated power.')
print(f'The battery is {battery_connection_type} connected.')
print(f'The battery charges at {battery_charge_efficiency:.2f}% efficiency')
print(f'The battery discharges at {battery_discharge_efficiency:.2f}% efficiency')
print(f'The total installed cost for the battery system is: ${battery_cost:.2f}')

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

# %% Inspect model outputs
# First, pull the relevant columns from each module output DataFrame into one DataFrame for the 
# system output
pv_df = pv_model_outputs['Lifetime 5 Minute Data'].reset_index()
pv_df = pv_df.loc[:,['Datetime', 'subarray1_dc_gross', 'ac_gross']]
pv_df.rename(columns={'subarray1_dc_gross':'PV DC Gross (kW)', 'ac_gross': 'PV AC Gross (kW)'}, inplace=True)
batt_df = battery_model_outputs['Lifetime 5 Minute Data'].reset_index()
batt_df = batt_df.loc[:,['Datetime', 'batt_to_grid', 'system_to_batt', 'system_to_grid']]
wind_df = wind_model_outputs['5 Minute Data'].reset_index()
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

# %% Compare load to generation
# Build a dataframe of load data
load = load_inspection_helpers.format_load_data(load_filepath='data/Project 2 - Load Profile.xlsx')
load['Load (kW)'] = load['Load (MW)'] * 1000

# Merge the two dataframes
system_df.reset_index(inplace=True)
system_df['Datetime'] = pd.to_datetime(system_df['Datetime'])
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

# Find the year 1 generation
system_df['Power to Grid (kW)'] = system_df['batt_to_grid'] + system_df['system_to_grid'] + system_df['Wind AC Gross (kW)']
system_df['Energy to Grid (kWh)'] = system_df['Power to Grid (kW)'] * (5/60)
year_1_mask = system_df['Datetime'] < pd.Timestamp(f'2013-01-01 00:00:00')
year_1_generation_kwh = system_df.loc[year_1_mask, 'Energy to Grid (kWh)'].sum()

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
# This plot demonstrates the dispatch stack during the highest load day (June 27th)
date_start = '2012-07-27 00:00:00'
date_end = '2012-07-28 00:00:00'
merged.set_index('Datetime', inplace=True)
pysam_helpers.plot_values_by_time_range(df=merged, start_time=date_start, end_time=date_end, y_columns=['Load (kW)', 'Generation to Grid (kW)', 'Total Generation (kW)'])

# These plots demonstrates the dispatch stack during a day where we fail to meet the load
date_start = '2012-12-19 00:00:00'
date_end = '2012-12-20 00:00:00'
pysam_helpers.plot_values_by_time_range(df=merged, start_time=date_start, end_time=date_end, y_columns=['Load (kW)', 'Generation to Grid (kW)', 'Total Generation (kW)'])
pysam_helpers.plot_values_by_time_range(df=battery_model_outputs['Lifetime 5 Minute Data'], start_time=date_start, end_time=date_end, y_columns=['batt_SOC'])
# %%
