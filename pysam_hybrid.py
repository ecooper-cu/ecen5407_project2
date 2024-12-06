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
import pickle

# %% Load the inputs from the JSON file
# Note that for the Hybrid System, we use a single JSON file rather than a file per module.
# The JSON file referenced here is from SAM code generator for a PV Wind Battery sytem with a
# Single Owner financial model
store_case = True # set to False if you don't want to generate a new case / write over existing case
case_name = 'Baseline_System_No_Geothermal' # change name!
inputs_file = 'data/test_cases/Baseline_System_No_Geothermal/Baseline_with_Updated_Econ_metrics.json'
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

# %% Set the custom dispatch
old_dispatch = list(m.battery.BatteryDispatch.batt_custom_dispatch)

dispatch_df = pd.read_csv("data/test_cases/Baseline_System_No_Geothermal/dispatch_target_5min.csv")
new_dispatch = dispatch_df["Battery Power Target (kW)"].to_list()

# Confirm that you've updated the dispatch
print(new_dispatch == old_dispatch)

m.battery.BatteryDispatch.batt_custom_dispatch = new_dispatch
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
pv_ops_cost = m.pv.HybridCosts.om_fixed[0] + m.pv.HybridCosts.om_capacity[0]*m.pv.SystemDesign.system_capacity

# Wind Stuff
wind_capacity_kWac = m.wind.Farm.system_capacity
wind_size_x = max(m.wind.Farm.wind_farm_xCoordinates) # meters
wind_size_y = max(m.wind.Farm.wind_farm_yCoordinates) # meters
wind_land_area = wind_size_x * wind_size_y # square meters
wind_cost = m.wind.HybridCosts.total_installed_cost
wind_ops_cost = m.wind.HybridCosts.om_fixed[0] + m.wind.HybridCosts.om_capacity[0]*m.wind.Farm.system_capacity

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
# store outputs in dict
system_info = {
        'PV System Size': pv_capacity_kWdc,
        'PV AC Capacity': total_inverter_capacity,
        'PV AC:DC Ratio': pv_dcac_ratio,
        'PV System Span': pv_land_area,
        'PV Cost': pv_cost,
        'PV Operating Cost': pv_ops_cost,
        'Wind System Size': wind_capacity_kWac,
        'Wind System Span': wind_land_area,
        'Wind Cost': wind_cost,
        'Wind Operating Cost' : wind_ops_cost,
        'Battery Nominal Power': battery_power_kWdc,
        'Battery Capacity': battery_capacity_kWhdc,
        'Battery Discharge (Hours)': battery_time_at_max_discharge,
        'Battery Connection': battery_connection_type,
        'Battery Charge Efficiency': battery_charge_efficiency,
        'Battery Discharge Efficiency': battery_discharge_efficiency,
        'Battery Cost': battery_cost,
        'System Cost': system_cost
}
if store_case:
    pysam_helpers.store_system_info(case_name, system_info)


# %% Store the outputs so we can generate a test case analysis!! 
# Create a dictionary of DataFrames with the outputs from each model
pv_model_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m.pv)
wind_model_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m.wind)
battery_model_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m.battery)
#grid_model_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m._grid)
#single_owner_model_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m.singleowner)

#%% 
# Generate a system output dataframe
system_output_dict = {'pv':pv_model_outputs, 
                      'wind': wind_model_outputs,
                      'battery': battery_model_outputs
                      }
test_case = pysam_helpers.merge_subsystem_5min_dfs(system_output_dict)

if store_case:
        test_case.to_csv(f'data/test_cases/{case_name}/{case_name}.csv')

# %% Generate some plots
if 'Datetime' in test_case.columns:
        test_case.set_index('Datetime', inplace=True)

date_start = '2012-07-27 00:00:00'
date_end = '2012-07-28 00:00:00'

columns_to_plot = ['Battery Discharge Power (kW)', 'PV to Grid (kW)', 'Net Wind Generation (kW)', 'Load (kW)']
pysam_helpers.plot_values_by_time_range(df=test_case, start_time=date_start, end_time=date_end, y_columns=columns_to_plot)
pysam_helpers.plot_values_by_time_range(df=battery_model_outputs['Lifetime 5 Minute Data'], start_time=date_start, end_time=date_end, y_columns=['batt_SOC'])
# %% Save the dataframes as pickled objects
# Saving/loading the dataframes in a CSV structure takes forever.
# We can save the data in a more efficient way with pickled objects:
im_a_pickle_dict = {
        'pv_model_outputs' : pv_model_outputs['Lifetime 5 Minute Data'],
        'wind_model_outputs' : wind_model_outputs['5 Minute Data'],
        'battery_model_outputs': battery_model_outputs['Lifetime 5 Minute Data'],
        'grid_model_outputs' : grid_model_outputs['Lifetime 5 Minute Data'],
        'single_owner_model_outputs': single_owner_model_outputs['Lifetime 5 Minute Data']
}

for pickle_filename, model_output_df in im_a_pickle_dict.items():
        filepath = '~/Downloads' + pickle_filename + '.pkl'
        with open(filepath, 'wb') as f:
                pickle.dump(model_output_df, f)
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

# Merge the two dataframes
system_df.reset_index(inplace=True)
system_df['Datetime'] = pd.to_datetime(system_df['Datetime'])
merged = pd.merge(system_df, load, on='Datetime', how='inner')

merged['Excess Generation (kW)'] = merged['Generation to Grid (kW)'] - merged['Load (kW)']
geothermal_capacity_kW = 77 * 0.95 * 1000
merged['Unmet Load (kW)'] = merged['Load (kW)'] - merged['Generation to Grid (kW)'] - geothermal_capacity_kW

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
