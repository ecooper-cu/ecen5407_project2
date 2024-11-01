# %%
# Import each module in the order that it will be executed
import PySAM.Pvsamv1 as PVSAM
import PySAM.Grid as Grid
import PySAM.Utilityrate5 as UtilityRate
import PySAM.Singleowner as SingleOwner

import json # To load inputs from SAM

# To organize and plot outputs from simulation
import pysam_helpers
# %% Create a new instance of each module
pvbatt_model = PVSAM.new()
grid = Grid.from_existing(pvbatt_model)
utility_rate = UtilityRate.from_existing(pvbatt_model)
single_owner = SingleOwner.from_existing(pvbatt_model)

# %% Load the inputs from the JSON file for each module
dir = 'data/PySAM_Inputs/PV_Battery_System_Demo/'
prefix = 'PV_Battery_System_Demo_'
file_names = ["pvsamv1", "grid", "utilityrate5", "singleowner"]
modules = [pvbatt_model, grid, utility_rate, single_owner]
for f, m in zip(file_names, modules):
    filepath = dir + prefix + f + '.json'
    print(f"Loading inputs from {filepath}")
    with open(filepath, 'r') as file:
        data = json.load(file)
        # Loop through each key-value pair and set the module inputs
        for k, v in data.items():
            # Note: I'm ignoring any 'adjustment factors' here, but these can be set afterwards.
            # See: https://nrel-pysam.readthedocs.io/en/main/modules/Pvsamv1.html#adjustmentfactors-group
            if k != 'number_inputs' and 'adjust_' not in k:
                m.value(k, v)

# %% Run the modules in order
for m in modules:
    m.execute()

# %% Create a dictionary of DataFrames with the outputs from each model
pvbatt_model_outputs = pysam_helpers.parse_model_outputs_into_dataframes(pvbatt_model)
grid_model_outputs = pysam_helpers.parse_model_outputs_into_dataframes(grid)
utility_rate_outputs = pysam_helpers.parse_model_outputs_into_dataframes(utility_rate)
single_owner_outputs = pysam_helpers.parse_model_outputs_into_dataframes(single_owner)

# %% Print some example results to show that execution was successful
num_cycles = pvbatt_model_outputs['Lifetime 30 Minute Data']['batt_cycles'].to_list()[-1]
computed_capacity_kwh = pvbatt_model.value('batt_computed_bank_capacity')
print(f"{computed_capacity_kwh:,.0f} kWh battery cycled {num_cycles} times.\n")

analysis_period = int(pvbatt_model.value('analysis_period'))
energy_to_grid = pvbatt_model_outputs['Annual_Data']['annual_export_to_grid_energy'].to_list()
print(f"Annual system AC output in year {analysis_period} = {energy_to_grid[-1]:.3f} kWh")

npv = single_owner_outputs['Single Values']['project_return_aftertax_npv'].values[0]
print(f'Net Present Value: ${npv:.2f}')
lcoe_nom = single_owner_outputs['Single Values']['lcoe_nom'].values[0]
print(f'Nominal LCOE: {lcoe_nom:.2f} c/kWh')
# %% Generate a plot of some of the system outputs
pysam_helpers.plot_values_by_time_range(df=pvbatt_model_outputs['Lifetime 30 Minute Data'], start_time='2012-07-27 00:00:00', end_time='2012-07-28 00:00:00', y_columns=['batt_SOC'])
pysam_helpers.plot_values_by_time_range(df=pvbatt_model_outputs['Lifetime 30 Minute Data'], start_time='2012-07-27 00:00:00', end_time='2012-07-28 00:00:00', y_columns=['batt_power', 'system_to_grid', 'grid_to_batt'])
# %%
