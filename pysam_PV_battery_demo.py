# %%
# Import each module in the order that it will be executed
import PySAM.Pvsamv1 as PVSAM
import PySAM.Grid as Grid
import PySAM.Utilityrate5 as UtilityRate
import PySAM.Singleowner as SingleOwner

# Also import JSON to load the inputs
import json

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
            if k != 'number_inputs' and 'adjust_' not in k:
                m.value(k, v)

# %% Run the modules in order
for m in modules:
    m.execute()

# %% Print some example results to show that execution was successful
print(f"{pvbatt_model.value('batt_computed_bank_capacity'):,.0f} kWh battery cycled {pvbatt_model.Outputs.batt_cycles[-1]} times.\n")
print(f"Annual system AC output in year {pvbatt_model.value('analysis_period')} = {pvbatt_model.Outputs.annual_export_to_grid_energy[-1]:.3f} kWh")
# %%
