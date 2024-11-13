# %%
# Import each module in the order that it will be executed
import PySAM.GenericSystem as GenericSystem
import PySAM.Battery as Battery
import PySAM.Grid as Grid
import PySAM.Utilityrate5 as UtilityRate
import PySAM.Singleowner as SingleOwner

# Also import JSON to load the inputs
import json

# %% Create a new instance of each module
generic_system = GenericSystem.new()
batt_model = Battery.from_existing(generic_system)
grid = Grid.from_existing(generic_system)
utility_rate = UtilityRate.from_existing(generic_system)
single_owner = SingleOwner.from_existing(generic_system)

# %% Load the inputs from the JSON file for each module
dir = 'data/PySAM_Inputs/Generic_Battery_System_Demo/'
prefix = 'Generic_Battery_System_Demo_'
file_names = ["generic_system", "battery", "grid", "utilityrate5", "singleowner"]
modules = [generic_system, batt_model, grid, utility_rate, single_owner]
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
print(f"{batt_model.value('batt_computed_bank_capacity'):,.0f} kWh battery cycled {batt_model.Outputs.batt_cycles[-1]} times.\n")