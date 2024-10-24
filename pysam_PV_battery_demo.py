# %%
import json
# PySAM requires that all modules be imported
import PySAM.Pvsamv1 as PVSAM
import PySAM.Battery as Battery
import PySAM.Grid as Grid
import PySAM.Utilityrate5 as UtilityRate
import PySAM.Singleowner as SingleOwner

# %% create a new instance of the Battery module
pv = PVSAM.new()
batt_model = Battery.from_existing(pv)
grid = Grid.from_existing(pv)
utility_rate = UtilityRate.from_existing(pv)
single_owner = SingleOwner.from_existing(pv)

# %% get the inputs from the JSON file
dir = 'data/PySAM_Inputs/'
prefix = 'PV_Battery_System_Demo_'
file_names = ["pvsamv1", "battery", "grid", "utilityrate5", "singleowner"]
modules = [pv, batt_model, grid, utility_rate, single_owner]
for f, m in zip(file_names, modules):
    filepath = dir + prefix + f + '.json'
    print(f"Loading inputs from {filepath}")
    with open(filepath, 'r') as file:
        data = json.load(file)
        # loop through each key-value pair and set the module inputs
        for k, v in data.items():
            if k != 'number_inputs':
                m.value(k, v)

# %% run the module
batt_model.execute()

# %% print results
print(f'{batt_model.value('batt_bank_size_capacity'):,.2f} kW battery cycled {batt_model.Outputs.batt_cycles} times.\n')

# %% run PVWatts for a series of nameplate capacities
capacities = [10, 100, 1000]
for c in capacities:
    # change the value of the system_capacity input
    batt_model.value('system_capacity',c)
    # run the module
    batt_model.execute()
    # print some results
    print('Annual AC output for {capacity:,.2f} kW system is {output:,.0f} kWh.'.format(capacity = batt_model.value('system_capacity'), output = batt_model.Outputs.ac_annual) )