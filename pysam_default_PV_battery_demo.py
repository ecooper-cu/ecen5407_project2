# %%
# Import each module in the order that it will be executed
import PySAM.Pvsamv1 as PVSAM
import PySAM.Grid as Grid
import PySAM.Utilityrate5 as UtilityRate
import PySAM.Singleowner as SingleOwner

# Also import JSON to load the inputs
import json

# %% Create a new instance of each module
# Use the default PV + battery system with a single owner financial model
pvbatt_model = PVSAM.default('PVBatterySingleOwner')
grid = Grid.from_existing(pvbatt_model, 'PVBatterySingleOwner')
utility_rate = UtilityRate.from_existing(pvbatt_model, 'PVBatterySingleOwner')
single_owner = SingleOwner.from_existing(pvbatt_model, 'PVBatterySingleOwner')

# Define the module list in the order to be executed
modules = [pvbatt_model, grid, utility_rate, single_owner]

# %% Load in the weather data for San Diego
filename = 'data/222628_32.73_-117.18_2012.csv'
pvbatt_model.SolarResource.solar_resource_file = filename

# %% Run the modules in order
for m in modules:
    m.execute()

# %% Print some example results to show that execution was successful
print(f"{pvbatt_model.value('batt_computed_bank_capacity'):,.0f} kWh battery cycled {pvbatt_model.Outputs.batt_cycles[-1]} times.\n")
print(f"Annual system AC output in year {pvbatt_model.value('analysis_period')} = {pvbatt_model.Outputs.annual_export_to_grid_energy[-1]:.3f} kWh")
# %%
