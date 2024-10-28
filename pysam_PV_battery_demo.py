# %%
# Import each module in the order that it will be executed
import PySAM.Pvsamv1 as PVSAM
import PySAM.Grid as Grid
import PySAM.Utilityrate5 as UtilityRate
import PySAM.Singleowner as SingleOwner

import json # To load inputs from SAM

# To organize and plot outputs from simulation
import pandas as pd 
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# %% Define some helper functions to manage the model outputs
def hour_to_date_string(hour):
    # Define the start of the year
    start_of_year = datetime(year=2012, month=1, day=1)
    # Add the specified number of hours to the start of the year
    date_time = start_of_year + timedelta(hours=hour)
    # Format the datetime to the desired string format
    return date_time.strftime('%Y-%m-%d, %H:%M:%S')

def thirty_min_to_date_string(thirty_min):
    # Define the start of the year
    start_of_year = datetime(year=2012, month=1, day=1)
    # Add the specified number of hours to the start of the year
    date_time = start_of_year + timedelta(minutes=thirty_min * 30)
    # Format the datetime to the desired string format
    return date_time.strftime('%Y-%m-%d, %H:%M:%S')

def parse_model_outputs_into_dataframes(model):
    """
    After executing each model, there will be a set of outputs associated with the simulation. 
    
    Outputs will come in different lengths, depending on the type of data. For a full breakdown of 
    the different types of outputs, try running 'help(pvbatt_model.Outputs)', or something similar
    if you want to look at outputs from another model. The outputs may be available over the 
    following timescales:
        - Single Values (e.g., Net present value)
        - 30-minute Data (e.g., Battery power [kW] - if PV + battery)
        - Hourly Data (e.g., Battery power [kW] - if standalone battery)
        - Monthly Data (e.g., Energy to grid from battery [kWh])
        - Annual Data (e.g., Annual energy exported to grid [kWh])
    
    The outputs are provided as tuples or floats by default, but we would prefer to work with 
    DataFrames to make life easier.

    This function creates a DataFrame for each type of output, (e.g., one for Hourly Data, another 
    for Annual Data). The dataframes are returned in a dictionary.
    """
    print(f"Exporting outputs from model: {model}")
    
    # Grab the outputs
    outputs_dict = model.Outputs.export()

    # Find the analysis period of the model
    analysis_period = model.value('analysis_period')
    
    # Calculate some expected DataFrame lengths
    years_in_analysis_period = int(analysis_period) + 1
    days_in_analysis_period = int(analysis_period * 365)
    hours_in_analysis_period = days_in_analysis_period * 24
    half_hours_in_analysis_period = hours_in_analysis_period * 2

    # Initialize a dictionary to store data for each unique length
    grouped_data = {}

    # Separate values based on length or type
    for k, v in outputs_dict.items():
        if isinstance(v, tuple):  # Check if value is a tuple
            length = len(v)
            if length not in grouped_data:
                grouped_data[length] = {}  # Create a new dictionary for this length
            grouped_data[length][k] = v
        elif isinstance(v, float):  # Separate case for floats
            if 'float' not in grouped_data:
                grouped_data['float'] = {}
            grouped_data['float'][k] = v

    # Create DataFrames for each group
    dataframes = {}
    for length, items in grouped_data.items():
        if length == 'float':
            # Load the data into a DataFrame
            df = pd.DataFrame([items.values()], columns=items.keys())

            # These are single-value data
            dataframes[f'Single Values'] = df
        else:
            # Load the data into a DataFrame
            df = pd.DataFrame.from_dict(items, orient='index').T

            # Find the data interval length based on the length of the DataFrame
            if length == years_in_analysis_period:
                keyname = 'Annual_Data'
            elif length == 12:
                keyname = 'Monthly Data'
                df.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
            elif length == 8760:
                keyname = 'Hourly Data'
                df.index = df.index.map(hour_to_date_string)
            elif length == hours_in_analysis_period:
                keyname = 'Lifetime Hourly Data'
                df.index = df.index.map(hour_to_date_string)
            elif length == half_hours_in_analysis_period:
                keyname = 'Lifetime 30 Minute Data'
                df.index = df.index.map(thirty_min_to_date_string)
            else:
                keyname = f'df_{length}'
            
            # Set the keys
            dataframes[keyname] = df
        
    # Return the dictionary of DataFrames
    return dataframes

def plot_values_by_time_range(df, start_time, end_time, y_columns):
    # Use the datetime index as a column
    this_df = df.reset_index(drop=False)
    this_df.rename(columns={'index':'Datetime'},  inplace=True)
    this_df['Datetime'] = pd.to_datetime(this_df['Datetime'])
    
    # Filter the DataFrame based on the time range
    mask = (this_df['Datetime'] >= start_time) & (this_df['Datetime'] <= end_time)
    df_filtered = this_df.loc[mask]
    
    # Plot each specified column
    plt.figure(figsize=(10, 6))
    
    for column in y_columns:
        if column in df_filtered.columns:
            plt.plot(df_filtered['Datetime'], df_filtered[column], marker='o', linestyle='-', label=column)
        else:
            print(f"Warning: '{column}' does not exist in the dataframe.")
    
    plt.xlabel('Datetime')
    plt.ylabel('Values')
    plt.title(f'Values vs Time from {start_time} to {end_time}')
    plt.legend()  # Add a legend to differentiate the lines
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.show()

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
pvbatt_model_outputs = parse_model_outputs_into_dataframes(pvbatt_model)
grid_model_outputs = parse_model_outputs_into_dataframes(grid)
utility_rate_outputs = parse_model_outputs_into_dataframes(utility_rate)
single_owner_outputs = parse_model_outputs_into_dataframes(single_owner)

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
plot_values_by_time_range(df=pvbatt_model_outputs['Lifetime 30 Minute Data'], start_time='2012-07-27 00:00:00', end_time='2012-07-28 00:00:00', y_columns=['batt_SOC'])
plot_values_by_time_range(df=pvbatt_model_outputs['Lifetime 30 Minute Data'], start_time='2012-07-27 00:00:00', end_time='2012-07-28 00:00:00', y_columns=['batt_power', 'system_to_grid', 'grid_to_batt'])
# %%
