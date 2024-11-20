#%% Imports
import pickle
import pandas as pd
import pysam_helpers

#%% Load pickled objects
pv_model_outputs = {'Lifetime 5 Minute Data':None}
wind_model_outputs = {'5 Minute Data':None}
battery_model_outputs = {'Lifetime 5 Minute Data':None}
grid_model_outputs = {'Lifetime 5 Minute Data':None}
single_owner_model_outputs = {'Lifetime 5 Minute Data':None}

im_a_pickle_dict = {
        'pv_model_outputs' : pv_model_outputs['Lifetime 5 Minute Data'],
        'wind_model_outputs' : wind_model_outputs['5 Minute Data'],
        'battery_model_outputs': battery_model_outputs['Lifetime 5 Minute Data'],
        'grid_model_outputs' : grid_model_outputs['Lifetime 5 Minute Data'],
        'single_owner_model_outputs': single_owner_model_outputs['Lifetime 5 Minute Data']
}

# Load PV data
pv_filepath = ''        # Replace this with your own filepath to the data in google drive
with open(pv_filepath, 'rb') as f:
    pv_model_outputs['Lifetime 5 Minute Data'] = pickle.load(f)

# Load wind data
wind_filepath = ''        # Replace this with your own filepath to the data in google drive
with open(wind_filepath, 'rb') as f:
    wind_model_outputs['5 Minute Data'] = pickle.load(f)

# Load battery data
battery_filepath = ''        # Replace this with your own filepath to the data in google drive
with open(battery_filepath, 'rb') as f:
    battery_model_outputs['Lifetime 5 Minute Data'] = pickle.load(f)

grid_filepath = ''        # Replace this with your own filepath to the data in google drive
with open(grid_filepath, 'rb') as f:
    grid_model_outputs['Lifetime 5 Minute Data'] = pickle.load(f)

single_owner_filepath = ''        # Replace this with your own filepath to the data in google drive
with open(single_owner_filepath, 'rb') as f:
    single_owner_model_outputs['Lifetime 5 Minute Data'] = pickle.load(f)

# %% Generate some plots
date_start = '2012-07-27 00:00:00'
date_end = '2012-07-28 00:00:00'

pysam_helpers.plot_values_by_time_range(df=pv_model_outputs['Lifetime 5 Minute Data'], start_time=date_start, end_time=date_end, y_columns=['gen'])
pysam_helpers.plot_values_by_time_range(df=wind_model_outputs['5 Minute Data'], start_time=date_start, end_time=date_end, y_columns=['gen'])
pysam_helpers.plot_values_by_time_range(df=battery_model_outputs['Lifetime 5 Minute Data'], start_time=date_start, end_time=date_end, y_columns=['batt_SOC'])
pysam_helpers.plot_values_by_time_range(df=battery_model_outputs['Lifetime 5 Minute Data'], start_time=date_start, end_time=date_end, y_columns=['batt_to_grid', 'system_to_batt', 'system_to_grid'])

# %%
