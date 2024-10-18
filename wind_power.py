# %% imports
import windpowerlib as wp
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
# %% onshore wind data
onshore_weather_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'wind_speeds', 'sd_2012_5m.csv'), header=1)
cols = onshore_weather_df.columns
onshore_weather_df = onshore_weather_df[['wind speed at 100m (m/s)', 
                         'air pressure at 100m (Pa)',
                         'air temperature at 100m (C)']]
onshore_weather_df.columns = ['wind_speed', 'temperature', 'pressure']
onshore_roughness = 0.055
onshore_weather_df.insert(3, 'roughness_length', [onshore_roughness]*onshore_weather_df.shape[0], True)
onshore_ws = onshore_weather_df['wind_speed']
# %% offshore wind data
weather_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'wind_speeds', 'sd_2012_5m_osw.csv'), header=1)
cols = weather_df.columns
weather_df = weather_df[['wind speed at 100m (m/s)', 
                         'air pressure at 100m (Pa)',
                         'air temperature at 100m (C)']]
weather_df.columns = ['wind_speed', 'temperature', 'pressure']
offshore_roughness = 0.0002
weather_df.insert(3, 'roughness_length', [offshore_roughness]*weather_df.shape[0], True)
ws = weather_df['wind_speed']
# %% plot wind speeds
lb = 0
ub = -1
plt.plot(range(len(ws[lb:ub])), ws[lb:ub], color='blue', label = 'offshore')
plt.plot(range(len(onshore_ws[lb:ub])), onshore_ws[lb:ub], color='red', label = 'onshore')
# plt.plot(range(len(onshore_ws[lb:ub])), ws[lb:ub] - onshore_ws[lb:ub], color='purple', label = 'offshore - onshore')
plt.legend()
plt.title('Wind Speed Over Time')
plt.ylabel('Wind Speed (m/s)')
plt.xlabel('Time (5min)')
plt.show()
# %% plot turbine power curve
hub_height = 100
turbine = wp.WindTurbine(hub_height, turbine_type='E-101/3050')
pc = turbine.power_curve['value'].values.tolist()
plt.plot(turbine.power_curve['wind_speed'], turbine.power_curve['value'])
plt.title('Single Turbine Power Curve')
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Power Output (W)')
plt.show()
# %% calculate power output

# add columns for multi-level
column_tuples = [('wind_speed', hub_height), 
                 ('temperature', hub_height), 
                 ('pressure', hub_height),
                 ('roughness_length', hub_height)]
multi_index = pd.MultiIndex.from_tuples(column_tuples, names=['variable', 'height'])
multi_index_1 = pd.MultiIndex.from_tuples(column_tuples, names=['variable', 'height'])
weather_df.columns = multi_index
onshore_weather_df.columns = multi_index_1
num_turbines = 5
# build offshore fleet
turbine = wp.WindTurbine(hub_height, turbine_type='E-101/3050')
turbine_group = turbine.to_group(number_turbines = num_turbines)
wind_farm = wp.WindFarm([turbine_group])
model_chain = wp.TurbineClusterModelChain(wind_farm)
output_osw = model_chain.run_model(weather_df).power_output.values.tolist()
# build onshore fleet
turbine = wp.WindTurbine(hub_height, turbine_type='E-101/3050')
turbine_group = turbine.to_group(number_turbines = num_turbines)
wind_farm = wp.WindFarm([turbine_group])
model_chain = wp.TurbineClusterModelChain(wind_farm)
output_onshore = model_chain.run_model(onshore_weather_df).power_output.values.tolist()
# %% plot power output over the year
plt.plot(range(len(output_osw)), output_osw, color='blue', label = 'Offshore Wind')
plt.plot(range(len(output_onshore)), output_onshore, color='red', label = 'Onshore Wind')
plt.xlabel('Time (5 min)')
plt.ylabel('Power Output (W)')
plt.title('2012 Wind Farm Power Output ')
plt.legend()
plt.show()
# %%
