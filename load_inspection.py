# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.interpolate import CubicSpline
import pvlib

# %% Define a few helper functions
# Function to parse datetime with different formats
def parse_leapday_datetimes(dt):
    # Attempt to parse with the second format (24-hour format)
    try:
        return datetime.strftime(dt, "%Y-%m-%d %H:%M:%S")
    except TypeError:
        pass  # If it fails, return None

    return dt  # Return the original string if all parsing attempts fail

# Function to parse datetime with different formats
def parse_datestring(dt_str):
    # Attempt to parse with the first format (12-hour format)
    try:
        return datetime.strptime(dt_str, '%m/%d/%Y %I:%M:%S %p')
    except ValueError:
        pass  # If it fails, try the next format

    # Attempt to parse with the second format (24-hour format)
    try:
        return datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        pass  # If it fails, return None

    return None  # Return None if all parsing attempts fail

# %% Prepare data
# Load the load data into a DataFrame
load = pd.read_excel('data/Project 2 - Load Profile.xlsx')
# Convert the leapday datetime entries to a string
load['Datetime'] = load['Datetime'].apply(parse_leapday_datetimes)
# Change to 2012 so that the datetime library properly handles the leap day
load['Datetime'] = load['Datetime'].str.replace('2022', '2012')
# Fix the extra day entries – they should just be Jan 1 of the following year
load['Datetime'] = load['Datetime'].str.replace('2021', '2013')
# Convert all strings to datetime 
load['Datetime'] = load['Datetime'].apply(parse_datestring)

# Load the solar resource data into a DataFrame
nsrdb = pd.read_csv('data/222628_32.73_-117.18_2012.csv', skiprows=2, usecols=['Year', 'Month', 'Day', 'Hour', 'Minute', 'Temperature', 'DHI', 'GHI',
       'DNI', 'Surface Albedo', 'Wind Speed', 'Pressure'])
local_index = pd.DatetimeIndex(nsrdb['Year'].astype(str) + '-' + nsrdb['Month'].astype(str) +
                             '-' + nsrdb['Day'].astype(str) + ' ' + nsrdb['Hour'].astype(str) +
                               ':' + nsrdb['Minute'].astype(str))
nsrdb.set_index(local_index, inplace=True)

# Interpolate NSRDB data
start_date = str(nsrdb.index[0])
end_date = str(nsrdb.index[-1])

times_interp = pd.date_range(start_date, end_date, freq='5min')

nsrdb_interpolated = pd.DataFrame(index=times_interp)
nsrdb_interpolated['dni_extra'] = pvlib.irradiance.get_extra_radiation(nsrdb_interpolated.index)
times_float = times_interp.to_numpy().astype(float)
for i in nsrdb.columns:
    cs = CubicSpline(nsrdb.index.to_numpy().astype(float), nsrdb[i].values)
    nsrdb_interpolated[i] = cs(times_float)
    nsrdb_interpolated[i] = np.maximum(nsrdb_interpolated[i], 0)

# Use the datetime as a column
nsrdb_interpolated.reset_index(drop=False, inplace=True)
nsrdb_interpolated.rename(columns={'index':'Datetime'}, inplace=True)

# Load the wind resource data into a DataFrame
onshore_weather_df = pd.read_csv('data/wind_speeds/sd_2012_5m.csv', header=1)
offshore_weather_df = pd.read_csv('data/wind_speeds/sd_2012_5m_osw.csv', header=1)

# Create datetime columns for the wind DataFrames
onshore_weather_df['Datetime'] = pd.to_datetime(onshore_weather_df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
offshore_weather_df['Datetime'] = pd.to_datetime(offshore_weather_df[['Year', 'Month', 'Day', 'Hour', 'Minute']])

# Rename columns in the offshore weather DataFrame so as to preserve them during the merge
offshore_weather_df.rename(columns=dict(zip(offshore_weather_df.columns.to_list(), [f'Offshore - {col_name}' for col_name in offshore_weather_df.columns.to_list()])), inplace=True)
offshore_weather_df.rename(columns={'Offshore - Datetime':'Datetime'}, inplace=True)
# %% 
# Find the load during which generation is lacking
merged = pd.merge(load, nsrdb_interpolated, on='Datetime', how='inner')
merged = pd.merge(merged, onshore_weather_df, on='Datetime', how='inner')
merged = pd.merge(merged, offshore_weather_df, on='Datetime', how='inner')
result = merged[
    (merged['wind speed at 100m (m/s)'] <= 4)
    & (merged['Offshore - wind speed at 100m (m/s)'] <= 4)
    & (merged['GHI'] <= 100)
    ]
# %%
# Build load duration curves to show what must be supplied by batteries / alternative generation
sorted_load = load['Load (MW)'].sort_values(ascending=False)
sorted_load.reset_index(drop=True, inplace=True)
peak_load = sorted_load[0]
base_load = sorted_load[len(sorted_load)-1]
print(f"For the complete dataset:\n \
    Peak system load is {peak_load:.2f} MW\n \
    System baseload is {base_load:.2f} MW")


sorted_load_no_generation = result['Load (MW)'].sort_values(ascending=False)
sorted_load_no_generation.reset_index(drop=True, inplace=True)
peak_load_no_generation = sorted_load_no_generation[0]
base_load_no_generation = sorted_load_no_generation[len(sorted_load_no_generation)-1]
print(f"When neither PV nor wind resources are available:\n \
    Peak system load is {peak_load_no_generation:.2f} MW\n \
    System baseload is {base_load_no_generation:.2f} MW")

plt.figure(figsize=(10,7))
plt.plot(sorted_load.index, sorted_load, label='Complete dataset')
plt.plot(sorted_load_no_generation.index, sorted_load_no_generation, label='Excl. PV & wind')
plt.xlabel('5-Minute Interval')
plt.ylabel('Load (MW)')
plt.title('5-Minute Load Duration Curve')
plt.legend()
plt.grid(True)
plt.show()

# %% 
# Print some load statistics
max_load = load['Load (MW)'].max()
max_load_datetime = load['Datetime'][load['Load (MW)'] == max_load].item()
print(f"The maximum load on the system is: {max_load:.2f} MW, which occurs at {max_load_datetime}.")

# %%
# To-do: find load that persists for 4 hours or more