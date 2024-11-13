# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

import load_inspection_helpers
# %% Prepare data
# Load the load data into a DataFrame
load = load_inspection_helpers.format_load_data(load_filepath='data/Project 2 - Load Profile.xlsx')

nsrdb_interpolated = load_inspection_helpers.interp_nsrdb(nsrdb_filepath='data/222628_32.73_-117.18_2012.csv')

onshore_weather_df = load_inspection_helpers.prepare_wind_data(wind_filepath='data/wind_speeds/sd_2012_5m.csv')
offshore_weather_df = load_inspection_helpers.prepare_wind_data(wind_filepath='data/wind_speeds/sd_2012_5m_osw.csv', is_offshore=True)

# %% Find the load during which generation is lacking
merged = pd.merge(load, nsrdb_interpolated, on='Datetime', how='inner')
merged = pd.merge(merged, onshore_weather_df, on='Datetime', how='inner')
merged = pd.merge(merged, offshore_weather_df, on='Datetime', how='inner')
result = merged[
    (merged['wind speed at 100m (m/s)'] <= 4)
    & (merged['Offshore - wind speed at 100m (m/s)'] <= 4)
    & (merged['GHI'] <= 100)
    ]
# %% Build load duration curves to show what must be supplied by batteries / alternative generation
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

# %% Print some load statistics
max_load = load['Load (MW)'].max()
max_load_datetime = load['Datetime'][load['Load (MW)'] == max_load].item()
print(f"The maximum load on the system is: {max_load:.2f} MW, which occurs at {max_load_datetime}.")

# %% Find instances where the load exceeds a given magnitude for a given duration
load_boundaries = [30, 40, 45, 50, 55, 60, 70, 80, 90, 100]
duration_boundaries = [5, 10, 15, 30, 60, 120, 180, 240, 300, 360]
counts = []

for l_bound in load_boundaries:
    this_list = []
    for d_bound in duration_boundaries:
        this_count = load_inspection_helpers.find_threshold_counts(load_threshold=l_bound, duration=d_bound, df=result)
        this_list.append(this_count)
    counts.append(this_list)

counts = np.array(counts)

# %% Plot the load threshold countsas a heatmap
fig, ax = plt.subplots()
im = ax.imshow(counts, cmap=colormaps['viridis'], origin='lower')

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(duration_boundaries)), labels=duration_boundaries)
ax.set_xlabel('Load Duration [Min]')
ax.set_yticks(np.arange(len(load_boundaries)), labels=load_boundaries)
ax.set_ylabel('Load Magnitude [MW]')

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(load_boundaries)):
    for j in range(len(duration_boundaries)):
        if i < 3:
            color = 'k'
        else:
            color = 'w'
        text = ax.text(j, i, counts[i, j],
                       ha="center", va="center", color=color)

ax.set_title("Instances of Excessive Load")
fig.tight_layout()
plt.show()
# %% Plot load data for a specific day
date = '2012-12-24'
filtered_df = load[load['Datetime'].dt.date == pd.to_datetime(date).date()]
filtered_df.plot(x='Datetime', y='Load (MW)', title=date, xlabel='Datetime', ylabel='Load (MW)', grid=True)

# %%
