#%%
import pandas as pd
import numpy as np

# Load the dataframe
df = pd.read_csv('pvbatt_30min.csv')

# Modify the columns – note that we only have to do this when loading from the csv in this script.
# When running this as part of the normal data processing flow, we will not need to.
df.rename(columns={'Unnamed: 0': 'Datetime'}, inplace=True)
df['Datetime'] = pd.to_datetime(df['Datetime'])

#%% Print original leap day data
start_date = '2012-02-28 00:00:00'
end_date = '2012-03-02 00:00:00'

# Filter the DataFrame for the specified range
subset_df = df[(df['Datetime'] >= start_date) & (df['Datetime'] <= end_date)]

# Select every 6 hours (every 6*60/5 = 72 rows, since entries are at 5-minute intervals)
subset_df_6h = subset_df.iloc[::12]

# Print the result
subset_df_6h

# %% Step 1: Identify leap years in the range
def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

leap_years = [year for year in range(df['Datetime'].dt.year.min(), df['Datetime'].dt.year.max() + 1) if is_leap_year(year)]
leap_years

# %% Step 2: Insert leap days
for leap_year in leap_years:
    feb_29_start = pd.Timestamp(f'{leap_year}-02-29 00:00:00')

    # Move entries on or after Feb 29 back by one day
    feb_29_mask = df['Datetime'] >= feb_29_start
    df.loc[feb_29_mask, 'Datetime'] += pd.Timedelta(days=1)


# Step 3: Sort the DataFrame by datetime again, just in case the shifts caused disorder
df = df.sort_values(by='Datetime').reset_index(drop=True)

#%% Print new leap day data
start_date = '2012-02-28 00:00:00'
end_date = '2012-03-02 00:00:00'

# Filter the DataFrame for the specified range
subset_df = df[(df['Datetime'] >= start_date) & (df['Datetime'] <= end_date)]

# Select every 6 hours (every 6*60/5 = 72 rows, since entries are at 5-minute intervals)
subset_df_6h = subset_df.iloc[::12]

# Print the result
subset_df_6h
