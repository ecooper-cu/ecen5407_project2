import pandas as pd
import numpy as np
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

def find_threshold_counts(load_threshold, duration, df):
    """
    Find the number of instances where the load exceeds a certain threshold for a given duration or
    longer.

    Parameters:
        - load_threshold: The load threshold in MW. Function will search for datapoints that exceed
        this value.
        - duration: The duration in minutes. Function will search for datapoints that last for at 
        least as long as this value.
        - df: The DataFrame to search through. Must have a column for 'Load (MW)' and 'Datetime'

    Returns:
        - A count for the number of instances in the input DataFrame that meet the search 
        conditions.
    """
    load_exceed_df = df[df['Load (MW)'] > load_threshold]
        
    # Add a helper column to identify groups of consecutive intervals
    load_exceed_df.loc[:,'consecutive_id'] = (load_exceed_df['Datetime'].diff() > pd.Timedelta(minutes=5)).cumsum()

    # Group by the consecutive_id and filter groups where the duration is at least the specified duration
    result_df = load_exceed_df.groupby('consecutive_id').filter(
        lambda x: (x['Datetime'].max() - x['Datetime'].min()).total_seconds() / 60 >= duration
    )

    # Return the number of instances that exceed the load threshold for the duration or longer
    if len(result_df['consecutive_id'].to_list()) > 0:
        return result_df['consecutive_id'].to_list()[-1]
    else:
        return 0

def interp_nsrdb(nsrdb_filepath):
    # Load the solar resource data into a DataFrame
    nsrdb = pd.read_csv(nsrdb_filepath, skiprows=2, usecols=['Year', 'Month', 'Day', 'Hour', 'Minute', 'Temperature', 'DHI', 'GHI',
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

    return nsrdb_interpolated

def format_load_data(load_filepath):
    load = pd.read_excel(load_filepath)
    # Convert the leapday datetime entries to a string
    load['Datetime'] = load['Datetime'].apply(parse_leapday_datetimes)
    # Change to 2012 so that the datetime library properly handles the leap day
    load['Datetime'] = load['Datetime'].str.replace('2022', '2012')
    # Fix the extra day entries – they should just be Jan 1 of the following year
    load['Datetime'] = load['Datetime'].str.replace('2021', '2013')
    # Convert all strings to datetime 
    load['Datetime'] = load['Datetime'].apply(parse_datestring)

    return load

def prepare_wind_data(wind_filepath, is_offshore=False):
    # Load the wind resource data into a DataFrame
    df = pd.read_csv(wind_filepath, header=1)

    # Create a datetime column
    df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])

    # Rename columns in the offshore weather DataFrame so as to preserve them during the merge
    if is_offshore:
        df.rename(columns=dict(zip(df.columns.to_list(), [f'Offshore - {col_name}' for col_name in df.columns.to_list()])), inplace=True)
        df.rename(columns={'Offshore - Datetime':'Datetime'}, inplace=True)

    return df
