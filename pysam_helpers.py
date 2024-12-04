import pandas as pd 
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

# Define some helper functions to manage the model outputs
def interval_to_date_string(interval, interval_type, year):
    # Define the start of the year
    start_of_year = datetime(year=year, month=1, day=1)
    # Define the datetime based on the type of increment
    if interval_type == 'hour':
        # Add the specified number of hour intervals to the start of the year
        date_time = start_of_year + timedelta(hours=interval)
    elif interval_type == 'half-hour':
        # Add the specified number of half-hour intervals to the start of the year
        date_time = start_of_year + timedelta(minutes=interval * 30)
    elif interval_type == '5-minute':
        # Add the specified number of 5-minute intervals to the start of the year
        date_time = start_of_year + timedelta(minutes=interval * 5)
    else:
        # Return empty string if the interval type provided doesn't match any option
        return ""
    # Format the datetime to the desired string format
    return date_time.strftime('%Y-%m-%d, %H:%M:%S')

def parse_model_outputs_into_dataframes(model, five_minutes_only=False):
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
    try:
        analysis_period = model.value('analysis_period')
    except AttributeError as e:
        print(f"Warning! While trying to access the analysis_period value for this model, the\
        following error was raised:\n{e}\nUsing 25 years as the default...")
        analysis_period = 25.0
    
    # Calculate some expected DataFrame lengths
    years_in_analysis_period = int(analysis_period) + 1
    days_in_analysis_period = int(analysis_period * 365)
    hours_in_analysis_period = days_in_analysis_period * 24
    half_hours_in_analysis_period = hours_in_analysis_period * 2
    five_minutes_in_analysis_period = hours_in_analysis_period * 12
    hours_in_year = 24*365
    five_minutes_in_year = 12*24*365

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
        if five_minutes_only:
            if length == 'float':
                continue
            elif length == five_minutes_in_analysis_period:
                # Load the data into a DataFrame
                df = pd.DataFrame.from_dict(items, orient='index').T

                # Format the data
                keyname = 'Lifetime 5 Minute Data'
                df.index = df.index.map(lambda x: interval_to_date_string(
                    interval=x, interval_type='5-minute', year=2012))
                df = manage_leap_years(df)
                
                # Set the keys
                dataframes[keyname] = df
            elif length == five_minutes_in_year:
                # Load the data into a DataFrame
                df = pd.DataFrame.from_dict(items, orient='index').T
            
                # Format the data
                keyname = '5 Minute Data'
                df.index = df.index.map(lambda x: interval_to_date_string(
                    interval=x, interval_type='5-minute', year=2012))
                df = manage_leap_years(df)

                # Set the keys
                dataframes[keyname] = df
            else:
                continue
        else:
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
                elif length == hours_in_year:
                    keyname = 'Hourly Data'
                    df.index = df.index.map(lambda x: interval_to_date_string(
                        interval=x, interval_type='hour', year=2012))
                elif length == five_minutes_in_year:
                    keyname = '5 Minute Data'
                    df.index = df.index.map(lambda x: interval_to_date_string(
                        interval=x, interval_type='5-minute', year=2012))
                    df = manage_leap_years(df)
                elif length == hours_in_analysis_period:
                    keyname = 'Lifetime Hourly Data'
                    df.index = df.index.map(lambda x: interval_to_date_string(
                        interval=x, interval_type='hour', year=2012))
                    df = manage_leap_years(df)
                elif length == half_hours_in_analysis_period:
                    keyname = 'Lifetime 30 Minute Data'
                    df.index = df.index.map(lambda x: interval_to_date_string(
                        interval=x, interval_type='half-hour', year=2012))
                    df = manage_leap_years(df)
                elif length == five_minutes_in_analysis_period:
                    keyname = 'Lifetime 5 Minute Data'
                    df.index = df.index.map(lambda x: interval_to_date_string(
                        interval=x, interval_type='5-minute', year=2012))
                    df = manage_leap_years(df)                
                else:
                    keyname = f'df_{length}'
                
                # Set the keys
                dataframes[keyname] = df
        
    # Return the dictionary of DataFrames
    return dataframes

def plot_values_by_time_range(df, start_time, end_time, y_columns):
    # Use the datetime index as a column
    if 'Datetime' not in df.columns:
        this_df = df.reset_index(drop=False)
        this_df.rename(columns={'index':'Datetime'},  inplace=True)
        this_df['Datetime'] = pd.to_datetime(this_df['Datetime'])
    else:
        this_df = df.copy()
    
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

def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def manage_leap_years(df):
    # Reset the index and convert the index to a datetime column
    df = df.reset_index().rename(columns={'index': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d, %H:%M:%S')

    # Identify leap years in the range
    leap_years = [year for year in range(df['Datetime'].dt.year.min(), df['Datetime'].dt.year.max() + 1) if is_leap_year(year)]

    # For each leap year, adjust the Feb 29 entries and shift remaining dates
    for leap_year in leap_years:
        feb_29_start = pd.Timestamp(f'{leap_year}-02-29 00:00:00')
        
        # Move entries on or after Feb 29 back by one day
        feb_29_mask = df['Datetime'] >= feb_29_start
        df.loc[feb_29_mask, 'Datetime'] += pd.Timedelta(days=1)

    # Sort the DataFrame by datetime again, just in case the shifts caused disorder
    df = df.sort_values(by='Datetime').reset_index(drop=True)

    # Convert 'Datetime' back to the original string format and set it as the index
    df['Datetime'] = df['Datetime'].dt.strftime('%Y-%m-%d, %H:%M:%S')
    df = df.set_index('Datetime')

    # Return the adjusted DataFrame
    return df

def store_system_info(case_name, system_info):
    # create/locate directory for test case
    os.makedirs(os.path.join('data', 'test_cases', case_name), exist_ok = True)
    # store output data
    si_df = pd.DataFrame(system_info, index = [0])
    si_df.to_csv(os.path.join('data', 'test_cases', case_name, f'{case_name}_system_info.csv'))


def merge_subsystem_5min_dfs(system_output_dict):
    """
    This function merges select columns from the disparate dataframes from each subsystem model 
    output into a single dataframe, which can be used for system-level analysis.

    Parameters:
        - system_output_dict: A dictionary of dictionaries. The keys describe the subsystem (e.g., 
        PV, Wind, Battery). The values are the dictionary of model outputs for that subsystem. That
        dictionary would be generated by the `parse_model_outputs_into_dataframes` function for a
        given subsystem. They keys of this dictionary describe the type of data, and the values are
        DataFrames. We're only trying to access 5 minute data with this function.
        
    Returns:
        - system_df: A DataFrame
    """
    # Define a top-level dictionary to store the slices that we'll take from each DataFrame
    subsystem_dfs = []
    
    # Look through the system output dictionary for all the different subsystems
    for subsystem_type, subsystem_df_dict in system_output_dict.items():
        # Start by accessing the DataFrame with 5-minute data 
        # We expect PV and Battery to have 5 minute data throughout their lifetime
        try:
            # Reset the index so that the datetime is available as a column
            this_df = subsystem_df_dict['Lifetime 5 Minute Data'].reset_index()
        except KeyError as e:
            # We expect Wind to have 5 minute data for the first year only
            print(f"Warning: {subsystem_type} does not have lifetime 5 minute data. \
                Trying Year 1 5 minute data instead.")
            try:
                # Reset the index so that the datetime is available as a column
                this_df = subsystem_df_dict['5 Minute Data'].reset_index()
            except KeyError as e:
                # If neither type is available, something is wrong
                print(f"The {subsystem_type} dictionary does not have 5 minute data.")
                return None
        
        # Each subsystem type will have different column names
        if 'pv' in subsystem_type.lower():
            # Define the columns in the subsystem DataFrame that we want to access
            columns_to_pull = ['Datetime', 'ac_gross']
            new_column_names = {'ac_gross': 'Net PV Generation (kW)'}
        elif 'wind' in subsystem_type.lower():
            # Define the columns in the subsystem DataFrame that we want to access
            columns_to_pull = ['Datetime', 'gen']
            new_column_names = {'gen': 'Net Wind Generation (kW)'}
        elif 'battery' in subsystem_type.lower():
            # Define the columns in the subsystem DataFrame that we want to access
            columns_to_pull = ['Datetime', 'batt_to_grid', 'system_to_batt', 'system_to_grid', 'batt_SOC']
            new_column_names = {'batt_to_grid': 'Battery Discharge Power (kW)',
                                'system_to_batt': 'Battery Charge Power (kW)',
                                'system_to_grid': 'PV to Grid (kW)',
                                'batt_SOC': 'Battery SOC'}
        else:
            columns_to_pull = []

        # Pull out the columns that we had identified
        this_df = this_df.loc[:,columns_to_pull]
        # Rename those columns so that they are mapped to the subsystem type
        this_df.rename(columns=new_column_names, inplace=True)

        # Add the dataframe to the list
        subsystem_dfs.append(this_df)
    
    # Loop through the list, merging each DataFrame
    system_df = pd.merge(subsystem_dfs[0], subsystem_dfs[1], on='Datetime', how='inner')
    for idx in range(1, len(subsystem_dfs) - 1):
        system_df = pd.merge(system_df, subsystem_dfs[idx+1], on='Datetime', how='inner')

    # Return the system-wide dataframe
    return system_df
