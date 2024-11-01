import pandas as pd 
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Define some helper functions to manage the model outputs
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
