import pandas as pd
import json
import matplotlib.pyplot as plt
import os
import numpy as np
import load_inspection_helpers
import calendar
from geothermal_constants import *
import pysam_helpers
from datetime import datetime, timedelta

def calculate_baseline_metrics(test_case, test_case_system_info):
    """
    Determine baseline metrics for the given test case.
    Metrics include:
     - whether the net generation is greater than the net load (feasibility) (bool)
     - total curtailed wind and solar (measure of excess generation) (float)
     - average battery capacity factor (measure of adequate battery sizing) (array size 12)

     Params:
     test_case (pd.DataFrame): contains time series of power generated for wind, solar, battery, geothermal, and load
     test_case_system_info (pd.DataFrame): contains sizing and cost information for system components
     Return:
     Dictionary with stored metrics
    """
    # check if the system is feasible throughout the year
    test_case, is_feas = determine_feasibility(test_case)
    unmet_load_metrics = {}
    #excess_gen_metrics = calculate_excess_generation(test_case)
    unmet_load_metrics = calculate_reliability_margin(test_case)
    # calculate curtailed wind and solar
    curtailment_report = calculate_curtailment_metrics(test_case, excess_threshold_percent=1e-2)
    # calculate average battery SOC
    avg_cap = determine_battery_cap(test_case, test_case_system_info)

    report = {'feasibility': {"Total generation > total load (kWh)":is_feas,
                              "Temporal Alignment of Load and Generation": unmet_load_metrics},
             'curtailment': curtailment_report,
             'Battery Capacity Factors': avg_cap}
    return report

def determine_feasibility(test_case):
    # sum over all generation
    total_gen = float(test_case['Available Generation (kW)'].sum(axis=0))
    total_load = float(sum(test_case['Load (kW)']))
    net_feas = total_gen >= total_load
    # store timesteps when load > gen
    #infeas_timesteps = [(test_case['Datetime'].iloc[i]).strftime("%Y-%m-%d %X")for i in range(test_case.shape[0]) if test_case['Generation to Grid (kW)'].iloc[i] < test_case['Load (kW)'].iloc[i]]
    test_case['Unmet Load'] = test_case['Load (kW)'] - test_case['Generation to Grid (kW)']
    return test_case, net_feas

def calculate_reliability_margin(test_case:pd.DataFrame, threshold = 1e-2):
    # Define a threshold
    unmet_threshold = (test_case['Load (kW)'].max()) * threshold # calculating 1% of peak load
    # Determine if there is unmet load
    unmet_load_mask = test_case['Unmet Load'] > unmet_threshold # whether unmet load is greater than 1% peak load
    # Find the number of unmet load instances
    unmet_load_instances = float(unmet_load_mask.sum()) # number of unmet loads above threshold
    
    ### Calculate the amount of time that the load was not met
    # First, find the total number of minutes in the year where the load was not met
    unmet_load_duration_mins = unmet_load_instances * 5
    # Express the magnitude as a percent of total minutes in the year
    mins_in_year = 525600
    unmet_load_percent_mins = (unmet_load_duration_mins / mins_in_year) * 100
    
    ### Calculate the amount of energy that was unmet
    # First, get a time series of unmet energy demand in 5 minute intervals
    unmet_load_ts = test_case['Unmet Load'][unmet_load_mask] * (5/60)
    # Find the total unmet energy
    unmet_load_magnitude_energy = unmet_load_ts.sum()
    
    # Express the magnitude of unmet energy as a percent
    # To do this, we first need to know the load demand in kWh for the year
    year_1_mask = test_case['Datetime'] < pd.Timestamp(f'2013-01-01 00:00:00')
    # Time series of energy demand in 5 minute intervals
    year_1_load_ts = test_case.loc[year_1_mask, 'Load (kW)'] * (5/60)
    # Total energy demand for the year
    year_1_load_kwh =  year_1_load_ts.sum()
    # As a percent
    unmet_load_percent_energy = (unmet_load_magnitude_energy / year_1_load_kwh) * 100
    #unmet_load_magnitude_energy = float(test_case['Unmet Load'][unmet_load_mask].sum()) # total amount of unmet load above threshold
    #unmet_load_percent_energy = float((unmet_load_magnitude / (test_case['Load (kW)'].sum())) * 100) # what percent of load is unmet
    reliability_metrics = {
        'Duration of Unmet Load (min)': unmet_load_duration_mins,
        'Duration of Unmet Load (%)': unmet_load_percent_mins,
        'Unmet Energy Demand (kWh)': unmet_load_magnitude_energy,
        'Unmet Energy Demand (%)': unmet_load_percent_energy
        }
    return reliability_metrics

def calculate_excess_generation(test_case:pd.DataFrame):
    # TODO: histogram of excess generation
    excess_gen_dict = {}
    # calculate total excess generation
    excess_gen = test_case['Available Generation (kW)'] - test_case['Load (kW)']
    total_excess_gen_annual = float(excess_gen.sum(axis=0))
    avg_excess_gen_annual = total_excess_gen_annual/test_case.shape[0]
    excess_gen_dict.update({'total_excess_gen_annual': total_excess_gen_annual, 'avg_excess_gen_annual':avg_excess_gen_annual})
    # calculate excess generation during evening peak
    start_time = pd.to_datetime("16:00").time()  # 4 PM
    end_time = pd.to_datetime("19:00").time()    # 7 PM
    peak_df = test_case[(test_case['Datetime'].dt.time >= start_time) & (test_case['Datetime'].dt.time <= end_time)]
    peak_excess_gen = peak_df['Available Generation (kW)'] - peak_df['Load (kW)']
    total_excess_gen_peak = float(peak_excess_gen.sum(axis=0))
    avg_excess_gen_peak = total_excess_gen_peak/peak_df.shape[0]
    excess_gen_dict.update({'total_excess_gen_peak': total_excess_gen_peak, 'avg_excess_gen_peak': avg_excess_gen_peak})
    # calculate excess generation during midday
    start_time = pd.to_datetime("11:00").time()  # 11 AM
    end_time = pd.to_datetime("15:00").time()    # 3 PM
    midday_df = test_case[(test_case['Datetime'].dt.time >= start_time) & (test_case['Datetime'].dt.time <= end_time)]
    midday_excess_gen = midday_df['Available Generation (kW)'] - midday_df['Load (kW)']
    total_excess_gen_midday = float(midday_excess_gen.sum(axis=0))
    avg_excess_gen_midday = total_excess_gen_midday/midday_df.shape[0]
    excess_gen_dict.update({'total_excess_gen_midday': total_excess_gen_midday, 'avg_excess_gen_midday': avg_excess_gen_midday})
    # calculate excess generation during night
    start_time = pd.to_datetime("21:00").time()  # 9 PM
    end_time = pd.to_datetime("5:00").time()    # 5 AM
    nighttime_df = test_case[((test_case['Datetime'].dt.time >= start_time) & 
                              (test_case['Datetime'].dt.time <= pd.to_datetime("23:55").time())) |
                             ((test_case['Datetime'].dt.time >= pd.to_datetime("23:55").time()) & 
                              (test_case['Datetime'].dt.time <= end_time)) ] 
    nighttime_excess_gen = nighttime_df['Available Generation (kW)'] - nighttime_df['Load (kW)']
    total_excess_gen_nighttime = float(nighttime_excess_gen.sum(axis=0))
    avg_excess_gen_nighttime = total_excess_gen_nighttime/nighttime_df.shape[0]
    excess_gen_dict.update({'total_excess_gen_nighttime': total_excess_gen_nighttime, 'avg_excess_gen_nighttime': avg_excess_gen_nighttime})
    return 

def add_load_to_test_case(test_case:pd.DataFrame, load_df:pd.DataFrame):
    # Merge the two dataframes
    merged_df = pd.merge(test_case, load_df, on='Datetime', how='inner')
    # Return the merged value
    return merged_df

def calculate_curtailment_metrics(test_case, excess_threshold_percent):
    # Define a threshold over which we will consider generation to be excessive
    # In this case, we choose a threshold value that is some percentage of the peak load.
    # I.e., if the threshold percentage is 0.1%, and the peak load is 100kW, then we are 
    # considering any generation over the load + 100W to be excessive.
    excess_threshold = (test_case['Load (kW)'].max()) * excess_threshold_percent
    
    # Find the difference between net generation and the load. Net excess generation is positive.
    # We're looking at 'Generation to Grid (kW)' here, which is the net uncurtailed generation from
    # wind, pv, and battery combined with the unramped geothermal generation (rectangular).
    test_case['Excess Generation (kW)'] = test_case['Generation to Grid (kW)'] - test_case['Load (kW)']
    
    # Take a slice of the net excess generation based on our threshold
    curtailment_mask = test_case['Excess Generation (kW)'] > excess_threshold
    # This is a time series of energy generation that exceeds our threshold. Each value is the 
    # excess energy generation during the 5 minute window, expressed in kWh
    curtailed_generation_ts = test_case['Excess Generation (kW)'][curtailment_mask] * (5/60) 
    
    # Find the magnitude of the curtailment in absolute
    curtailment_magnitude = curtailed_generation_ts.sum()
    
    # Find the magnitude of the curtailment in percent
    # To do this, we first need to know the uncurtailed and unramped generation
    year_1_mask = test_case['Datetime'] < pd.Timestamp(f'2013-01-01 00:00:00')
    year_1_generation_kwh = test_case.loc[year_1_mask, 'Available Energy (kWh)'].sum()
    curtailment_percent = (curtailment_magnitude / year_1_generation_kwh) * 100

    metrics = {'Curtailed energy in year 1 (kWh)': float(curtailment_magnitude), 
               'Curtailed energy in year 1 (%)': float(curtailment_percent)}

    return metrics

def determine_battery_cap(test_case, tc_si):
    # Get total battery capacity from test case system info
    battery_capacity_kwh = tc_si['Battery Capacity'].values.tolist()[0]
    
    # Define a dictionary to fill in with utilization metrics
    utilization = {}

    ### Calculate energy discharged by the battery in Year 1
    # Get time series of energy discharged
    test_case['Battery Energy Discharged (kWh)'] = test_case['Battery Discharge Power (kW)'] * (5/60)
    # Make sure we're only looking at Year 1
    year_1_mask = test_case['Datetime'].dt.year == 2012
    year_1_generation_kwh = test_case.loc[year_1_mask, 'Battery Energy Discharged (kWh)'].sum()
    # Calculate ideal energy discharged for the year – "ideal" means 100% DoD/day
    year_1_ideal_generation_kwh = battery_capacity_kwh * 365
    # Express capacity factor as ratio between actual and ideal utilization
    year_1_capacity_factor = year_1_generation_kwh / year_1_ideal_generation_kwh
    # Update the dictionary
    utilization["Year 1"] = year_1_capacity_factor

    ### Calculate energy discharged by the battery for each month
    for month in range(1,13):
        # Find total energy discharged this month
        month_mask = test_case['Datetime'].dt.month == month
        month_generation_kwh = test_case.loc[month_mask, 'Battery Energy Discharged (kWh)'].sum()

        # Calculate ideal energy discharged for the month – "ideal" means 100% DoD/day
        days_in_month = calendar.monthrange(2012, month)[1]
        month_ideal_generation_kwh = battery_capacity_kwh * days_in_month

        # Express capacity factor as ratio between actual and ideal utilization
        month_capacity_factor = month_generation_kwh / month_ideal_generation_kwh

        # Update dictionary
        month_as_string = calendar.month_name[month]
        utilization[month_as_string] = month_capacity_factor
    
    return utilization

def generate_dispatch_stack(test_case, day_to_study):
    """
    On a given day, create the dispatch stack at hourly time steps.
    NOTE: this does not calculate battery charging, only discharging
    
    Parameters:
        - test_case: The dictionary of system output
        - day_to_study: A string indicating the day that you want to generate a dispatch stack for

    Return:
    Dictionary with time series generation for each generation source
    Dictionary with excess generation
    """
    # filter out generation for a given day
    day_gen = test_case[test_case['Datetime'].dt.date == pd.to_datetime(day_to_study).date()].reset_index(drop=True)
    # store generation for every resource at each time step
    pv_gen = [0]* day_gen.shape[0] 
    wind_gen = [0]* day_gen.shape[0]
    geo_gen = [0]* day_gen.shape[0]
    batt_gen = [0]* day_gen.shape[0]
    # store leftover generation at each time step
    pv_excess = [0]* day_gen.shape[0]
    wind_excess = [0]* day_gen.shape[0]
    geo_excess = [0]* day_gen.shape[0]
    batt_excess = [0]* day_gen.shape[0]
    # calculate power used
    for i, row in day_gen.iterrows():
        load = row['Load (kW)']
        # utilize available geothermal
        load= determine_resource_use(load, row['Geothermal Generation (kW)'], geo_gen, geo_excess, i)
        if load == 0:
            continue
        # utilize available wind and solar
        if 'Net Wind Generation (kW)' in  day_gen.columns:
            load = determine_resource_use(load, row['Net Wind Generation (kW)'], wind_gen, wind_excess, i)
            if load == 0:
                continue
        load = determine_resource_use(load, row['PV to Grid (kW)'], pv_gen, pv_excess, i)
        if load == 0:
            continue
        # utilize available battery
        load = determine_resource_use(load, row['Battery Discharge Power (kW)'], batt_gen, batt_excess, i)
        if load == 0:
            continue


    # return dictionaries
    gen_dict = {'Net Wind Generation (kW)': wind_gen, 'PV to Grid (kW)': pv_gen, 'Geothermal Generation (kW)': geo_gen, 'Battery Discharge Power (kW)': batt_gen}
    excess_dict = {'Net Wind Generation (kW)': wind_excess, 'PV to Grid (kW)': pv_excess, 'Geothermal Generation (kW)': geo_excess, 'Battery Discharge Power (kW)': batt_excess}
    return gen_dict, excess_dict

def determine_resource_use(load, resource, resource_gen:list, resource_excess:list, i):
    # catch negative generation
    if resource < 0:
        return load
    # if all the load is met, store excess generation
    if load <= resource:
        resource_gen[i] = float(load)
        resource_excess[i] = float(resource - load)
        load = 0
    # if not all load is met, store remaining load
    else:
        load = load - resource
        resource_gen[i] = float(resource)
        resource_excess[i] = 0
    return load
    
def plot_dispatch_stack(generation_stack, gen_sources, file_pth, day_name):
    """
    Create a plot of generation stack & save
    """
    plt.clf()
    # colors for each generation type:
    color_dict = {'PV to Grid (kW)': 'gold', 'Net Wind Generation (kW)': 'royalblue', 'Geothermal Generation (kW)': 'tomato', 'Battery Discharge Power (kW)': 'silver'}
    x = range(len(generation_stack['PV to Grid (kW)']))
    alpha = 0.5
    # plot generation sources stacked on top
    order = [('Geothermal Generation (kW)', 'Geothermal'), ('PV to Grid (kW)', 'PV'), ('Net Wind Generation (kW)', 'Wind'), 
             ('Battery Discharge Power (kW)', 'Battery'), ]
    prev_gen = [0]*len(generation_stack['PV to Grid (kW)'])
    for gen, label in order:
        if gen not in gen_sources:
            continue
        plt.plot(x, prev_gen + np.array(generation_stack[gen]), label = label, color =  color_dict[gen])
        # fill between lines
        plt.fill_between(x, prev_gen, prev_gen + np.array(generation_stack[gen]), color =  color_dict[gen], alpha = alpha)
        prev_gen += np.array(generation_stack[gen])

    plt.xlabel('Timestep (5 min interval)')
    plt.ylabel('Generation Level')
    plt.title(f'Generation Dispatch for {day_name}')
    plt.legend(bbox_to_anchor=(1.25, 1.0))
    plt.tight_layout()
    # save figure
    plt.savefig(os.path.join(file_pth, f'{day_name}_dispatch_stack.png'), dpi=300, format='png', bbox_inches='tight')

def add_geothermal_timeseries(test_case, gen_sources, geo_mw = GEOTHERMAL_NAMEPLATE_MW, geo_cf = GEOTHERMAL_CAPACITY_FACTOR):
    # TO-DO: FIND OUT WHY THIS SEEMS TO ONLY BE WORKING FOR RAMPING DOWN
    unmet_load = test_case['Load (kW)'] - test_case['System to Grid (kW)'] # remaining load
    unmet_load = np.clip(unmet_load, 0, a_max=None) # Ensure that we only evaluate cases where there is actually unmet load
    geothermal_capacity_kW = geo_mw * geo_cf * 1000 # amount of geothermal available (assume constant)
    # As a first pass, the geothermal generation should be assumed to meet whatever unmet load 
    # there is (provided that this remains within the geothermal plant capacity)
    test_case['Geothermal Generation (kW)'] = [min(geothermal_capacity_kW, l) for l in unmet_load]
    gen_sources.append('Geothermal Generation (kW)')
    # Additionally, the geothermal plant is bound by a certain ramp rate, which we will now include
    max_change = MAX_GEOTHERMAL_STEP

    # Ensure the differences are within the limits
    for i in range(1, len(test_case)):
        prev_value = test_case.loc[i - 1, "Geothermal Generation (kW)"]
        current_value = test_case.loc[i, "Geothermal Generation (kW)"]
        difference = current_value - prev_value

        # Adjust if difference exceeds the maximum allowed
        if difference > max_change:
            adjusted_value = prev_value + max_change
        elif difference < -1*max_change:
            adjusted_value = prev_value - max_change
        else:
            adjusted_value = current_value
    
        # Finally, don't allow the geothermal generation to go above/below the max/min limits
        adjusted_value = np.clip(adjusted_value, GEOTHERMAL_MIN_GENERATION, GEOTHERMAL_NAMEPLATE_MW*1000)

        # Set the generation to the calculated value
        test_case.loc[i, "Geothermal Generation (kW)"] = adjusted_value

    test_case['Available Generation (kW)'] = test_case['System to Grid (kW)'] + np.array([geothermal_capacity_kW] * test_case['System to Grid (kW)'].shape[0])
    test_case['Generation to Grid (kW)'] = test_case['System to Grid (kW)'] + test_case['Geothermal Generation (kW)']
    return test_case, gen_sources

def store_results(file_pth, baseline_metrics = {}, generation_stack = {}, day_name = ''):
    """
    Store test case results
    """
    # store metrics
    if baseline_metrics != {}:
        with open(os.path.join(file_pth, 'metrics.json'), 'w') as f:
            json.dump(baseline_metrics, f, indent=4)

    if generation_stack != {}:
        generation_df = pd.DataFrame(generation_stack)
        generation_df.to_csv(os.path.join(file_pth, f'{day_name}_generation_stack.csv'), index=False)

def adjust_battery_dispatch(test_case):
    """
    We created the battery dispatch timeseries using a coulomb-counting method for SOC estimation.
    We then fed that into PySAM, which uses a more realistic method for SOC estimation.
    As an example, consider 7pm on January 31st. The battery discharges 1023 kWh over the 5 minute
    interval between 7pm and 7:05pm. Using coulomb counting alone, we would expect the battery SOC
    to reduce by 0.0028, or 0.28% during these 5 minutes. However, we see that the battery SOC 
    reduces by 0.035, or 3.5%. 
    When we created our dispatch strategy, we expected to still have energy available in the 
    battery at 7:05pm. As a result, we are using a higher ratio of battery dispatch to geothermal 
    dispatch during this time than we should be.
    Adjusting the dispatch strategy slightly is a realistic approach that would extend the battery
    duration slightly in order to meet the load without exceeding the geothermal ramp limits.
    """
    pass

if __name__ == "__main__":
    # names:
    available_gen_sources = ['Battery Discharge Power (kW)', 'PV to Grid (kW)']

    # read in stored data for test case
    case_name = 'remove_wind'
    test_case = pd.read_csv(os.path.join('data', 'test_cases', case_name, f'{case_name}.csv'))
    test_case_system_info = pd.read_csv(os.path.join('data', 'test_cases', case_name, f'{case_name}_system_info.csv'))

    # determine which generation resources are available
    gen_sources = [i for i in test_case.columns if i in available_gen_sources]
    test_case['System to Grid (kW)'] = test_case[gen_sources].sum(axis=1) # available PV/Wind
    test_case['Available Generation (kW)'] = test_case['System to Grid (kW)'] # uncurtailed PV/Wind + unramped geothermal
    test_case['Available Energy (kWh)'] = test_case['Available Generation (kW)'] * 5/60 # available energy
    test_case['Generation to Grid (kW)'] = test_case['System to Grid (kW)'] # uncurtailed PV/Wind + ramped geothermal

    # Ensure that the 'Datetime' column in the test case is full of Datetime objects
    test_case['Datetime'] = pd.to_datetime(test_case['Datetime'])

    # read in stored data for load
    load_filepath = 'data/Project 2 - Load Profile.xlsx'
    load = load_inspection_helpers.format_load_data(load_filepath=load_filepath)

    # add the load timeseries to the test case
    test_case = add_load_to_test_case(test_case=test_case, load_df=load)

    # add geothermal
    test_case, gen_sources = add_geothermal_timeseries(test_case, gen_sources)

    # calculate baseline metrics
    baseline_metrics = calculate_baseline_metrics(test_case, test_case_system_info)
    store_results(os.path.join('data', 'test_cases', case_name), baseline_metrics)

    # generate dispatch stack
    days_to_study = ['2012-01-16', '2012-04-30', '2012-05-20', '2012-07-27', '2012-09-11', '2012-10-01', '2012-11-15', '2012-12-22', '2012-12-24']
    for day_to_study in days_to_study:
        gen_dict, excess_dict = generate_dispatch_stack(test_case, day_to_study)

        # store results
        store_results(os.path.join('data', 'test_cases', case_name), {}, gen_dict, day_to_study)

        # # plot figure for dispatch
        plot_dispatch_stack(gen_dict, gen_sources, os.path.join('data', 'test_cases', case_name), day_name=day_to_study)
        # Convert to datetime object
        date = datetime.strptime(day_to_study, "%Y-%m-%d")

        # Start of the day
        start_of_day = date.strftime("%Y-%m-%d 00:00:00")

        # End of the day (midnight of the next day minus 1 second)
        end_of_day = (date + timedelta(days=1) - timedelta(seconds=1)).strftime("%Y-%m-%d 23:59:59")
        #pysam_helpers.plot_values_by_time_range(df=test_case, start_time=start_of_day, end_time=end_of_day, y_columns=['Load (kW)', 'Generation to Grid (kW)', 'Battery Charge Power (kW)'])
        #pysam_helpers.plot_values_by_time_range(df=test_case, start_time=start_of_day, end_time=end_of_day, y_columns=['Load (kW)', 'System to Grid (kW)', 'Geothermal Generation (kW)', 'Battery Discharge Power (kW)', 'Battery Charge Power (kW)'])
        #pysam_helpers.plot_values_by_time_range(df=test_case, start_time=start_of_day, end_time=end_of_day, y_columns=['Battery SOC'])