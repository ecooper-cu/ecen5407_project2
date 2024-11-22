
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
import numpy as np
import load_inspection_helpers

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
    # Ensure that the 'Datetime' column in the test case is full of Datetime objects
    test_case['Datetime'] = pd.to_datetime(test_case['Datetime'])
    
    # check if the system is feasible throughout the year
    test_case, is_feas, infeas_steps = determine_feasibility(test_case)
    unmet_load_metrics = {}
    excess_gen_metrics = calculate_excess_generation(test_case)
    if not is_feas:
        unmet_load_metrics = calculate_reliability_margin(test_case)
    # calculate curtailed wind and solar
    curtailment_report = calculate_curtailment_metrics(test_case, excess_threshold_percent=0.001)
    energy_curtailed = curtailment_report['Curtailed energy in year 1 (kWh)']
    percent_curtailed = curtailment_report['Curtailed energy in year 1 (%)']
    # calculate average battery SOC
    avg_cap = determine_battery_cap(test_case, test_case_system_info)

    return {'feasibility': [is_feas, infeas_steps, unmet_load_metrics],
             'energy_curtailed': energy_curtailed,
             'percent_curtailed': percent_curtailed, 
             'avg_battery_capacity_factor': avg_cap}

def determine_feasibility(test_case):
    # sum over all generation
    total_gen = float(test_case['Generation to Grid (kW)'].sum(axis=0))
    total_load = float(sum(test_case['Load (kW)']))
    net_feas = total_gen >= total_load
    # store timesteps when load > gen
    infeas_timesteps = [test_case['Datetime'].iloc[i]for i in range(test_case.shape[0]) if test_case['Generation to Grid (kW)'].iloc[i] < test_case['Load (kW)'].iloc[i]]
    test_case['Unmet Load'] = test_case['Generation to Grid (kW)'] - test_case['Load (kW)']
    return test_case, net_feas, infeas_timesteps

def calculate_reliability_margin(test_case:pd.DataFrame, threshold = 1e-3):
    unmet_threshold = (test_case['Load (kW)'].max()) * threshold # calculating .1% of peak load
    unmet_load_mask = test_case['Unmet Load'] > unmet_threshold # whether unmet load is greater than .1% peak load
    unmet_load_instances = unmet_load_mask.sum() # number of unmet loads above threshold
    unmet_load_magnitude = test_case['Unmet Load'][unmet_load_mask].sum() # total amount of unmet load above threshold
    unmet_load_percent = (unmet_load_magnitude / (test_case['Load (kW)'].sum())) * 100 # what percent of load is unmet
    return {'number_unmet_load': unmet_load_instances, 'total_unmet_load': unmet_load_magnitude, 'unmet_load_percent': unmet_load_percent}

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
    year_1_generation_kwh = test_case.loc[year_1_mask, 'Energy to Grid (kWh)'].sum()
    curtailment_percent = (curtailment_magnitude / year_1_generation_kwh) * 100

    metrics = {'Curtailed energy in year 1 (kWh)': curtailment_magnitude, 
               'Curtailed energy in year 1 (%)': curtailment_percent}

    return metrics

def determine_battery_cap(test_case, tc_si):
    # calculate capacity factor at hourly timestep
    battery_capacity = tc_si['Battery Capacity'].values.tolist()[0]
    batt_cap = np.array(test_case['Battery Discharge Power (kW)'])/battery_capacity
    # split into monthly steps
    monthly_cfs = np.array_split(batt_cap, len(batt_cap) // 8760)
    monthly_avg_cf = [] # store monthly average
    for cf in monthly_cfs:
        # remove negative values (indicating charging)
        cf = cf[cf >= 0]
        # compute average
        monthly_avg_cf.append(float(np.mean(cf)))
    return monthly_avg_cf

def generate_dispatch_stack(test_case, index_range:list):
    """
    On a given day, create the dispatch stack at hourly time steps.
    NOTE: this does not calculate battery charging, only discharging

    Return:
    Dictionary with time series generation for each generation source
    Dictionary with excess generation
    """
    # TODO replace index_range with a date, then pull the load and generation associated with that date
    day_gen = test_case.iloc[index_range[0]: index_range[1]]
    # store generation levels at every time step and excess generation
    pv_gen = [0]* day_gen.shape[0]
    wind_gen = [0]* day_gen.shape[0]
    geo_gen = [0]* day_gen.shape[0]
    batt_gen = [0]* day_gen.shape[0]
    pv_excess = [0]* day_gen.shape[0]
    wind_excess = [0]* day_gen.shape[0]
    geo_excess = [0]* day_gen.shape[0]
    batt_excess = [0]* day_gen.shape[0]
    # at each time step, calculate power used
    for i, row in day_gen.iterrows():
        ind = i - index_range[0]
        # start by utilizing available wind and solar
        load = row['Load (kW)']
        load = determine_resource_use(load, row['Net Wind Generation (kW)'], wind_gen, wind_excess, ind)
        if load == 0:
            continue
        load = determine_resource_use(load, row['PV to Grid (kW)'], pv_gen, pv_excess, ind)
        if load == 0:
            continue
        # utilize available battery
        load = determine_resource_use(load, row['Battery Discharge Power (kW)'], batt_gen, batt_excess, ind)
        if load == 0:
            continue
        # utilize available geothermal
        load= determine_resource_use(load, row['geothermal'], geo_gen, geo_excess, ind)
        if load == 0:
            continue

    # return dictionaries
    gen_dict = {'Net Wind Generation (kW)': wind_gen, 'PV to Grid (kW)': pv_gen, 'geothermal': geo_gen, 'Battery Discharge Power (kW)': batt_gen}
    excess_dict = {'Net Wind Generation (kW)': wind_excess, 'PV to Grid (kW)': pv_excess, 'geothermal': geo_excess, 'Battery Discharge Power (kW)': batt_excess}
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
    
def plot_dispatch_stack(generation_stack, file_pth, day_name):
    """
    Create a plot of generation stack & save
    """
    # colors for each generation type:
    color_dict = {'PV to Grid (kW)': 'gold', 'Net Wind Generation (kW)': 'royalblue', 'geothermal': 'tomato', 'Battery Discharge Power (kW)': 'silver'}
    x = range(len(generation_stack['PV to Grid (kW)']))
    alpha = 0.5
    # plot generation sources stacked on top
    plt.plot(x, np.array(generation_stack['PV to Grid (kW)']), label = 'PV', color =  color_dict['PV to Grid (kW)'])
    curr_gen_0 = np.array(generation_stack['PV to Grid (kW)'])
    plt.plot(x, np.array(generation_stack['Net Wind Generation (kW)']) + curr_gen_0, label = 'Wind', color =  color_dict['Net Wind Generation (kW)'])
    curr_gen_1 = curr_gen_0 + np.array(generation_stack['Net Wind Generation (kW)'])
    plt.plot(x, np.array(generation_stack['geothermal']) + curr_gen_1, label = 'Geothermal', color =  color_dict['geothermal'])
    curr_gen_2 = curr_gen_1 + np.array(generation_stack['geothermal'])
    plt.plot(x, np.array(generation_stack['Battery Discharge Power (kW)']) + curr_gen_2, label = 'Batt', color =  color_dict['Battery Discharge Power (kW)'])
    # fill between lines
    plt.fill_between(x, [0] * len(x), curr_gen_0, color = color_dict['PV to Grid (kW)'], alpha = alpha)
    plt.fill_between(x, curr_gen_0, curr_gen_1, color = color_dict['Net Wind Generation (kW)'], alpha = alpha)
    plt.fill_between(x, curr_gen_2, curr_gen_2, color = color_dict['geothermal'], alpha = alpha)
    plt.fill_between(x, curr_gen_2, curr_gen_2 + np.array(generation_stack['Battery Discharge Power (kW)']), color = color_dict['Battery Discharge Power (kW)'], alpha = alpha)
    plt.xlabel('Hourly Timestep')
    plt.ylabel('Generation Level')
    plt.title('Generation Dispatch')
    plt.legend()
    # save figure
    plt.savefig(os.path.join(file_pth, f'{day_name}_dispatch_stack.png'), dpi=300, format='png')

def add_geothermal_timeseries(test_case, geo_mw = 77, geo_cf = 0.95):
    unmet_load = test_case['Load (kW)'] - test_case['System to Grid (kW)'] # remaining load
    geothermal_capacity_kW = geo_mw * geo_cf * 1000 # amount of geothermal available (assume constant)
    test_case['Geothermal Generation (kW)'] = [min(geothermal_capacity_kW, l) for l in unmet_load] # geothermal time series
    test_case['Available Generation (kW)'] = test_case['System to Grid (kW)'] + np.array([geothermal_capacity_kW] * test_case['System to Grid (kW)'].shape[0])
    test_case['Generation to Grid (kW)'] = test_case['System to Grid (kW)'] + test_case['Geothermal Generation (kW)']
    return test_case

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
       


if __name__ == '__main__':
    # names:
    available_gen_sources = ['Battery Discharge Power (kW)', 'PV to Grid (kW)', 'Net Wind Generation (kW)']

    # read in stored data for test case
    case_name = 'base_case0'
    test_case = pd.read_csv(os.path.join('data', 'test_cases', case_name, f'{case_name}_gen.csv'))
    test_case_system_info = pd.read_csv(os.path.join('data', 'test_cases', case_name, f'{case_name}_system_info.csv'))

    # determine which generation resources are available
    gen_sources = [i for i in test_case.columns if i in available_gen_sources]
    test_case['System to Grid (kW)'] = test_case[gen_sources].sum(axis=1) # available PV/Wind
    test_case['Available Generation (kW)'] = test_case['System to Grid (kW)'] # uncurtailed PV/Wind + unramped geothermal
    test_case['Available Energy (kWh)'] = test_case['Available Generation (kW)'] * 5/60 # available energy
    test_case['Generation to Grid (kW)'] = test_case['System to Grid (kW)'] # uncurtailed PV/Wind + ramped geothermal
    # add geothermal
    test_case = add_geothermal_timeseries(test_case)

    # calculate baseline metrics
    baseline_metrics = calculate_baseline_metrics(test_case, test_case_system_info)

    # read in stored data for load
    load_filepath = 'data/Project 2 - Load Profile.xlsx'
    load = load_inspection_helpers.format_load_data(load_filepath=load_filepath)

    # add the load timeseries to the test case
    test_case = add_load_to_test_case(test_case=test_case, load_df=load)

    # generate dispatch stack
    gen_dict, excess_dict = generate_dispatch_stack(test_case, [196, 210])

    # store results
    store_results(os.path.join('data', 'test_cases', case_name), baseline_metrics, gen_dict, 'TEST_196_210')

    # # plot figure for dispatch
    # plot_dispatch_stack(gen_dict, os.path.join('data', 'test_cases', case_name), 'TEST_196_210')
 