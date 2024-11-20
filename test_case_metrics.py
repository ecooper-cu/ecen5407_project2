
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
import numpy as np

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
    is_feas, infeas_steps = determine_feasibility(test_case)
    # calculate curtailed wind and solar
    percent_curtailed = determine_curtailed_ren(test_case)
    # calculate average battery SOC
    avg_cap = determine_battery_cap(test_case, test_case_system_info)

    return {'feasibility': [is_feas, infeas_steps], 'percent_curtailed': percent_curtailed, 'avg_battery_capacity_factor': avg_cap}

def determine_feasibility(test_case):
    # sum over all generation
    gen_timestep = test_case[['pv', 'wind', 'geothermal', 'batt']].sum(axis=1)
    total_gen = float(gen_timestep.sum(axis=0))
    total_load = float(sum(test_case['load']))
    net_feas = total_gen >= total_load
    # store timesteps when load > gen
    infeas_inds = [i for i in range(test_case.shape[0]) if gen_timestep.iloc[i] < test_case['load'].iloc[i]]
    return net_feas, infeas_inds

def determine_curtailed_ren(test_case):
    # TODO FILL IN
    return 0

def determine_battery_cap(test_case, tc_si):
    # calculate capacity factor at hourly timestep
    battery_capacity = tc_si['Battery capacity'].values.tolist()[0]
    batt_cap = np.array(test_case['batt'])/battery_capacity
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
        load = row['load']
        load = determine_resource_use(load, row['wind'], wind_gen, wind_excess, ind)
        if load == 0:
            continue
        load = determine_resource_use(load, row['pv'], pv_gen, pv_excess, ind)
        if load == 0:
            continue
        # TODO check order on this
        # utilize available geothermal
        load= determine_resource_use(load, row['geothermal'], geo_gen, geo_excess, ind)
        if load == 0:
            continue
        # utilize available battery
        load = determine_resource_use(load, row['batt'], batt_gen, batt_excess, ind)
        if load == 0:
            continue
    # return dictionaries
    gen_dict = {'wind': wind_gen, 'pv': pv_gen, 'geothermal': geo_gen, 'batt': batt_gen}
    excess_dict = {'wind': wind_excess, 'pv': pv_excess, 'geothermal': geo_excess, 'batt': batt_excess}
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
    color_dict = {'pv': 'gold', 'wind': 'royalblue', 'geothermal': 'tomato', 'batt': 'silver'}
    x = range(len(generation_stack['pv']))
    alpha = 0.5
    # plot generation sources stacked on top
    plt.plot(x, np.array(generation_stack['pv']), label = 'PV', color =  color_dict['pv'])
    curr_gen_0 = np.array(generation_stack['pv'])
    plt.plot(x, np.array(generation_stack['wind']) + curr_gen_0, label = 'Wind', color =  color_dict['wind'])
    curr_gen_1 = curr_gen_0 + np.array(generation_stack['wind'])
    plt.plot(x, np.array(generation_stack['geothermal']) + curr_gen_1, label = 'Geothermal', color =  color_dict['geothermal'])
    curr_gen_2 = curr_gen_1 + np.array(generation_stack['geothermal'])
    plt.plot(x, np.array(generation_stack['batt']) + curr_gen_2, label = 'Batt', color =  color_dict['batt'])
    # fill between lines
    plt.fill_between(x, [0] * len(x), curr_gen_0, color = color_dict['pv'], alpha = alpha)
    plt.fill_between(x, curr_gen_0, curr_gen_1, color = color_dict['wind'], alpha = alpha)
    plt.fill_between(x, curr_gen_2, curr_gen_2, color = color_dict['geothermal'], alpha = alpha)
    plt.fill_between(x, curr_gen_2, curr_gen_2 + np.array(generation_stack['batt']), color = color_dict['batt'], alpha = alpha)
    plt.xlabel('Hourly Timestep')
    plt.ylabel('Generation Level')
    plt.title('Generation Dispatch')
    plt.legend()
    # save figure
    plt.savefig(os.path.join(file_pth, f'{day_name}_dispatch_stack.png'), dpi=300, format='png')

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
    # read in stored data for test case
    case_name = 'base_case0'
    test_case = pd.read_csv(os.path.join('data', 'test_cases', case_name, f'{case_name}_gen.csv'))
    test_case_system_info = pd.read_csv(os.path.join('data', 'test_cases', case_name, f'{case_name}_system_info.csv'))

    # calculate baseline metrics
    baseline_metrics = calculate_baseline_metrics(test_case, test_case_system_info)

    # generate dispatch stack
    gen_dict, excess_dict = generate_dispatch_stack(test_case, [196, 210])

    # store results
    store_results(os.path.join('data', 'test_cases', case_name), baseline_metrics, gen_dict, 'TEST_196_210')

    # # plot figure for dispatch
    # plot_dispatch_stack(gen_dict, os.path.join('data', 'test_cases', case_name), 'TEST_196_210')
 