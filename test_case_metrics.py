import pandas as pd
import json
import matplotlib.pyplot as plt
import os
import numpy as np

def calculate_baseline_metrics(test_case):
    """
    Determine baseline metrics for the given test case.
    Metrics include:
     - whether the net generation is greater than the net load (feasibility) (bool)
     - total curtailed wind and solar (measure of excess generation) (float)
     - average battery SOC each month (measure of adequate battery sizing) (array size 12)

     Params:
     test_case (pd.DataFrame): contains time series of power generated for wind, solar, battery, geothermal, and load
     Return:
     Dictionary with stored metrics
    """
    # check if the system is feasible throughout the year
    is_feas, infeas_steps = determine_feasibility(test_case)
    # calculate curtailed wind and solar
    percent_curtailed = determine_curtailed_ren(test_case)
    # calculate average battery SOC
    avg_SOC = determine_battery_SOC(test_case)

    return {'feasibility': [is_feas, infeas_steps], 'percent_curtailed': percent_curtailed, 'avg_SOC': avg_SOC}

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

def determine_battery_SOC(test_case):
    batt_SOC = np.array(test_case['batt_SOC'])
    tsm = 8760 # number of time steps per month
    avg_batt = batt_SOC.reshape(-1, tsm).mean(axis=1)
    return [avg_batt]

def generate_dispatch_stack(test_case, day: pd.DatetimeIndex):
    """
    On a given day, create the dispatch stack at hourly time steps.
    Return:
    Dictionary with time series generation for each generation source
    """
    # TODO FILL IN
    return {}

def plot_dispatch_stack(generation_stack):
    """
    Create a plot of generation stack & save
    """
    # TODO FILL IN

def store_results(case_name, baseline_metrics = {}, generation_stack = {}):
    """
    Store test case results
    """
    # TODO FILL IN

if __name__ == '__main__':
    # read in stored data for test case
    case_name = 'base_case0'
    test_case = pd.read_csv(os.path.join('data', 'test_cases', case_name, f'{case_name}.csv'))

    # calculate baseline metrics
    baseline_metrics = calculate_baseline_metrics(test_case)



