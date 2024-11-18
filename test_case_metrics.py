import pandas as pd
import json
import matplotlib.pyplot as plt
import os

def calculate_baseline_metrics(test_case):
    """
    Determine baseline metrics for the given test case.
    Metrics include:
     - whether the net generation is greater than the net load (feasibility) (bool)
     - total curtailed wind and solar (measure of excess generation) (float)
     - average battery SOC (measure of adequate battery sizing) (array)

     Params:
     test_case (pd.DataFrame): contains time series of power generated for wind, solar, battery, geothermal, and load
     Return:
     Dictionary with stored metrics
    """
    # check if the system is feasible throughout the year
    is_feas = determine_feasibility(test_case)
    # calculate curtailed wind and solar
    percent_curtailed = determine_curtailed_ren(test_case)
    # calculate average battery SOC
    battery_SOC = determine_battery_SOC(test_case)

    return {'feasibility': is_feas, 'percent_curtailed': percent_curtailed, 'battery_SOC': battery_SOC}

def determine_feasibility(test_case):
    # TODO FILL IN
    return False

def determine_curtailed_ren(test_case):
    # TODO FILL IN
    return 0

def determine_battery_SOC(test_case):
    # TODO FILL IN
    return []

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



