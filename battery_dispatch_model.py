# %% Module imports
### PySAM stuff
# Get performance model for each subsystem
import PySAM.Pvsamv1 as pv_model
import PySAM.Windpower as wind_model
import PySAM.Battery as battery_model

# Get function for managing hybrid variables and simulations
from PySAM.Hybrids.HybridSystem import HybridSystem

# To load inputs from SAM
import json

# To organize and plot outputs from PySAM simulation
import pandas as pd
import pysam_helpers
import numpy as np

### Load stuff
import load_inspection_helpers

# %% Define functions
def run_pysam_model(inputs_file):
        # Load the inputs from the JSON file
        # Note that for the Hybrid System, we use a single JSON file rather than a file per module.
        # The JSON file referenced here is from SAM code generator for a PV Wind Battery sytem with a
        # Single Owner financial model
        print("Loading inputs...")
        with open(inputs_file, 'r') as f:
                inputs = json.load(f)['input']

        # Create the hybrid system model using performance model names as defined by the import 
        # statements above. The string 'singleowner' indicates the financial model ('hostdeveloper' would
        # be another option).
        print("Creating hybrid system...")
        m = HybridSystem([pv_model, wind_model, battery_model], 'singleowner')
        m.new()

        # Load the inputs from the JSON file into the main module
        print("Assigning inputs to hybrid system....")
        unassigned = m.assign(inputs) # returns a list of unassigned variables if any
        print(unassigned)

        # Forbid the battery from charging from the system – now it can't participate
        m.battery.value("batt_dispatch_auto_can_charge", 0)

        # Run a simulation
        print("Running PySAM simulation...")

        m.execute()

        return m

def produce_generation_dataframe(m):
        # Create a dictionary of DataFrames with the outputs from each model
        pv_model_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m.pv)
        wind_model_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m.wind)

        # Build a dataframe of generation data
        pv_df = pv_model_outputs['Lifetime 5 Minute Data'].reset_index()
        pv_gen = pv_df[['Datetime', 'subarray1_dc_gross', 'gen']]
        pv_gen.rename(columns={'subarray1_dc_gross':'PV DC Gross (kW)', 'gen': 'PV Generation (kW)'}, inplace=True)
        wind_df = wind_model_outputs['5 Minute Data'].reset_index()
        wind_gen = wind_df[['Datetime', 'gen']]
        wind_gen.rename(columns={'gen':'Wind Generation (kW)'}, inplace=True)
        gen = pd.merge(pv_gen, wind_gen, on='Datetime', how='inner')
        gen['Datetime'] = pd.to_datetime(gen['Datetime'])
        gen['RE Power (kW)'] = gen['PV Generation (kW)'] + gen['Wind Generation (kW)']

        #Add geothermal generation, assuming flat generation on 77MW capacity with 95% capacity factor
        gen['Net Power (kW)'] = gen['RE Power (kW)'] + 77 * 1000 * 0.95

        return gen

def produce_load_dataframe(load_filepath):
        # Build a dataframe of load data
        load = load_inspection_helpers.format_load_data(load_filepath)
        load['Load (kW)'] = load['Load (MW)'] * 1000

        return load

def battery_dispatch_model_with_ramp_limits(merged):
    """
    Dispatch model for lithium-ion battery with geothermal plant ramp limits.

    Parameters:
            merged (pd.DataFrame): DataFrame with columns "Datetime", "PV Generation (kW)",
                                   "Wind Generation (kW)", "Load (kW)"
    
    Returns:
            pd.DataFrame: Dataframe with three additional columns:
                            "Battery Power Target (kW)": Battery power target in kW (negative
                                    for charging, positive for discharging)
                            "SOC": Battery state of charge 
                            "Expected Geothermal Output (kW)": Expected geothermal output (kW)
    """

    # Calculate total renewable generation
    merged["Renewable Generation (kW)"] = merged["PV Generation (kW)"] + merged["Wind Generation (kW)"]

    # Initialize columns for SOC, battery target power, and geothermal output
    merged["SOC"] = 0.0  # Start at 50% SOC
    merged["Battery Power Target (kW)"] = 0.0
    merged["Expected Geothermal Output (kW)"] = 0.0

    # Loop through timesteps to update SOC incrementally
    for t in range(len(merged)):
        # Get previous SOC and geothermal output
        if t >= 1:
            prev_soc = merged.at[t - 1, "SOC"]
            prev_geo_output = merged.at[t - 1, "Expected Geothermal Output (kW)"]
        else:
            # Start with battery at 50% SOC and load met 100% by geothermal
            prev_soc = INITIAL_SOC
            prev_geo_output = merged.at[0, "Load (kW)"]

        # Find what the current load is
        load = merged.at[t, "Load (kW)"]
        
        # Predict what the load will be at the next timestep
        if t < (len(merged)-1):
            next_load = merged.at[t+1, "Load (kW)"]
        else:
            next_load = merged.at[t, "Load (kW)"]

        # Find what's currently available from wind and solar
        renewable_generation = merged.at[t, "Renewable Generation (kW)"]

        # Available charge and discharge power based on SOC
        # In practice, the battery would be limited much sooner than this charge limit would 
        # kick in. Especially for charging, it's not possible to run at nameplate power right up to 
        # 100% SOC (or even 95% SOC), because the battery has to taper with CV charging. SAM will 
        # take care of this derating.
        # (SOC_MAX - current_soc) * BATTERY_ENERGY_CAPACITY calculates the amount of energy
        # (in kWh) that the battery can accept. Dividing by the timestep converts this 
        # energy capacity into a rate (in kW) for a 5-minute timestep. Dividing by the 
        # charging efficiency accounts for losses during charging.
        max_charge_power = min(BATTERY_MAX_CHARGE_POWER, 
                               (SOC_MAX - prev_soc) * BATTERY_ENERGY_CAPACITY / 
                               (CHARGING_EFFICIENCY * timestep))
        max_discharge_power = min(BATTERY_MAX_DISCHARGE_POWER, 
                                  (prev_soc - SOC_MIN) * BATTERY_ENERGY_CAPACITY / 
                                  (DISCHARGING_EFFICIENCY * timestep))

        # Calculate the minimum and maximum allowable geothermal output based on the ramp rate
        max_geo_output = prev_geo_output + GEOTHERMAL_RAMP_RATE * GEOTHERMAL_CAPACITY
        # Ensures that ramp does not exceed maximum generation levels
        max_geo_output = np.clip(max_geo_output, max=GEOTHERMAL_CAPACITY) 

        min_geo_output = prev_geo_output - GEOTHERMAL_RAMP_RATE * GEOTHERMAL_CAPACITY
        # Ensures that ramp does not exceed minimum generation levels
        min_geo_output = np.clip(min_geo_output, min=GEOTHERMAL_MIN_GENERATION) 

        # Dispatch logic
        if (load <= min_geo_output):
            # The minimum geothermal output exceeds the load! Geothermal output should be ramped 
            # down, and battery should charge with any extra power.

            # Set the geothermal output as low as possible
            geothermal_output = min_geo_output
            
            # Charge the battery with excess power
            excess_power = (geothermal_output + renewable_generation) - load 
            charge_power = min(excess_power, max_charge_power)
            battery_power_target = -1 * charge_power # Charging is negative
            soc_step = charge_power * CHARGING_EFFICIENCY * timestep / BATTERY_ENERGY_CAPACITY
        else:
            # The load exceeds the minimum geothermal output.
            # Meet the load using least-cost methods first, and stay within geothermal ramp limits
            if load <= (renewable_generation + min_geo_output):
                # Set the geothermal output as low as possible
                geothermal_output = min_geo_output
                
                # Charge the battery with excess power
                excess_power = (geothermal_output + renewable_generation) - load 
                charge_power = min(excess_power, max_charge_power)
                battery_power_target = -1 * charge_power # Charging is negative
                soc_step = charge_power * CHARGING_EFFICIENCY * timestep / BATTERY_ENERGY_CAPACITY # SOC should increase
            else:
                if load >= (renewable_generation + max_discharge_power):
                    # Set battery discharging power as high as possible
                    discharge_power = max_discharge_power
                    battery_power_target = discharge_power # Discharging is positive
                    soc_step = -1 * discharge_power * DISCHARGING_EFFICIENCY * timestep / BATTERY_ENERGY_CAPACITY # SOC should decrease
                    
                    # Set the geothermal output to fill in the difference
                    deficit = load - (renewable_generation + battery_power_target)
                    geothermal_output = np.clip(deficit, min_geo_output, max_geo_output)
                else:
                    # Keep the geothermal output constant
                    geothermal_output = np.clip(prev_geo_output, min_geo_output, max_geo_output)
                    
                    # Discharge the battery to meet the load
                    deficit = load - (renewable_generation + geothermal_output)
                    discharge_power = min(deficit, max_discharge_power)
                    battery_power_target = discharge_power # Discharging is positive
                    soc_step = -1 * discharge_power * DISCHARGING_EFFICIENCY * timestep / BATTERY_ENERGY_CAPACITY # SOC should decrease

        # Clip SOC to its limits
        new_soc = np.clip((prev_soc + soc_step), SOC_MIN, SOC_MAX)

        # Update values in array for next iteration
        merged.at[t, "SOC"] = new_soc
        merged.at[t, "Battery Power Target (kW)"] = battery_power_target
        merged.at[t, "Expected Geothermal Output (kW)"] = geothermal_output

    return merged

def get_battery_utilization(result, m):
        # From initial analysis, the battery is highly underutilized:
        # Grab discharging targets from the battery dispatch time series
        result['Battery Discharge Target (kW)'] = result['Battery Power Target (kW)'][result['Battery Power Target (kW)'] > 0]

        # Calculate energy discharged in year 1 based on these targets
        result['Battery Energy Discharged (kWh)'] = result['Battery Discharge Target (kW)'] * (5/60)
        year_1_mask = result.index < pd.Timestamp(f'2013-01-01 00:00:00')
        year_1_battery_generation_kwh = result.loc[year_1_mask, 'Battery Energy Discharged (kWh)'].sum()

        # We need a baseline against which to evaluate the utilization of the battery. We will choose an 
        # 'ideal' dispatch of 100% of the battery's capacity per day as this baseline.
        battery_capacity_kWhdc = m.battery.BatterySystem.batt_computed_bank_capacity
        ideal_annual_battery_generation_kwh = battery_capacity_kWhdc * 365

        # Compare actual energy discharged in year 1 against ideal energy discharged in year 1
        battery_capacity_factor = year_1_battery_generation_kwh / ideal_annual_battery_generation_kwh
        print(f"Battery utilization is {battery_capacity_factor * 100:.2f}% of the ideal utilization.")

# %% Execute PySAM model
m = run_pysam_model(inputs_file='data/PySam_Inputs/Hybrid_Project/Hybrid.json')

# %% Define some system characteristics that the dispatch model requires
# Battery stuff from SAM
BATTERY_ENERGY_CAPACITY = m.battery.BatterySystem.batt_computed_bank_capacity
BATTERY_MAX_DISCHARGE_POWER = m.battery.BatterySystem.batt_power_discharge_max_kwac
BATTERY_MAX_CHARGE_POWER = m.battery.BatterySystem.batt_power_charge_max_kwac
SOC_MIN = m.battery.BatteryCell.batt_minimum_SOC
SOC_MAX = m.battery.BatteryCell.batt_maximum_SOC
INITIAL_SOC = m.battery.BatteryCell.batt_initial_SOC
CHARGING_EFFICIENCY = m.battery.BatterySystem.batt_ac_dc_efficiency
DISCHARGING_EFFICIENCY = m.battery.BatterySystem.batt_dc_ac_efficiency
#ROUNDTRIP_EFFICIENCY = m.battery.BatterySystem.batt_ac_dc_efficiency * m.battery.BatterySystem.batt_dc_ac_efficiency

# Geothermal stuff
GEOTHERMAL_CAPACITY_FACTOR = 0.95
GEOTHERMAL_CAPACITY = 77 * GEOTHERMAL_CAPACITY_FACTOR * 1000    # Capacity in kW
GEOTHERMAL_MIN_GENERATION = 0.1 * GEOTHERMAL_CAPACITY
GEOTHERMAL_RAMP_RATE = 0.05

# Iteration stuff
timestep = 5/60 # 5-minute timestep in hours

# %% Prepare dataset
load = produce_load_dataframe(load_filepath='data/Project 2 - Load Profile.xlsx')
gen = produce_generation_dataframe(m=m)

#%% Merge the two dataframes
merged = pd.merge(gen, load, on='Datetime', how='inner')

# %% Run dispatch model
result = battery_dispatch_model(merged)

# %% Generate some plots
date_start = '2012-12-21 00:00:00'
date_end = '2012-12-22 00:00:00'
result.set_index('Datetime', inplace=True)
pysam_helpers.plot_values_by_time_range(df=result, start_time=date_start, end_time=date_end, y_columns=['PV Generation (kW)', 'Wind Generation (kW)', 'Load (kW)', 'Battery Power Target (kW)'])
pysam_helpers.plot_values_by_time_range(df=result, start_time=date_start, end_time=date_end, y_columns=['SOC'])

# %% Print the battery utilization
get_battery_utilization(result, m)

# %% Generate a csv with the dispatch target
result['Battery Power Target (kW)'].to_csv('data/PySam_Inputs/Hybrid_Project/dispatch_target_5min.csv', index=False)

# %% Generate a csv of system power output without the battery
#result.to_csv('data/PySAM_Outputs/baseline_system_output.csv')
