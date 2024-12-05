# %% Module imports
### PySAM stuff
# Get performance model for each subsystem
import PySAM.Pvsamv1 as PVSAM
import PySAM.Grid as Grid
import PySAM.Utilityrate5 as UtilityRate
import PySAM.Singleowner as SingleOwner
#import PySAM.Windpower as wind_model
#import PySAM.Battery as battery_model

# Get function for managing hybrid variables and simulations
#from PySAM.Hybrids.HybridSystem import HybridSystem

# To load inputs from SAM
import json

# To organize and plot outputs from PySAM simulation
import pandas as pd
import pysam_helpers
import numpy as np

### Load stuff
import load_inspection_helpers
from geothermal_constants import *

# %% Define functions
def run_pysam_model(inputs_file):
        # Create a new instance of each module
        pvbatt_model = PVSAM.new()
        grid = Grid.from_existing(pvbatt_model)
        utility_rate = UtilityRate.from_existing(pvbatt_model)
        single_owner = SingleOwner.from_existing(pvbatt_model)

        # Load the inputs from the JSON file into each module
        file_names = ["pvsamv1", "grid", "utilityrate5", "singleowner"]
        modules = [pvbatt_model, grid, utility_rate, single_owner]
        for f, m in zip(file_names, modules):
            filepath = inputs_file + "_" + f + '.json'
            print(f"Loading inputs from: {filepath}")
            with open(filepath, 'r') as f:
                data = json.load(f)
                # loop through each key-value pair
                for k, v in data.items():
                    # Note: I'm ignoring any 'adjustment factors' here, but these can be set afterwards.
                    # See: https://nrel-pysam.readthedocs.io/en/main/modules/Pvsamv1.html#adjustmentfactors-group
                    if k != "number_inputs" and 'adjust_' not in k:
                        m.value(k, v)

        # Override battery dispatch to prevent participation
        pvbatt_model.value("batt_custom_dispatch", np.zeros(len(pvbatt_model.value("batt_custom_dispatch"))))

        # Run a simulation
        print("Running PySAM simulation...")
        for m in modules:
            m.execute()

        module_dict = {
            "pvbatt_model":pvbatt_model,
            "grid":grid,
            "utility_rate":utility_rate,
            "single_owner":single_owner
        }

        return module_dict

def produce_generation_dataframe(module_dict):
        # Create a dictionary of DataFrames with the outputs from each model
        pvbatt_model_outputs = pysam_helpers.parse_model_outputs_into_dataframes(module_dict["pvbatt_model"], five_minutes_only=True)
        #wind_model_outputs = pysam_helpers.parse_model_outputs_into_dataframes(m.wind, five_minutes_only=True)

        # Build a dataframe of generation data
        pvbatt_df = pvbatt_model_outputs['Lifetime 5 Minute Data'].reset_index()
        gen = pvbatt_df[['Datetime', 'subarray1_dc_gross', 'gen']]
        gen.rename(columns={'subarray1_dc_gross':'PV DC Gross (kW)', 'gen': 'PV Generation (kW)'}, inplace=True)
        #wind_df = wind_model_outputs['5 Minute Data'].reset_index()
        #wind_gen = wind_df[['Datetime', 'gen']]
        #wind_gen.rename(columns={'gen':'Wind Generation (kW)'}, inplace=True)
        #gen = pd.merge(pv_gen, wind_gen, on='Datetime', how='inner')
        gen['Datetime'] = pd.to_datetime(gen['Datetime'])
        gen['System Generation (kW)'] = gen['PV DC Gross (kW)']

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
    #merged["Renewable Generation (kW)"] = merged["PV Generation (kW)"] + merged["Wind Generation (kW)"]
    merged["Renewable Generation (kW)"] = merged["System Generation (kW)"]

    # Make sure that renewable generation is always positive
    merged["Renewable Generation (kW)"] = np.clip(merged["Renewable Generation (kW)"], a_min = 0, a_max = None)

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
            prev_battery_target = merged.at[t-1, "Battery Power Target (kW)"]
        else:
            # Start with battery at 50% SOC and load met 100% by geothermal
            prev_soc = INITIAL_SOC
            prev_geo_output = merged.at[0, "Load (kW)"]
            prev_battery_target = merged.at[0, "Battery Power Target (kW)"]

        # Find what the current load is
        load = merged.at[t, "Load (kW)"]
        
        # Predict what the load will be at the next timestep
        if t < (len(merged)-1):
            next_load = merged.at[t+1, "Load (kW)"]
            next_renewable_gen = merged.at[t+1, "Renewable Generation (kW)"]
        else:
            next_load = merged.at[t, "Load (kW)"]
            next_renewable_gen = merged.at[t, "Renewable Generation (kW)"]

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

        # Calculate the maximum allowable geothermal output based on the ramp rate
        max_geo_output = prev_geo_output + MAX_GEOTHERMAL_STEP
        # Ensures that ramp does not exceed maximum generation levels
        max_geo_output = np.clip(max_geo_output, min=GEOTHERMAL_MIN_GENERATION, max=GEOTHERMAL_CAPACITY_KW) 

        # Calculate the minimum allowable geothermal output based on the ramp rate
        min_geo_output = prev_geo_output - MAX_GEOTHERMAL_STEP
        # Ensures that ramp does not exceed minimum generation levels
        min_geo_output = np.clip(min_geo_output, min=GEOTHERMAL_MIN_GENERATION, max=GEOTHERMAL_CAPACITY_KW) 

        
        load_to_meet = load - min_geo_output
        
        # Dispatch logic
        if load_to_meet < 0:
            # load < min_geo_output
            # The load is less than the minimum geothermal output, and the excess energy must
            # be stored in the battery (subject to battery charge limitations)
            # The battery may need to charge from any excess PV here too, but we assume we can
            # curtail that instantly to zero if we need to
            excess_generation = min_geo_output + renewable_generation - load
            charge_power = min(excess_generation, max_charge_power)
            battery_power_target = -1*charge_power # Charging is negative!
            soc_step = charge_power * CHARGING_EFFICIENCY * timestep / BATTERY_ENERGY_CAPACITY # SOC should increase
            # The geothermal should be ramped down as quickly as possible
            geothermal_output = min_geo_output

        # The load exceeds the minimum geothermal output – so we know we need at least that 
        # much generation and possibly more.
        # load > min_geo_output
        # Is load > maximum generation?
        elif (renewable_generation + max_discharge_power + max_geo_output) <= load:
            # We need everything we can get
            # Ramp up the geothermal as much as possible
            # Set the battery to discharge as much as possible
            discharge_power = max_discharge_power
            battery_power_target = discharge_power # Discharging is positive!
            soc_step = -1 * discharge_power * DISCHARGING_EFFICIENCY * timestep / BATTERY_ENERGY_CAPACITY # SOC should decrease
            geothermal_output = max_geo_output
        else:
            # min_geo_output < load < (max_geo_output + max_discharge_power + renewable_generation)
            # 0 < load_to_meet < [(max_geo_output - min_geo_output) + max_discharge_power + renewable_generation]
            
            # The load is more than the minimum geothermal output, but less than the sum of our
            # available generation, so now the question becomes:
            # How best should the load be met?              
            
            # Can we meet the load using only PV and the minimum geothermal output?
            # load < min_geo_output + renewable_generation ? 
            # load - min_geo_output < renewable_generation ? 
            # load_to_meet < renewable_generation ? 
            if load_to_meet <= renewable_generation:    
                # There is enough PV generation to meet the load
                # Set the geothermal output to the minimum
                # Charge the battery with any excess.
                # If the battery cannot charge that's fine – we'll curtail the PV instantly
                geothermal_output = min_geo_output
                excess_generation = renewable_generation + min_geo_output - load
                charge_power = min(excess_generation, max_charge_power)
                battery_power_target = -1*charge_power # Charging is negative!
                soc_step = charge_power * CHARGING_EFFICIENCY * timestep / BATTERY_ENERGY_CAPACITY # SOC should increase
                # The geothermal should be ramped down as quickly as possible
                geothermal_output = min_geo_output
            else:
                # load_to_meet > renewable_generation
                # load - min_geo_output > renewable_generation
                # load > min_geo_output + renewable_generation 
                
                # We need more than just the PV and the minimum geothermal output
                # Do we have available battery power to meet the deficit?
                # If the deficit can be met by the maximum battery output, we'll continue using
                # the minimum geothermal output

                # load < [min_geo_output + max_discharge_power + renewable_generation] ? 
                # load - renewable_generation - min_geo_output < max_discharge_power ? 
                # load - (renewable_generation + min_geo_output) < max_discharge_power ?
                deficit = load - (renewable_generation + min_geo_output)
                if deficit < max_discharge_power:
                    # Set the geothermal to minimum generation
                    geothermal_output = min_geo_output
                    # Set the battery to max discharge
                    discharge_power = max_discharge_power
                    battery_power_target = discharge_power # Discharging is positive!
                    soc_step = -1 * discharge_power * DISCHARGING_EFFICIENCY * timestep / BATTERY_ENERGY_CAPACITY # SOC should decrease
                    predicted_soc = np.clip((prev_soc + soc_step), SOC_MIN, SOC_MAX)
                    predicted_max_discharge_power = min(BATTERY_MAX_DISCHARGE_POWER, 
                                  (predicted_soc - SOC_MIN) * BATTERY_ENERGY_CAPACITY / 
                                  (DISCHARGING_EFFICIENCY * timestep))
                    # # If the load at the next timestep is not going to be able to be met, we need 
                    # # to start ramping up geothermal now
                    # if next_load > (next_renewable_gen + max_geo_output + predicted_max_discharge_power):
                    #     print("Ramping down battery to prepare for geothermal ramp!")
                    #     geothermal_output = prev_geo_output + MAX_GEOTHERMAL_STEP
                    #     geothermal_output = np.clip(geothermal_output, min=GEOTHERMAL_MIN_GENERATION, max=GEOTHERMAL_CAPACITY_KW) 
                    #     discharge_power = load - (renewable_generation + geothermal_output)
                    #     discharge_power = np.clip(discharge_power, 0, max_discharge_power)
                    #     battery_power_target = discharge_power # Discharging is positive!
                    #     soc_step = -1 * discharge_power * DISCHARGING_EFFICIENCY * timestep / BATTERY_ENERGY_CAPACITY # SOC should decrease   
                    if (battery_power_target > 0) and predicted_soc <= 45:
                        print("Ramping down battery to prepare for geothermal ramp!")
                        geothermal_output = prev_geo_output + 0.8 * MAX_GEOTHERMAL_STEP
                        geothermal_output = min(geothermal_output, load)
                        geothermal_output = np.clip(geothermal_output, min=GEOTHERMAL_MIN_GENERATION, max=GEOTHERMAL_CAPACITY_KW) 
                        discharge_power = load - (renewable_generation + geothermal_output)
                        discharge_power = np.clip(discharge_power, 0, max_discharge_power)
                        battery_power_target = discharge_power # Discharging is positive!
                        soc_step
                else:
                    # We do not have sufficient battery power to meet the load, so we should 
                    # use geothermal too

                    # deficit = load - (renewable_generation + min_geo_output) > max_discharge_power

                    # Reorganized: load > min_geo_output + max_discharge_power + renewable_generation
                    
                    # But we know: load < (max_geo_output + max_discharge_power + renewable_generation)
                    
                    # So: (min_geo_output + max_discharge_power + renewable_generation) < load < (max_geo_output + max_discharge_power + renewable_generation)
                    
                    # Simplify: min_geo_output < load - (max_discharge_power + renewable_generation) < max_geo_output

                    # So it's fine to set the geothermal output to load - (max_discharge_power + renewable_generation)

                    # Set the battery to max discharge
                    discharge_power = max_discharge_power
                    battery_power_target = discharge_power # Discharging is positive!
                    soc_step = -1 * discharge_power * DISCHARGING_EFFICIENCY * timestep / BATTERY_ENERGY_CAPACITY # SOC should decrease
                    
                    # Set the geothermal to make up the difference
                    geothermal_output = load - (max_discharge_power + renewable_generation)

        # Clip SOC to its limits
        new_soc = np.clip((prev_soc + soc_step), SOC_MIN, SOC_MAX)

        # Update values in array for next iteration
        merged.at[t, "SOC"] = new_soc
        merged.at[t, "Battery Power Target (kW)"] = battery_power_target
        merged.at[t, "Expected Geothermal Output (kW)"] = geothermal_output

    return merged

def get_battery_utilization(result, module_dict):
        # From initial analysis, the battery is highly underutilized:
        # Grab discharging targets from the battery dispatch time series
        result['Battery Discharge Target (kW)'] = result['Battery Power Target (kW)'][result['Battery Power Target (kW)'] > 0]

        # Calculate energy discharged in year 1 based on these targets
        result['Battery Energy Discharged (kWh)'] = result['Battery Discharge Target (kW)'] * (5/60)
        year_1_mask = result.index < pd.Timestamp(f'2013-01-01 00:00:00')
        year_1_battery_generation_kwh = result.loc[year_1_mask, 'Battery Energy Discharged (kWh)'].sum()

        # We need a baseline against which to evaluate the utilization of the battery. We will choose an 
        # 'ideal' dispatch of 100% of the battery's capacity per day as this baseline.
        battery_capacity_kWhdc = module_dict['pvbatt_model'].BatterySystem.batt_computed_bank_capacity
        ideal_annual_battery_generation_kwh = battery_capacity_kWhdc * 365

        # Compare actual energy discharged in year 1 against ideal energy discharged in year 1
        battery_capacity_factor = year_1_battery_generation_kwh / ideal_annual_battery_generation_kwh
        print(f"Battery utilization is {battery_capacity_factor * 100:.2f}% of the ideal utilization.")

# %% Execute PySAM model
module_dict = run_pysam_model(inputs_file='data/test_cases/remove_wind/Remove_Wind')

# %% Define some system characteristics that the dispatch model requires
# Battery stuff from SAM
BATTERY_ENERGY_CAPACITY = module_dict['pvbatt_model'].BatterySystem.batt_computed_bank_capacity
BATTERY_MAX_DISCHARGE_POWER = module_dict['pvbatt_model'].BatterySystem.batt_power_discharge_max_kwdc
BATTERY_MAX_CHARGE_POWER = module_dict['pvbatt_model'].BatterySystem.batt_power_charge_max_kwdc
SOC_MIN = module_dict['pvbatt_model'].BatteryCell.batt_minimum_SOC
SOC_MAX = module_dict['pvbatt_model'].BatteryCell.batt_maximum_SOC
INITIAL_SOC = module_dict['pvbatt_model'].BatteryCell.batt_initial_SOC
CHARGING_EFFICIENCY = module_dict['pvbatt_model'].BatterySystem.batt_dc_dc_efficiency
DISCHARGING_EFFICIENCY = module_dict['pvbatt_model'].BatterySystem.batt_dc_ac_efficiency
#ROUNDTRIP_EFFICIENCY = m.battery.BatterySystem.batt_ac_dc_efficiency * m.battery.BatterySystem.batt_dc_ac_efficiency

# Iteration stuff
timestep = 5/60 # 5-minute timestep in hours

# %% Prepare dataset
load = produce_load_dataframe(load_filepath='data/Project 2 - Load Profile.xlsx')
gen = produce_generation_dataframe(module_dict=module_dict)

#%% Merge the two dataframes
merged = pd.merge(gen, load, on='Datetime', how='inner')

# %% Run dispatch model
result = battery_dispatch_model_with_ramp_limits(merged)
result.set_index('Datetime', inplace=True)

# %% Report on whether the ramp rate was obeyed
result['Geothermal Steps (kW)'] = result['Expected Geothermal Output (kW)'].diff()
result['Violates Geothermal Ramp Rate'] = result['Geothermal Steps (kW)'].abs() > MAX_GEOTHERMAL_STEP

has_exceeded = result['Violates Geothermal Ramp Rate'].any()

if has_exceeded:
    print("There are differences exceeding the threshold.")
     # Filter rows where the condition is True
    exceeded_days = result.loc[result['Violates Geothermal Ramp Rate']].index.unique()
    
    print("The geothermal ramp rate was exceeded on these days:")
    for day in exceeded_days:
        print(day)
else:
    print("No differences exceed the threshold.")


# %% Generate some plots
date_start = '2012-01-31 00:00:00'
date_end = '2012-02-01 00:00:00'
pysam_helpers.plot_values_by_time_range(df=result, start_time=date_start, end_time=date_end, y_columns=['PV Generation (kW)', 'Wind Generation (kW)', 'Load (kW)', 'Battery Power Target (kW)', 'Expected Geothermal Output (kW)'])
pysam_helpers.plot_values_by_time_range(df=result, start_time=date_start, end_time=date_end, y_columns=['SOC'])

# %% Print the battery utilization
get_battery_utilization(result, m)

# %% Generate a csv with the dispatch target
result['Battery Power Target (kW)'].to_csv("data/test_cases/remove_wind/dispatch_target_5min.csv", index=False)

# %% Generate a csv of system power output without the battery
#result.to_csv('data/PySAM_Outputs/baseline_system_output.csv')
