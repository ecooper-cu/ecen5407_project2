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

def battery_dispatch_model(merged):
        """
        Dispatch model for lithium-ion battery.

        Parameters:
                merged (pd.DataFrame): DataFrame with columns "Datetime", "PV Generation (kW)",
                                       "Wind Generation (kW)", "Load (kW)"
        
        Returns:
                pd.DataFrame: Dataframe with two additional columns:
                                "Battery Power Target (kW)": Battery power target in kW (negative
                                        for charging, positive for discharging)
                                "SOC": Battery state of charge 
        """

        # Calculate total renewable generation
        merged["Renewable Generation (kW)"] = merged["PV Generation (kW)"] + merged["Wind Generation (kW)"]

        # Initialize SOC (state of charge)
        merged["SOC"] = INITIAL_SOC  # Start at 50% SOC

        # Calculate power excess/deficit
        merged["Excess Power (kW)"] = merged["Renewable Generation (kW)"] - merged["Load (kW)"]
        merged["Deficit Power (kW)"] = merged["Load (kW)"] - merged["Renewable Generation (kW)"]

        # Masks for different conditions
        load_below_geo = merged["Load (kW)"] <= GEOTHERMAL_CAPACITY
        load_above_geo = ~load_below_geo

        # Battery dispatch decisions – these steps take into account power limits on the battery as
        # a result of the SOC. We're ignoring this for now because:
        # 1) In practice, the battery would be limited much sooner than this charge limit would 
        # kick in. Especially for charging, it's not possible to run at nameplate power right up to 
        # 100% SOC (or even 95% SOC), because the battery has to taper with CV charging. SAM will 
        # take care of this derating.
        # 2) This isn't implemented correctly as is, and we don't think it's worth it at the moment
        #charge_limit = (SOC_MAX - merged["SOC"]) * BATTERY_ENERGY_CAPACITY / (ROUNDTRIP_EFFICIENCY * DT)
        #discharge_limit = (merged["SOC"] - SOC_MIN) * BATTERY_ENERGY_CAPACITY / DT

        # Initialize battery target
        merged["Battery Power Target (kW)"] = 0.0

        # Case 1: Load ≤ geothermal capacity
        mask_charge = load_below_geo & (merged["Excess Power (kW)"] >= 0)
        merged.loc[mask_charge, "Battery Power Target (kW)"] = -np.minimum(
                merged.loc[mask_charge, "Excess Power (kW)"], BATTERY_MAX_CHARGE_POWER)
        # merged.loc[mask_charge, "Battery Target (kW)"] = (
        #         np.minimum(merged.loc[mask_charge, "Excess Power (kW)"], 
        #                 np.minimum(BATTERY_POWER_NAMEPLATE, charge_limit[mask_charge]))
        # )
        
        mask_discharge = load_below_geo & (merged["Excess Power (kW)"] < 0) & (merged["SOC"] > SOC_MIN)
        merged.loc[mask_discharge, "Battery Power Target (kW)"] = np.minimum(
                -merged.loc[mask_discharge, "Deficit Power (kW)"], BATTERY_MAX_DISCHARGE_POWER)
        # merged.loc[mask_discharge, "Battery Target (kW)"] = (
        #         -np.minimum(merged.loc[mask_discharge, "Deficit Power (kW)"], 
        #                 np.minimum(BATTERY_POWER_NAMEPLATE, discharge_limit[mask_discharge]))
        # )

        # Case 2: Load > geothermal capacity
        adjusted_deficit = merged["Load (kW)"] - GEOTHERMAL_CAPACITY
        mask_charge_high_load = load_above_geo & (merged["Excess Power (kW)"] > 0)
        merged.loc[mask_charge_high_load, "Battery Power Target (kW)"] = -np.minimum(
                merged.loc[mask_charge_high_load, "Excess Power (kW)"], -BATTERY_MAX_CHARGE_POWER)
        # merged.loc[mask_charge_high_load, "Battery Target (kW)"] = (
        #         np.minimum(merged.loc[mask_charge_high_load, "Excess Power (kW)"], 
        #                 np.minimum(BATTERY_POWER_NAMEPLATE, charge_limit[mask_charge_high_load]))
        # )
        
        mask_discharge_high_load = load_above_geo & (adjusted_deficit > 0) & (merged["SOC"] > SOC_MIN)
        merged.loc[mask_discharge_high_load, "Battery Power Target (kW)"] = np.minimum(
                merged.loc[mask_discharge_high_load, "Excess Power (kW)"], 
                BATTERY_MAX_DISCHARGE_POWER)
        # merged.loc[mask_discharge_high_load, "Battery Target (kW)"] = (
        #         -np.minimum(adjusted_deficit[mask_discharge_high_load], 
        #                 np.minimum(BATTERY_POWER_NAMEPLATE, discharge_limit[mask_discharge_high_load]))
        # )

        # Update SOC
        # merged["SOC"] += (
        #         merged["Battery Target (kW)"] * (ROUNDTRIP_EFFICIENCY ** (merged["Battery Target (kW)"] > 0)) * DT / BATTERY_ENERGY_CAPACITY
        # )
        # merged["SOC"] = np.clip(merged["SOC"], SOC_MIN, SOC_MAX)  # Ensure SOC stays within bounds

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

# Iteration stuff
timestep = 5/60 # 5-minute timestep in hours

# %% Prepare dataset
load = produce_load_dataframe(load_filepath='data/Project 2 - Load Profile.xlsx')
gen = produce_generation_dataframe(m=m)

# Merge the two dataframes
merged = pd.merge(gen, load, on='Datetime', how='inner')

# %% Run dispatch model
result = battery_dispatch_model(merged)

# %% Generate some plots
date_start = '2012-12-21 00:00:00'
date_end = '2012-12-22 00:00:00'
#result.set_index('Datetime', inplace=True)
pysam_helpers.plot_values_by_time_range(df=result, start_time=date_start, end_time=date_end, y_columns=['PV Generation (kW)', 'Wind Generation (kW)', 'Load (kW)', 'Battery Power Target (kW)'])

# %% Print the battery utilization
get_battery_utilization(result, m)

# %% Generate a csv with the dispatch target
#result['Battery Power Target (kW)'].to_csv('data/PySam_Inputs/Hybrid_Project/dispatch_target_5min.csv', index=False)

# %% Generate a csv of system power output without the battery
#result.to_csv('data/PySAM_Outputs/baseline_system_output.csv')
