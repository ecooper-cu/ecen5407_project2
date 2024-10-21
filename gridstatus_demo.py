# %%
import gridstatus
import pandas as pd
import plotly.express as px
import pickle
# %%
# Access CAISO data
caiso = gridstatus.CAISO()
# %%
# Historical Locational Marginal Pricing (LMP)

# Define time window
start = pd.Timestamp("Jan 1, 2022").normalize()
end = pd.Timestamp("Dec 31, 2022").normalize()

# Three locations in downtown San Diego
locations = ["SAMPSON_6_N010", "CORONA_1_N001", "NOISLMTR_6_N101"]

# Get pickled data if it exists to save time
pickle_name = f"data/pickled_dataframes/LMP_{start.strftime('%Y_%m_%d')}_to_{end.strftime('%Y_%m_%d')}.pkl"
try:
    with open(pickle_name, 'rb') as f:
        lmp_df = pickle.load(f)
    print("DataFrame loaded from pickle.")
except FileNotFoundError:
    lmp_df = pd.DataFrame()
    print("No pickle found.")

if lmp_df.empty:
    # Load DataFrame
    lmp_df = caiso.get_lmp(
        start=start,
        end=end,
        market="DAY_AHEAD_HOURLY",
        locations=locations,
        sleep=5
    )
else:
    # Check if all the desired locations are present in the existing DataFrame
    existing_locations = list(lmp_df['Location'].unique())
    missing_locations = [loc for loc in locations if loc not in existing_locations]

    if missing_locations:
        # Pull missing locations
        print(f"Pulling data for missing locations: {missing_locations}")
        new_lmp_df = caiso.get_lmp(
            start=start,
            end=end,
            market="DAY_AHEAD_HOURLY",
            locations=missing_locations,
            sleep=5
        )

        # Add missing locations into existing DataFrame
        lmp_df = pd.concat([lmp_df, new_lmp_df], ignore_index=True)
    else:
        print("All locations are already present in the DataFrame.")

# Pickle the DataFrame for future use
with open(pickle_name, 'wb') as f:
    pickle.dump(lmp_df, f)

print(lmp_df.head())
# %%
# Show negative LMPs
negative_lmps = lmp_df[lmp_df["LMP"] < 0].set_index("Time")
negative_per_month = (
    negative_lmps.groupby("Location").resample("MS")["LMP"].count().reset_index()
)
fig = px.bar(
    negative_per_month,
    x="Time",
    y="LMP",
    title="Negative LMPs per Month - Downtown San Diego Hubs",
    color="Location"
)
fig.update_yaxes(title="Number of Negative LMPs")
fig.show("svg", width=1200, height=600)
# %%
