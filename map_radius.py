# %% Load dependencies
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Point
import matplotlib.pyplot as plt

# %% 
## Step 1: Load in the required data
# Load the shapefile
california_counties = gpd.read_file('data/ca_counties/CA_Counties.shp')

# Filter San Diego County
san_diego = california_counties[california_counties['NAME'] == 'San Diego']

# Ensure consistent coordinate system (using EPSG:3310 for California Albers projection)
california_counties = california_counties.to_crs(epsg=3310)
san_diego = san_diego.to_crs(epsg=3310)

## Step 2: Buffer the border of San Diego
# Create the buffer (convert miles to meters: 1 mile = 1609.34 meters)
buffer_distance = 50 * 1609.34
san_diego_buffer = san_diego.geometry.buffer(buffer_distance)

# Merge into a single geometry
buffer_union = unary_union(san_diego_buffer)

## Step 3: Overlay on top of Southern California
# Filter southern counties
southern_counties = california_counties[
    california_counties['NAME'].isin(['San Diego', 'Los Angeles', 'Orange', 'Riverside', 'San Bernardino', 'Imperial'])
]

# Intersect buffer with southern counties
within_50_miles = southern_counties[southern_counties.geometry.intersects(buffer_union)]

## Step 4: Add a point for the proposed geothermal plant
# Define the coordinates
coords = Point(-115.636810, 33.158393)

# Create a GeoDataFrame for Niland
plant_location = gpd.GeoDataFrame(
    [{'name': 'Geothermal_Plant', 'geometry': coords}],
    crs="EPSG:4326"  # Use WGS84 (latitude and longitude)
)

# Reproject to match other layers if needed
plant_location = plant_location.to_crs(epsg=3310)

#%%
## Step 5: Plot the map
# Plot southern counties and the buffer
fig, ax = plt.subplots(figsize=(10, 10))

southern_counties.plot(ax=ax, color='lightgrey', edgecolor='black')
san_diego.plot(ax=ax, color='blue', edgecolor='black')
gpd.GeoSeries(buffer_union).plot(ax=ax, color='blue', alpha=0.3)
within_50_miles.plot(ax=ax, color='red', alpha=0.5)
plant_location.plot(ax=ax, color='black', marker='o', label='Geothermal Plant', markersize=50)

plt.title("Areas within 50 miles of San Diego County Border")
plt.show()
# %%