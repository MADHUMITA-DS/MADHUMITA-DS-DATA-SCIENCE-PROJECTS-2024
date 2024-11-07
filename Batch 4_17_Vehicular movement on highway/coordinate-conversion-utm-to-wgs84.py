from pyproj import Transformer

# Initialize the Transformer for UTM Zone 14N to WGS84 (latitude/longitude)
transformer = Transformer.from_crs("epsg:32614", "epsg:4326", always_xy=True)

# Function to convert x-y to lat-long using the new Transformer class
def convert_xy_to_latlong(x, y):
    longitude, latitude = transformer.transform(x, y)
    return latitude, longitude

# Apply the conversion to the vehicle data
vehicle_data['latitude'], vehicle_data['longitude'] = zip(*vehicle_data.apply(
    lambda row: convert_xy_to_latlong(row['vehicle_x'], row['vehicle_y']), axis=1))

# Display the updated vehicle data with lat-long
print(vehicle_data[['vehicle_x', 'vehicle_y', 'latitude', 'longitude']].head())
