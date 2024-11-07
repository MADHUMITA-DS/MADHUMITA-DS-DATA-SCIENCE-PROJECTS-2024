import matplotlib.pyplot as plt

# Plot original vehicle paths (before matching)
plt.scatter(vehicle_data['longitude'], vehicle_data['latitude'], color='blue', label='Vehicle Path')

# Plot map-matched paths
plt.scatter(vehicle_data['road_lon'], vehicle_data['road_lat'], color='red', label='Map-Matched Path')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Vehicle Path vs. Map-Matched Path')
plt.legend()
plt.show()
