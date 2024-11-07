#Here i hav done loading and Preprocessing Techniques


import pandas as pd

# Load the datasets
vehicle_data = pd.read_csv("/kaggle/input/vehicular-movements-datasets/vehicle path (Day).csv")
geo_data = pd.read_csv("/kaggle/input/vehicular-movements-datasets/geo1.csv")

# Display the first few rows of each dataset to understand their structure
print("Vehicle Path Data:")
print(vehicle_data.head())

print("\nGeo1 Data:")
print(geo_data.head())

# Display basic info and statistics
print("\nVehicle Path Data Info:")
print(vehicle_data.info())

print("\nGeo1 Data Info:")
print(geo_data.info())

print("\nVehicle Path Data Statistics:")
print(vehicle_data.describe())

print("\nGeo1 Data Statistics:")
print(geo_data.describe())



# Check for missing values
print("\nMissing values in Vehicle Path Data:")
print(vehicle_data.isnull().sum())

print("\nMissing values in Geo1 Data:")
print(geo_data.isnull().sum())

# Drop or fill missing values as needed
vehicle_data = vehicle_data.dropna()  # or fillna() depending on the dataset characteristics
geo_data = geo_data.dropna()  # or fillna()

# Ensure correct data types
vehicle_data['timestep_time'] = pd.to_datetime(vehicle_data['timestep_time'])
# Similarly, convert other columns to their appropriate types if necessary

# Preview the cleaned data
print("\nCleaned Vehicle Path Data:")
print(vehicle_data.head())

print("\nCleaned Geo1 Data:")
print(geo_data.head())
