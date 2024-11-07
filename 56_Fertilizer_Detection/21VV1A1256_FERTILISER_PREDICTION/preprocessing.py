# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv1D, Flatten, Dense, MaxPooling1D, Dropout

# # Load the dataset
# data = pd.read_csv("E:\customProjects\cropYield\Fertilizer_Prediction.csv")
# # "E:\customProjects\cropYield\Fertilizer_Prediction.csv"
# # Inspect the dataset
# print(data.head())

# # Encode the 'Fertilizer_Name' column to numerical values
# label_encoder = LabelEncoder()
# data['Fertilizer_Code'] = label_encoder.fit_transform(data['Fertilizer_Name'])

# # Separate features and target
# X = data[['Nitrogen', 'Potassium', 'Phosphorous']]
# y = data['Fertilizer_Code']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Reshape the input data (add a dummy dimension for Conv1D)
# X_train = np.expand_dims(X_train.values, axis=-1)
# X_test = np.expand_dims(X_test.values, axis=-1)

# # Build the CNN model
# model = Sequential([
#     Conv1D(32, kernel_size=2, activation='relu', input_shape=(3, 1)),
#     MaxPooling1D(pool_size=2),
#     Conv1D(64, kernel_size=2, activation='relu'),
#     MaxPooling1D(pool_size=2),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(len(label_encoder.classes_), activation='softmax')  # Output layer
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, MaxPooling1D, Dropout

# Load the dataset
# data = pd.read_csv(r"E:\customProjects\cropYield\Fertilizer_Prediction.csv")
# data = pd.read_csv(r"/mnt/Fertilizer_Prediction.csv")
data = pd.read_csv(r"Fertilizer_Prediction.csv")

# Encode the 'Fertilizer_Name' column to numerical values
label_encoder = LabelEncoder()
data['Fertilizer_Code'] = label_encoder.fit_transform(data['Fertilizer_Name'])

# Separate features and target
X = data[['Nitrogen', 'Potassium', 'Phosphorous']]
y = data['Fertilizer_Code']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the input data (add a dummy dimension for Conv1D)
X_train = np.expand_dims(X_train.values, axis=-1)  # Shape: (samples, 3, 1)
X_test = np.expand_dims(X_test.values, axis=-1)

# Build the CNN model with 'same' padding to avoid dimension collapse
model = Sequential([
    Conv1D(32, kernel_size=2, activation='relu', padding='same', input_shape=(3, 1)),
    MaxPooling1D(pool_size=2, padding='same'),
    Conv1D(64, kernel_size=2, activation='relu', padding='same'),
    MaxPooling1D(pool_size=2, padding='same'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Re-save the model to ensure it's not corrupted
model.save("fertilizer_model.h5", save_format='h5')

