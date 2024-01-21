import numpy as np
import tensorflow as tf

# Load the saved model
loaded_model = tf.keras.models.load_model('wind_energy_model.h5')

# Define the feature names for user input
feature_names = ['soil_temp', 'temp', 'pressure', 'wind_speed']

# Initialize an empty list to store user input values
user_inputs = []

# Get user input for each feature
for feature_name in feature_names:
    user_input = input(f"Enter {feature_name}: ")
    try:
        # Convert user input to a float
        user_input = float(user_input)
        user_inputs.append(user_input)
    except ValueError:
        print(f"Invalid input for {feature_name}. Please enter a numeric value.")
        exit(1)

# Convert user inputs to a NumPy array
user_inputs = np.array(user_inputs).reshape(1, -1)  # Reshape to match model input shape

# Make predictions using the loaded model
predicted_wind_energy = loaded_model.predict(user_inputs)

# Display the predicted wind energy
print(f'Predicted Wind Energy: {predicted_wind_energy[0][0]}')
