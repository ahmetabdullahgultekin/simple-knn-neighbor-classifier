# Ahmet Abdullah Gültekin 150121025
import json

import pandas as pd

# Constant for the target column
TARGET_COLUMN = 'PlayTennis'
# Global variable for the K value and distance metric
k_value = -1
# 0 for Euclidean Distance, 1 for Manhattan Distance
distance_metric = -1


# Load and prepare your dataset
def load_data(file_path):
    data = pd.read_json(file_path)
    print('---------------------------------------------------------\n')
    print('Dataset:')
    print(data)
    print('\n---------------------------------------------------------')
    summarize_data(data)
    return data


# Summarize the dataset
def summarize_data(data):
    print('\nUnique values in each column:\n')
    for column in data.columns:
        print(f'{column}: {data[column].unique()}')
    print('\n---------------------------------------------------------')


# Summarize the occurrences of each value in the target column
def summarize_occurrences(data):
    target_value_counts = data[TARGET_COLUMN]
    # clear content of json
    with open('values.json', 'a') as json_file:
        json_file.seek(0)
        json_file.truncate()
    add_data_to_json(data, target_value_counts)
    # print_values_from_json()
    # print_values_from_memory(target_value_counts, data)


# Write the datas to the json file
def add_data_to_json(data, target_value_counts):
    # Open the json file in append mode
    with open('values.json', 'a') as values_file:
        values_file.write('{')
        values_file.write('"Values": [')
        values_file.write(target_value_counts.value_counts().to_json() + ',')
        values_file.write(target_value_counts.value_counts(normalize=True).to_json() + ',')
        for column in data.drop(columns=[TARGET_COLUMN]).columns:
            values_file.write(data.groupby([column, TARGET_COLUMN]).size().to_json() + ',')
            # Do not add comma at the end
            if column == data.drop(columns=[TARGET_COLUMN]).columns[-1]:
                values_file.write(
                    (data.groupby([column, TARGET_COLUMN]).size() / data.groupby(TARGET_COLUMN).size()).to_json())
            else:
                values_file.write(
                    (data.groupby([column, TARGET_COLUMN]).size() / data.groupby(TARGET_COLUMN).size()).to_json() + ',')
        values_file.write(']')
        values_file.write('}')


# Update the json file with the new values of one-hot encoded dataset
def generate_encoded_json_file(data):
    # Open the json file in append mode
    with open('encoded_values.json', 'w') as values_file:
        encode_one_hot(data)
        # Write the new dataset to the json file
        values_file.write(data.to_json())


# Print the values from the json file
def print_values_from_json():
    with open('values.json', 'r') as values_file:
        values = json.load(values_file)
        print(values)


# Print the values from the memory
def print_values_from_memory(target_value_counts, data):
    print('Occurrences of each value in the target column:')
    print(target_value_counts.value_counts())
    print('\n---------------------------------------------------------')
    print('Occurrences of each value in the target column (normalized):')
    print(target_value_counts.value_counts(normalize=True))
    print('\n---------------------------------------------------------')

    # Summarize the occurrences of each value for Yes and No
    for column in data.drop(columns=[TARGET_COLUMN]).columns:
        print(f'Occurrences of each value in the {column} column:')
        print(data.groupby([column, TARGET_COLUMN]).size())
        print('\n---------------------------------------------------------')
        print(f'Occurrences of each value in the {column} column (likelihoods):')
        print(data.groupby([column, TARGET_COLUMN]).size() / data.groupby(TARGET_COLUMN).size())
        print('\n---------------------------------------------------------')


# Calculate the distance between the new instance and the current row
def calculate_distance(instance, data, target_column):
    # Calculate the distances of each row
    distances = []
    distance = 0
    j = 0
    # Iterate each row of the target column
    for target in data[target_column]:
        # Calculate the distance of the new instance and the current row
        print(f"Value: {target} - target_column: {target_column}")
        for value in instance:
            if value in data.columns:
                print(f"Value: {value} is in the columns.")
                if data[value][j] == 1:
                    distance += 1
                    print(f"Distance: {distance}")
        j += 1
        distances.append(distance)
        distance = 0
    # print(f"Distances: {distances}")
    return distances


# Train the KNN model
def train_knn_model():
    # Hash values with their indices
    instance_distances = calculate_distance(new_prediction_data, data_set, TARGET_COLUMN)
    instance_distances_with_indices = [(index, distance) for index, distance in enumerate(instance_distances)]
    print("---------------------------------------------------------")
    print("Training the KNN model...")
    print("---------------------------------------------------------")
    print("These are the distance points of the new instance for each row:")
    print("HIGH POINTS ARE CLOSEST TO THE NEW INSTANCE!")
    print(f"Distance Points: {instance_distances}")
    print(f"Distance Points with indices: {instance_distances_with_indices}")
    print("---------------------------------------------------------")
    # sort the distances in descending order
    instance_distances.sort(reverse=True)
    instance_distances_with_indices.sort(key=lambda x: x[1], reverse=True)
    print(f"Sorted Distance Points: {instance_distances}")
    print(f"Sorted Distance Points with indices: {instance_distances_with_indices}")
    neighbors_with_target_values = [(data_set[TARGET_COLUMN][instance_distances_with_indices[i][0]]) for i in
                                    range(k_value)]
    print(f"Neighbors with target values: {neighbors_with_target_values}")
    for i in range(k_value):
        # Select the k nearest neighbors
        print(
            f"Index: {instance_distances_with_indices[i][0]} - Distance Point: {instance_distances_with_indices[i][1]}")

    # Count the occurrences of each class
    occurrences = count_the_occurrences(neighbors_with_target_values)
    # Return the KNN model
    return occurrences


# List the classes of the new instance
def count_the_occurrences(target_values):
    # Count target occurrences
    target_unique = data_set[TARGET_COLUMN].unique()
    print(f"Unique targets: {target_unique}")
    target_occurrences = {}
    for target in target_unique:
        target_occurrences[target] = target_values.count(target)
    print(f"Target occurrences: {target_occurrences}")
    return target_occurrences


# Print the results
def print_results(new_values):
    print("---------------------------------------------------------")
    print(f"Result of the new instance: {new_values}")
    # Make the prediction
    if new_values[1] > new_values[0]:
        print("Comparison: Yes > No ")
        print("The new instance is classified as 'Yes'.")
    else:
        print("Comparison: No > Yes")
        print("The new instance is classified as 'No'.")


# Ask parameters from the user
def assign_parameters():
    global k_value, distance_metric
    # K is a hyperparameter that you need to get from user.
    # Ask user to enter the value of K.
    while k_value < 1 or k_value > len(data_set):
        try:
            k_value = int(input("Enter the value of K: "))
        except ValueError:
            print("Invalid value for K.")
        if k_value < 1:
            print("The value of K must be greater than 0.")

    print(f"K is selected as {k_value}.")
    # As distance metric you can use Euclidean Distance or Manhattan Distance. Ask user which one to use.
    # 0 for Euclidean Distance, 1 for Manhattan Distance
    while distance_metric != 0 and distance_metric != 1:
        try:
            distance_metric = int(
                input("Enter the distance metric (0 for Euclidean Distance, 1 for Manhattan Distance): "))
        except ValueError:
            print("Invalid value for distance metric.")
        if distance_metric == 0:
            print("Euclidean Distance is selected.")
        elif distance_metric == 1:
            print("Manhattan Distance is selected.")
        else:
            print("Invalid distance metric.")


def encode_one_hot(data):
    # One-hot encoding
    for column in data.columns:
        unique_values = data[column].unique()  # Get unique values in the column
        print(f'Unique values in the {column} column:')
        print(unique_values)
        for value in unique_values:
            # Create a new column for each unique value
            data[f"{column}_{value}"] = (data[column] == value).astype(int)
            print(f"Column {column}_{value} is created.")
    print('---------------------------------------------------------\n')
    print('One-hot encoded dataset:')
    print(data)
    print('\n---------------------------------------------------------')
    return data


# Encode the new instance
def encode_new_instance(new_instance):
    new_instance_encoded = {}
    for key, value in new_instance.items():
        new_instance_encoded[f"{key}_{value}"] = 1
    return new_instance_encoded


def predict_result():
    predict_the_class = max(knn_model, key=knn_model.get)
    if knn_model[predict_the_class] == 0:
        print("---------------------------------------------------------")
        print("Prediction cannot be made.")
        print("No class has any occurrences.")
        print("---------------------------------------------------------")
        return
    # Check the equality of the values
    if knn_model['Yes'] == knn_model['No']:
        print("---------------------------------------------------------")
        print("Prediction cannot be made.")
        print("Both classes have the same number of occurrences.")
        print("---------------------------------------------------------")
        return
    print("---------------------------------------------------------")
    print(f"Predicted class of the new instance: {TARGET_COLUMN} = {predict_the_class}")
    print("---------------------------------------------------------")
    print("Prediction completed.")
    print("---------------------------------------------------------")


# Start the program
if __name__ == "__main__":
    # Enter the new instance
    new_prediction_data = {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Strong'}
    # Encode the new instance
    new_prediction_data = encode_new_instance(new_prediction_data)
    print('Encoded new instance:')
    print(new_prediction_data)
    # Load the dataset
    data_set = load_data('data_set_play_tennis.json')
    generate_encoded_json_file(data_set)
    # Summarize the dataset
    summarize_occurrences(data_set)
    # Ask parameters from the user
    assign_parameters()
    # Train the KNN model
    knn_model = train_knn_model()
    # Predict the class of the new instance
    predict_result()
