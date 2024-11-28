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
            # values_file.write((data.groupby([column, TARGET_COLUMN]).size() / data.groupby(TARGET_COLUMN).size()).to_json() + ',')
            if column == data.drop(columns=[TARGET_COLUMN]).columns[-1]:
                values_file.write(
                    (data.groupby([column, TARGET_COLUMN]).size() / data.groupby(TARGET_COLUMN).size()).to_json())
            else:
                values_file.write(
                    (data.groupby([column, TARGET_COLUMN]).size() / data.groupby(TARGET_COLUMN).size()).to_json() + ',')
        values_file.write(']')
        values_file.write('}')


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
def calculate_distance(row, data, target_column):
    distance = 0
    for column in data.drop(columns=[target_column]).columns:
        if distance_metric == 0:
            distance += (row[column] - data[column]) ** 2
        elif distance_metric == 1:
            distance += abs(row[column] - data[column])
    return distance

# Train the KNN model
def train_knn_model(data, target_column):
    # KNN model
    knn_model = []
    # Iterate through the rows of the dataset
    for index, row in data.iterrows():
        # Calculate the distance between the new instance and the current row
        distance = calculate_distance(row, data, target_column)
        # Append the distance and the row to the KNN model
        knn_model.append((distance, row))
    # Sort the KNN model by distance
    knn_model.sort(key=lambda x: x[0])
    # Return the KNN model
    return knn_model


# List the classes of the new instance
def predict_the_result(new_instance, target_column):
    # total = P(Outlook = Sunny) * P(Temperature = Cool) * P(Humidity = High) * P(Wind = Strong) * P(PlayTennis = Yes)
    sum_of_yes = 1
    # total = P(Outlook = Sunny) * P(Temperature = Cool) * P(Humidity = High) * P(Wind = Strong) * P(PlayTennis = No)
    sum_of_no = 1
    # sum
    prediction_value = 1
    # Search the classes of the new instance in the json file
    with open('values.json', 'r') as values_file:
        values = json.load(values_file)
        for value in values['Values']:
            for key, instance in new_instance.items():
                # ('Sunny', 'No') is the format in the json file
                filter_string = f"('{instance}', '{target_column}')"
                if filter_string in value:
                    if 0 < value[filter_string] < 1:
                        print(f"key: {key} - value: {instance} - likelihood: {value[filter_string]}")
                        prediction_value *= value[filter_string]
                        print("---------------------------------------------------------")
                        print(f"Prediction value for {target_column}: {prediction_value}")
                        """if target_column == 'Yes':
                            sum_of_yes *= value[filter_string]
                            print("---------------------------------------------------------")
                            print(f"Sum of {target_column}: {sum_of_yes}")
                        else:
                            sum_of_no *= value[filter_string]
                            print("---------------------------------------------------------")
                            print(f"Sum of {target_column}: {sum_of_no}")"""

    # Get the values of prediction from the model belongs to the new instance
    # New values array
    # new_values = [sum_of_yes, sum_of_no]
    return prediction_value


# Print the results
def print_results(new_values):
    print("---------------------------------------------------------")
    print(f"Result of the new instance: {new_values}")
    # Make the prediction
    if new_prediction_classes[1] > new_prediction_classes[0]:
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
    while k_value < 1:
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


# Start the program
if __name__ == "__main__":
    # Enter the new instance
    new_prediction_data = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'}
    # Load the dataset
    data_set = load_data('data_set_play_tennis.json')
    # Summarize the dataset
    summarize_occurrences(data_set)
    # Ask parameters from the user
    assign_parameters()
    # Train the KNN model
    training_model = train_knn_model(data_set, TARGET_COLUMN)
    # Predict the result for each target column value
    new_prediction_classes = []
    for target_value in data_set[TARGET_COLUMN].unique():
        print("---------------------------------------------------------")
        print(f"Prediction for {target_value}:")
        print("---------------------------------------------------------")
        new_prediction_classes.append(predict_the_result(new_prediction_data, target_value))
    # Print the results
    print_results(new_prediction_classes)
