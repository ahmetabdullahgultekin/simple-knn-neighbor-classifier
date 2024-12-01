```markdown
# K-Nearest Neighbors (k-NN) Classifier

## Introduction

This project implements a K-Nearest Neighbors (k-NN) classifier from scratch. The classifier is applied to a dataset to
predict the class of a new instance based on the majority class of its k nearest neighbors.

## Prerequisites

- Python 3.x
- Pandas library
- JSON library

## Installation

1. Clone the repository or download the source code.
2. Ensure you have Python 3.x installed on your system.
3. Install the required libraries using pip:

```bash
pip install pandas
```

## Dataset

The dataset used in this project is stored in a JSON file named `data_set_play_tennis_updated.json`. Ensure this file is
in the same directory as the source code.

## Running the Code

1. Open a terminal or command prompt.
2. Navigate to the directory containing the source code.
3. Run the `main.py` file:

```bash
python main.py
```

4. Follow the prompts to enter the value of k and the distance metric (0 for Euclidean Distance, 1 for Manhattan
   Distance).
5. The program will output the predicted class for the new instance and update the dataset with the new instance.

## Files

- `main.py`: The main script that implements the k-NN classifier.
- `data_set_play_tennis_updated.json`: The dataset used for training and prediction.
- `values.json`: A JSON file used to store intermediate values.
- `encoded_values.json`: A JSON file used to store the one-hot encoded dataset.

## Example

Here is an example of how to run the code:

```bash
python main.py
```

You will be prompted to enter the value of k and the distance metric. After entering the values, the program will output
the predicted class for the new instance and update the dataset.

```

This `README.md` file provides instructions on how to set up and run the k-NN classifier code. It includes information 
on prerequisites, installation, running the code, and an example.

Ahmet Abdullah Gültekin