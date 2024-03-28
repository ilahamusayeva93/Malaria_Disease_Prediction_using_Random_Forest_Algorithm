# Malaria_Disease_Prediction_using_Random_Forest_Algorithm
## Introduction
This project focuses on analyzing clinical data related to Malaria to identify patterns, correlations, and predict clinical diagnoses based on various test results. By employing machine learning techniques, specifically the RandomForestClassifier, we aim to assist in the early detection and classification of Malaria cases.

## Setup
Dependencies
This project requires the following Python libraries:

numpy
scipy
pandas
matplotlib
seaborn
scikit-learn
Installation
Install the required libraries using pip:

bash
Copy code
pip install numpy scipy pandas matplotlib seaborn scikit-learn
## Data Preparation
The dataset, malaria_clinical_data.csv, contains clinical records of patients tested for Malaria. The following steps were taken to prepare the data for analysis:

Loading the Data: The dataset is loaded into a pandas DataFrame.

python
Copy code
import pandas as pd
data_file = "/path/to/malaria_clinical_data.csv"
dataframe = pd.read_csv(data_file)
Initial Exploration:

Shape of the DataFrame: dataframe.shape
First 10 records: dataframe.head(n=10)
Columns in the dataset: dataframe.columns
Information about data types and null values: dataframe.info()
Subset Selection: Focus on a subset of the data for analysis, specifically the columns from the 17th column onwards.

python
Copy code
subset = dataframe.iloc[:, 16:]
Handling Missing Values: Rows with missing values were dropped to ensure data quality.

python
Copy code
subset.dropna(inplace=True)
Categorical Encoding: The Clinical_Diagnosis column, which is categorical, was encoded into numerical labels for model training.

python
Copy code
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(subset['Clinical_Diagnosis'])
## Exploratory Data Analysis (EDA)
Clinical Diagnosis Distribution: Analysis of the distribution of clinical diagnoses.
Statistical Summary: Descriptive statistics for the numerical features.
Correlation Matrix: A heatmap visualizing the correlations between numerical features.
## Preprocessing
Feature Scaling: MinMaxScaler was applied to scale the features to a range between 0 and 1.
Data Splitting: The data was split into training and test sets, with 20% of the data reserved for testing.
## Model Training
A RandomForestClassifier was trained on the scaled training data.

## Evaluation
The model was evaluated using several metrics, including balanced accuracy, F1 score, recall, and a confusion matrix. These metrics provide insight into the model's performance across different classes.
