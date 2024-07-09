# Import necessary libraries and modules
from models import get_model  # Custom function to retrieve the model
import joblib  # For loading saved models
from data_handling import get_procesed_data  # Custom function to process the data
import pandas as pd  # For data manipulation and analysis
from sklearn.model_selection import KFold  # For k-fold cross-validation
from sklearn.metrics import accuracy_score, f1_score  # For performance metrics
import argparse  # For parsing command-line arguments
import warnings  # For controlling warnings
warnings.filterwarnings("ignore")  # Ignore warnings

# Argument parser to take command-line arguments
parser = argparse.ArgumentParser(description='Cluepoints Case 1: Length of Stay in Hospital Classification Test - Dhruv Aditya Mittal')
parser.add_argument('--fold', default=1, type=int, dest='fold', help='Select which fold to test')
parser.add_argument('--model', default=1, type=int, dest='classifier', help="Please select a model")
args = parser.parse_args()

# Path where the models are saved
model_save_path = "Models/"

# Get the specified model using the custom function
model = get_model(args.classifier)
classifier_name = model.__class__.__name__

# Load the trained model for the specified fold
with open(f"{model_save_path}{classifier_name}_Fold_{args.fold-1}.pkl", 'rb') as f:
    model = joblib.load(f)

# Load the training data
train = pd.read_csv(r"C:\Users\Dhruv\Downloads\cluepoints_case\cluepoints_case\Case_Study_1\Data\hospital_stay_data.csv")

# Process the data using a custom function
processed_data = get_procesed_data(train)

# Separate features (X) and target variable (Y)
X = processed_data.loc[:, train.columns != 'Stay']
Y = processed_data['Stay']

# Initialize KFold cross-validator with 5 splits, shuffle data, and set random seed
kf = KFold(n_splits=5, shuffle=True, random_state=1)

# Get the training and testing indices for the specified fold
train_index, test_index = list(kf.split(X))[args.fold-1]

# Predict the target variable for the training and testing sets
y_pred_train = model.predict(X.iloc[train_index])
y_pred_test = model.predict(X.iloc[test_index])

# Print the performance metrics for the specified fold
print(f"****** Test : Fold {args.fold} on {classifier_name} ******")
print(f"Accuracy Train: {accuracy_score(Y.iloc[train_index], y_pred_train):.4f}, Accuracy Test: {accuracy_score(Y.iloc[test_index], y_pred_test):.4f}")
print(f"F1 Train: {f1_score(Y.iloc[train_index], y_pred_train, average='macro'):.4f}, F1 Test: {f1_score(Y.iloc[test_index], y_pred_test, average='macro'):.4f}")
