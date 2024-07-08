import pandas as pd
import warnings
from data_handling import get_procesed_data
from sklearn.metrics import accuracy_score,  f1_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from utils import get_class_dist
from models import get_model
import argparse
import joblib
import numpy as np

# save

# Suppress warnings
warnings.filterwarnings("ignore")


# Argument parser to take command-line arguments
parser = argparse.ArgumentParser(description='Cluepoints Case 1: Length of Stay in Hospital Classification - Dhruv Aditya Mittal')
parser.add_argument('--k', default=5, type=int, dest='fold_size', help='Please pass the value of K in K fold')
parser.add_argument('--model', default=5, type=int, dest='classifier', help="Please select a model")
args = parser.parse_args()
# Load the dataset

f1_train_before_sampling, f1_train_after_sampling = [], []
f1_test_before_sampling, f1_test_after_sampling = [], []
acc_train_before_sampling, acc_train_after_sampling = [], []
acc_test_before_sampling, acc_test_after_sampling = [], []

if __name__ == '__main__':
    train = pd.read_csv(r"C:\Users\Dhruv\Downloads\cluepoints_case\cluepoints_case\Health\Data\hospital_stay_data.csv")

    # Process the data using a custom function
    processed_data = get_procesed_data(train)

    # Separate features and target variable
    X = processed_data.loc[:, train.columns != 'Stay']
    Y = processed_data['Stay']
    # get_class_dist(Y)

    # Split the data into training and final test sets
    # x_train, x_test_final, y_train, y_test_final = train_test_split(X, Y, test_size=0.10, random_state=42, stratify=Y)

    result_path = 'Results/'
    model_save_path = "Models/"

    kf = KFold(n_splits=args.fold_size, shuffle=True, random_state=1)
    model = get_model(args.classifier)
    classifier_name =model.__class__.__name__

    print(f"Classifier Selected : {classifier_name}")

    print(f"\n******************************** Classifier: {classifier_name} ********************************\n")
    with open(f"{result_path}{classifier_name}_results.txt", "w") as file:
            file.write(f"\n******************************** Classifier: {classifier_name} ********************************\n")
            
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        # Initialize and train a Selected Classifier
        model.fit(x_train, y_train)

        # Predict on the final test set
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)

        f1_scr_train = f1_score(y_train, y_pred_train, average='macro')
        f1_scr_test = f1_score(y_test, y_pred_test, average='macro')
        

        # Calculate and print accuracy for training and test sets
        accuracy_test = accuracy_score(y_test, y_pred_test)
        accuracy_train = accuracy_score(y_train, y_pred_train)

        acc_train_before_sampling.append(accuracy_train)
        acc_test_before_sampling.append(accuracy_test)
        f1_train_before_sampling.append(f1_scr_train)
        f1_test_before_sampling.append(f1_scr_test)

        result_before_sampling = f"Before Sampling: Acc Train: {accuracy_train * 100:.2f}%, Test: {accuracy_test * 100:.2f}%, F1 Train: {f1_scr_train:.2f}, F1 Test: {f1_scr_test:.2f}\n"    
        
        print(f"\n----------------------------------------Fold: {fold+1}-----------------------------------------------------\n")
        print(result_before_sampling)
        with open(f"{result_path}{classifier_name}_results.txt", "a") as file:
            file.write(f"\n----------------------------------------Fold: {fold+1}-----------------------------------------------------\n")
            file.write(result_before_sampling)



        # Apply SMOTE to balance the dataset
        oversample = SMOTE()
        X_resampled, Y_resampled = oversample.fit_resample(X, Y)

        # Display new class distribution after resampling
        # get_class_dist(Y_resampled)

        # Train a DecisionTreeClassifier on the resampled data
        model.fit(X_resampled, Y_resampled)

        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)
        
        # Calculate F1 score and accuracy for the resampled model

        f1_scr_train = f1_score(y_test, y_pred_test, average='macro')
        f1_scr_test = f1_score(y_train, y_pred_train, average='macro')

        accuracy_test = accuracy_score(y_test, y_pred_test)
        accuracy_train = accuracy_score(y_train, y_pred_train)

        acc_train_after_sampling.append(accuracy_train)
        acc_test_after_sampling.append(accuracy_test)
        f1_train_after_sampling.append(f1_scr_train)
        f1_test_after_sampling.append(f1_scr_test)

        # print(f'{classifier_name} F1 Score: {f1_scr_test:.4f}')
        result_after_samping = f"After Resampling: Acc Train: {accuracy_train * 100:.2f}%, Test: {accuracy_test * 100:.2f}%, F1 Train: {f1_scr_train:.2f}, F1 Test: {f1_scr_test:.2f}\n"
        

        print(result_after_samping)
        

        with open(f"{result_path}{classifier_name}_results.txt", "a") as file:
            file.write(result_after_samping)
        
        joblib.dump(model, f"{model_save_path}{classifier_name}_Fold_{fold}.pkl") 

    
    result_summary = ""

    result_summary += (
        "\n----------------------------------------Result Summary ----------------------------------------\n" 
        f"Accuracy Train Before Sampling {np.mean(acc_train_before_sampling):.4f} ± {np.std(acc_train_before_sampling):.4f}\n"
        f"Accuracy Train After Sampling {np.mean(acc_train_after_sampling):.4f} ± {np.std(acc_train_after_sampling):.4f}\n\n"
        f"Accuracy Test Before Sampling {np.mean(acc_test_before_sampling):.4f} ± {np.std(acc_test_before_sampling):.4f}\n"
        f"Accuracy Test After Sampling {np.mean(acc_test_after_sampling):.4f} ± {np.std(acc_test_after_sampling):.4f}\n\n"
        
        f"F1 Train Before Sampling {np.mean(f1_train_before_sampling):.4f} ± {np.std(f1_train_before_sampling):.4f}\n"
        f"F1 Train After Sampling {np.mean(f1_train_after_sampling):.4f} ± {np.std(f1_train_after_sampling):.4f}\n\n"
        f"F1 Test Before Sampling {np.mean(f1_test_before_sampling):.4f} ± {np.std(f1_test_before_sampling):.4f}\n"
        f"F1 Test After Sampling {np.mean(f1_test_after_sampling):.4f} ± {np.std(f1_test_after_sampling):.4f}\n\n"
        "\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n" 
    )

    print(result_summary)
    with open(f"{result_path}{classifier_name}_results.txt", "a") as file:
            file.write(result_summary)
    

    file.close()
    print("Execution Successful..")
    print(f"View Logs at ->{result_path}{classifier_name}_results.txt")

    