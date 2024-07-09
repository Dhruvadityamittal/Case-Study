# Healthcare Dataset Analysis and Prediction

## Author: Dhruv Aditya Mittal

### Overview
This directory contains a comprehensive set of tools for analyzing and predicting length of stay from a healthcare dataset.

### Contents

#### Files:
- **main.py**: Implements 5-fold cross-validation with various classifiers.
  - Arguments
    - K: No of Folds
    - model: Type of Classifer
      -   Logistic Regression
      -   Decision Tree Classifier
      -   Random Forest Classifier
      -   K Neighbors Classifier
      -   Adaboost Classifier
      -   Quadratic Discriminant Analysis
      -   Linear Disciminant Analysis 
- **test.py**: Evaluates the performance of trained classifiers on different test datasets.
- **hospital_stay_EDA.ipynb**: Jupyter notebook for data analysis and Exploratory Data Analysis (EDA).
- **models.py**: Contains models for training.
- **utils.py**: Utility file for data analysis.
- **data_handling.py**: File containing functions for processing the data.
- **requirements.txt**: Libraries used to run the code.
  - To install the requirements :
    ```
    pip install -r requirements.txt
    ```

#### Folders:
- **Results**: Contains performance logs of various models.
- **Models**: Stores trained models from the training process.

### Web Application
- **test_web_application.py**: A Streamlit web application for predicting hospital stay duration.
  - To run the application, use the following command:
    ```
    streamlit run test_web_application.py
    ```
