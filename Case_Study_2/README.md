# Stack Overflow Questions Classification

## Author: Dhruv Aditya Mittal

### Overview
This directory provides a complete set of tools for classifying Stack Overflow questions based on their quality. Using various machine learning and natural language processing techniques, the project aims to categorize questions into different quality tiers, aiding in the effective management of community-generated content. The tools provided facilitate data preprocessing, model training, evaluation, and deployment through a web application.

### Contents

#### Files:
- **main.py**: Implements the training process using 5-fold cross-validation with various classifiers.
- Arguments
    - BATCH_SIZE: Batch Size.
    - EPOCHS: Number of Epochs.
    - MAX_LEN: Max Tokenization length.
    - TOKENIZER: Tokenizer type -> bert/bert_large.
    - REMOVE_STOP_WORDS: Remove Stop Words -> True/False.
    - FOLDS: Number of Fold for cross validation.
    - EARLY_STOPPING_COUNT: Early stopping count for best validation accuracy.
      
- **model.py**: Contains the architecture of the classifiers used for predicting the quality of Stack Overflow questions.
- **data_handling.py**: Includes functions for loading and preprocessing the dataset.
- **utils_EDA.py**: Includes functions for Exploratory Data Analysis (EDA).
- **utils.py**: Provides utility functions used throughout the project.
- **test_web_application.py**: Tests the output of trained classifiers on different test datasets.
- **Stack_Overflow_EDA.ipynb**: Jupyter notebook for data analysis and Exploratory Data Analysis (EDA).
- - **requirements.txt**: Libraries used to run the code.
  - To install the requirements :
    ```
    pip install -r requirements.txt
    ```

#### Folders:
- **Results**: Stores performance logs of various models.
- **Saved_Models**: Contains trained models for each fold from the training process.
- **Data**: Includes the dataset used for training and testing the models.

### Web Application
- **test_web_application.py**: A Streamlit web application for predicting the quality of Stack Overflow questions.
  - To run the application, use the following command:
    ```
    streamlit run test_web_application.py
    ```
### Web Application Video Snip    
https://github.com/Dhruvadityamittal/Case-Study/assets/22571412/26af9ea2-5086-4829-a139-ec367435ef64


