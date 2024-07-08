# Import necessary libraries and modules ---> Run using -> Streamlit run test_web_application.py
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
import joblib  # For loading saved models
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation

def main():
    # HTML template for the app's header
    html_temp = """
    <div style="background-color:tomato;padding:10px;border-radius:10px">
    <h2 style="color:white;text-align:center;">Hospital Stay Duration Prediction Tool</h2>
    </div>
    <br>
    """
    st.markdown(html_temp, unsafe_allow_html=True)  # Display the header

    # Encoding dictionaries for various categorical features
    age_encode = { '0-10': 1, '11-20': 2, '21-30': 3, '31-40': 4, '41-50': 5, '51-60': 6, '61-70': 7, '71-80': 8, '81-90': 9, '91-100': 10 }
    department_encode = { 'radiotherapy': 1, 'anesthesia': 2, 'gynecology': 3, 'TB & Chest disease': 4, 'surgery': 5 }
    ward_type_encode = { 'R': 1, 'S': 2, 'Q': 3, 'P': 4, 'T': 5, 'U': 6 }
    ward_facility_code_encode = { 'F': 1, 'E': 2, 'D': 3, 'B': 4, 'A': 5, 'C': 6 }
    admission_type_encode = { 'Emergency': 1, 'Trauma': 2, 'Urgent': 3 }
    illness_severity_encode = { 'Extreme': 1, 'Moderate': 2, 'Minor': 3 }
    hospital_type_code_encode = { 'c': 1, 'e': 2, 'b': 3, 'a': 4, 'f': 5, 'd': 6, 'g': 7 }
    hospital_region_code_encode = { 'Z': 1, 'X': 2, 'Y': 3 }
    stay_encode = { '0-10': 1, '11-20': 2, '21-30': 3, '31-40': 4, '41-50': 5, '51-60': 6, '61-70': 7, '71-80': 8, '81-90': 9, '91-100': 10, 'More than 100 Days': 11 }
    reversed_stay_encode = {value: key for key, value in stay_encode.items()}  # Reverse the stay encoding dictionary for decoding predictions

    # Model selection dropdown in the app
    model_name = st.selectbox('Select the Base Model', [
        "Logistic Regression", 
        "Decision Tree", 
        "Random Forest Classifier", 
        "K Neighbour Neighbours", 
        "Adaboost Classifier", 
        "Quadric Discriminant Analysis", 
        "Linear Discriminant Analysis"
    ])

    # User input fields for various features
    hospital_code = st.text_input("Hospital Code")
    hospital_type_code = st.selectbox('Hospital Type Code', hospital_type_code_encode.keys())
    city_code_hospital = st.text_input("City Code")
    hospital_region_code = st.selectbox('Hospital Region Code', hospital_region_code_encode.keys())
    available_extra_rooms_hospital = st.text_input("Available Extra Rooms")
    department = st.selectbox('Department', department_encode.keys())
    ward_type = st.selectbox('Ward Type', ward_type_encode.keys())
    ward_facility_code = st.selectbox('Ward Facility Code', ward_facility_code_encode.keys())
    bed_grade = st.selectbox('Bed Grade', ["1", "2", "3", "4"])
    city_code_patient = st.text_input("City Code Patient")
    type_of_admission = st.selectbox('Type of Admission', admission_type_encode.keys())
    severity_of_illness = st.selectbox('Severity of Illness', illness_severity_encode.keys())
    visitors_with_the_patient = st.text_input("Visitors")
    age = st.selectbox('Age', age_encode.keys())
    admission_deposit = st.text_input("Admission Deposit")

    # Preparing the input data for prediction
    input_data = [
        hospital_code, 
        hospital_type_code_encode[hospital_type_code], 
        city_code_hospital, 
        hospital_region_code_encode[hospital_region_code],
        available_extra_rooms_hospital, 
        department_encode[department], 
        ward_type_encode[ward_type], 
        ward_facility_code_encode[ward_facility_code],
        bed_grade, 
        city_code_patient, 
        admission_type_encode[type_of_admission], 
        illness_severity_encode[severity_of_illness],
        visitors_with_the_patient, 
        age_encode[age], 
        admission_deposit
    ]

    # Dictionary to map model names to their corresponding sklearn classifier
    model_name_dict = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest Classifier": RandomForestClassifier(),
        "K Neighbour Neighbours": KNeighborsClassifier(),
        "Adaboost Classifier": AdaBoostClassifier(),
        "Quadric Discriminant Analysis": QuadraticDiscriminantAnalysis(),
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis()
    }

    # Path where the models are saved
    model_save_path = "Models/"
    model_class = model_name_dict[model_name].__class__.__name__  # Get the class name of the selected model

    # Prediction button
    if st.button("Predict", type="primary"):
        # Load the trained model for the first fold
        with open(f"{model_save_path}{model_class}_Fold_0.pkl", 'rb') as f:
            model = joblib.load(f)
        # Convert input data to a numpy array of floats
        input_data = np.array([[float(data) for data in input_data]])
        # Make the prediction
        out = model.predict(input_data)
        # Display the predicted stay length
        st.write(f"Predicted Stay Length: {reversed_stay_encode[out[0]]} days")

# Entry point of the application
if __name__ == "__main__":
    main()
