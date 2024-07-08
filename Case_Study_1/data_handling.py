import numpy as np

def get_procesed_data(train):
    # Fill missing values with mode
    Bed_Grade_mode = train['Bed Grade'].mode()[0]
    train['Bed Grade'] = train['Bed Grade'].fillna(Bed_Grade_mode)
    City_Code_Patient_mode = train['City_Code_Patient'].mode()[0]
    train['City_Code_Patient'] = train['City_Code_Patient'].fillna(City_Code_Patient_mode)
    
    # Drop unnecessary columns
    train.drop(['case_id', 'patientid'], axis=1, inplace=True)

    # Identify categorical and numerical columns
    categorical_cols = [col for col in train.columns if train[col].dtypes == 'object']
    numerical_cols = [col for col in train.columns if train[col].dtypes != 'object']

    print(f"\nCategorical Features: {categorical_cols}")
    print(f"\nNumerical Features: {numerical_cols}\n" )

    # Encode categorical columns
    stay_encode = {
        '0-10': 1, '11-20': 2, '21-30': 3, '31-40': 4, '41-50': 5, '51-60': 6, '61-70': 7, '71-80': 8,
        '81-90': 9, '91-100': 10, 'More than 100 Days': 11
    }
    train['Stay'] = train['Stay'].map(stay_encode)

    age_encode = {
        '0-10': 1, '11-20': 2, '21-30': 3, '31-40': 4, '41-50': 5, '51-60': 6, '61-70': 7, '71-80': 8,
        '81-90': 9, '91-100': 10
    }
    train['Age'] = train['Age'].map(age_encode)

    department_encode = {
        'radiotherapy': 1, 'anesthesia': 2, 'gynecology': 3, 'TB & Chest disease': 4, 'surgery': 5
    }
    train['Department'] = train['Department'].map(department_encode)

    ward_type_encode = {
        'R': 1, 'S': 2, 'Q': 3, 'P': 4, 'T': 5, 'U': 6
    }
    train['Ward_Type'] = train['Ward_Type'].map(ward_type_encode)

    ward_facility_code_encode = {
        'F': 1, 'E': 2, 'D': 3, 'B': 4, 'A': 5, 'C': 6
    }
    train['Ward_Facility_Code'] = train['Ward_Facility_Code'].map(ward_facility_code_encode)

    admission_type_encode = {
        'Emergency': 1, 'Trauma': 2, 'Urgent': 3
    }
    train['Type of Admission'] = train['Type of Admission'].map(admission_type_encode)

    illness_severity_encode = {
        'Extreme': 1, 'Moderate': 2, 'Minor': 3
    }
    train['Severity of Illness'] = train['Severity of Illness'].map(illness_severity_encode)

    hospital_type_code_encode = {
        'c': 1, 'e': 2, 'b': 3, 'a': 4, 'f': 5, 'd': 6, 'g': 7
    }
    train['Hospital_type_code'] = train['Hospital_type_code'].map(hospital_type_code_encode)

    hospital_region_code_encode = {
        'Z': 1, 'X': 2, 'Y': 3
    }
    train['Hospital_region_code'] = train['Hospital_region_code'].map(hospital_region_code_encode)

    # Convert all columns to integers
    for column in train:
        train[column] = train[column].astype(np.int64)

    return train