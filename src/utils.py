import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def preprocess_data(train, test):
    imputer = SimpleImputer(strategy='most_frequent')
    train_features = train.drop(columns=['Cancer'])
    train_imputed = pd.DataFrame(imputer.fit_transform(train_features), columns=train_features.columns)
    train_imputed['Cancer'] = train['Cancer'].values
    test_imputed = pd.DataFrame(imputer.transform(test), columns=test.columns)

    num_cols = ['Age', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result']
    train_imputed[num_cols] = train_imputed[num_cols].astype(float)
    test_imputed[num_cols] = test_imputed[num_cols].astype(float)

    train_imputed['T3_T4_ratio'] = train_imputed['T3_Result'] / (train_imputed['T4_Result'] + 1e-5)
    test_imputed['T3_T4_ratio'] = test_imputed['T3_Result'] / (test_imputed['T4_Result'] + 1e-5)
    train_imputed['TSH_T4_ratio'] = train_imputed['TSH_Result'] / (train_imputed['T4_Result'] + 1e-5)
    test_imputed['TSH_T4_ratio'] = test_imputed['TSH_Result'] / (test_imputed['T4_Result'] + 1e-5)
    train_imputed['T4_minus_T3'] = train_imputed['T4_Result'] - train_imputed['T3_Result']
    test_imputed['T4_minus_T3'] = test_imputed['T4_Result'] - test_imputed['T3_Result']

    cat_cols = ['Gender', 'Country', 'Race', 'Family_Background', 
                'Radiation_History', 'Iodine_Deficiency', 'Smoke', 
                'Weight_Risk', 'Diabetes']
    for col in cat_cols:
        le = LabelEncoder()
        train_imputed[col] = le.fit_transform(train_imputed[col])
        test_imputed[col] = le.transform(test_imputed[col])

    return train_imputed, test_imputed