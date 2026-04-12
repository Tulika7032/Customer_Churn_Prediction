import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    df=pd.read_csv(file_path)
    return df

def clean_data(df):
    df=df.copy()
    df.drop("customerID", axis=1, inplace=True)
    df["TotalCharges"]=pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    return df

def split_features(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X = encode_categorical(X)
    y = encode_target(y)
    return X, y

# Encode categorical features using one-hot encoding
def encode_categorical(X):
    X = pd.get_dummies(X, drop_first=True)
    return X

# Encode target variable if it's categorical (e.g., Yes/No)
def encode_target(y):
    if y.dtype == 'object' or str(y.dtype).startswith('category'):
        return y.map({'Yes': 1, 'No': 0}).astype(int)
    return y