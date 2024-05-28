import pandas as pd
import statsmodels.api as sm

def load_data(file_path):
    return pd.read_csv(file_path)

def prepare_features(df, features):
    X = df.loc[:, features]
    X = sm.add_constant(X)
    return X

def get_target_values(df, target_names):
    return [df[target] for target in target_names]
