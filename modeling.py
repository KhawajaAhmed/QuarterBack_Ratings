import statsmodels.api as sm
import pandas as pd

def fit_model(X, Y):
    model = sm.OLS(Y, X).fit()
    return model

def compare_predictions(model, X, Y):
    y_pred = model.predict(X)
    compare_df = pd.DataFrame({"Actual": Y, "Predicted": y_pred})
    print(compare_df)
    return compare_df

def calculate_difference(Y1, Y2):
    return pd.Series(Y2 - Y1)

def print_bottom_qbs(diff_series, df):
    bott_5_qbs = diff_series.sort_values(ascending=True).head(5)
    print(bott_5_qbs)
    indices_ls = bott_5_qbs.index.to_list()
    bott_5_qbs_df = df.iloc[indices_ls]
    print(bott_5_qbs_df)
    return bott_5_qbs_df
