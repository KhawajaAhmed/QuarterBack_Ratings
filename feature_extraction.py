def extract_significant_features(fitted_model):
    results = fitted_model.summary2()
    p_values = results.tables[1]['P>|t|']
    significant_features = p_values[p_values < 0.05].index.tolist()
    print("\nMost Significant Features:", significant_features, '\n')
    return significant_features
