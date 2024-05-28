from data_processing import load_data, prepare_features, get_target_values
from feature_extraction import extract_significant_features
from rating_calculation import calculate_all_ratings
from modeling import fit_model, compare_predictions, calculate_difference, print_bottom_qbs

def main():
    file_path = 'data/qb2018_simple.csv'
    features = ['Age', 'G', 'GS', 'Cmp', 'Att', 'Cmp%', 'Yds',
                'TD', 'TD%', 'Int', 'Int%', 'Lng', 'Y/A', 'AY/A',
                'Y/C', 'Y/G', 'Sk', 'Yds.1', 'NY/A', 'ANY/A', 'Sk%']
    passer_rating_name = 'Rate'
    espn_rating_name = 'QBR'
    
    df = load_data(file_path)
    print(df)
    print(df.columns)

    X = prepare_features(df, features)
    Y1, Y2 = get_target_values(df, [passer_rating_name, espn_rating_name])

    calculate_all_ratings(df, passer_rating_name)

    model1 = fit_model(X, Y1)
    extract_significant_features(model1)

    compare_predictions(model1, X, Y1)

    model2 = fit_model(X, Y2)
    extract_significant_features(model2)

    qbr_features = ['Att', 'Cmp%', 'Y/C']
    qbr_df = prepare_features(df, qbr_features)
    model_qbr = fit_model(qbr_df, Y2)
    compare_predictions(model_qbr, qbr_df, Y2)

    rating_diff = calculate_difference(Y2, Y1)
    print(rating_diff)

    diff_model = fit_model(X, rating_diff)
    extract_significant_features(diff_model)

    print_bottom_qbs(rating_diff, df)

if __name__ == "__main__":
    main()
