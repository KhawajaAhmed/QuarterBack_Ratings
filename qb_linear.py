"""
    CSC 370: Quarterback ratings, linear models, information from data
    Group members: Khawaja Hussain Ahmed & Suleman Baloch
"""
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

def extract_features(fitted_model):
    results = fitted_model.summary2()
    p_values = results.tables[1]['P>|t|']
    
    print("\nMost Significant Features:" , p_values[p_values<0.05].index.tolist(),'\n') 

def calc_passer_rating(row):
    
    num_attempts = row['Att']
    pass_completions = row['Cmp']
    pass_yds= row['Yds']
    pass_tds=row['TD']
    interceptions=row['Int']

    a = min(max((pass_completions/num_attempts - 0.3)*5,0),2.375)
    b = min(max((pass_yds/num_attempts -3 )*0.25,0),2.375)
    c = min(max((pass_tds/num_attempts)*20,0),2.375)
    d = min(max(2.375-(interceptions/num_attempts *25),0),2.375)

    passer_rating = ((a+b+c+d)/6) *100
    return passer_rating

def main():
    df = pd.read_csv('data/qb2018_simple.csv')
    print(df)
    print(df.columns)
   
    # use these as your starting features.
    # visit here to learn more about each stat: 
    # https://www.pro-football-reference.com/years/2018/passing.htm, click on the Glossary link.
    features = ['Age', 'G', 'GS', 'Cmp', 'Att', 'Cmp%', 'Yds',
                'TD','TD%', 'Int', 'Int%', 'Lng', 'Y/A', 'AY/A',
                'Y/C', 'Y/G', 'Sk', 'Yds.1', 'NY/A', 'ANY/A', 'Sk%']

    # these are the names of your target values.
    passer_rating_name = 'Rate'
    espn_rating_name = 'QBR'

    # some useful subsets
    X = df.loc[:,features]
    X = sm.add_constant(X)

    for index in range(len(df)):
        row = df.iloc[index]
        qb_rating = calc_passer_rating(row)

        print("Name:",row['Player'],"  Rating: ",row[passer_rating_name],"   Calc Rating:",qb_rating)

    Y1 = df.loc[:, passer_rating_name]
    Y2 = df.loc[:, espn_rating_name]

    # Create and fit the model on your data
    model1 = sm.OLS(Y1, X).fit()
    extract_features(model1)

    '''
    Task 2
        The following features were significant contributors to Rate prediction:
            Cmp%
            TD%
            Int%
            AY/A
            NY/A
            ANY/A
        
            Cmp%,TD% and Int% are all used in the passer rating formula and therefore have a significant impact
            on the Rate prediction model. AY/A, NY/A and ANY/A are more advanced performace statistics that take
            a range of factors into account for e.g. touchdowns, interceptions, and yards gained.
            
    '''
    
    # Comparing the Actual and Predicted Values
    y_pred1 = model1.predict(X)
    compare_df = pd.DataFrame( {"Actual":Y1, "Predicted":y_pred1} )
    print(compare_df)

    # Creating and fitting the model on the QBR data
    model2 = sm.OLS(Y2,X).fit()
    extract_features(model2)
    '''
    Task 4
        Passer Rating offers a simpler perspective of passing efficiency. Whereas QBR considers completion percentage, attempts, 
        and yards per completion while factoring in a broader range of situational aspects. It seems like the significant features
        for QBR are more context-specifc and hence in-depth. It offers a deeper and more dynamic understanding of a quarterback's 
        impact on the game compared to the traditional Passer Rating, which provides a simpler but more limited perspective on passing 
        efficiency. 
    '''
    # QBR: Most Significant features
    qbr_features = ['Att', 'Cmp%', 'Y/C'] 

    qbr_df = df.loc[:,qbr_features]
    qbr_df = sm.add_constant(qbr_df)

    # Predicting the QBR values using the significant features
    model_qbr = sm.OLS(Y2,qbr_df).fit()
    y_pred2 = model_qbr.predict(qbr_df)

    # Data Frame to compare the Predicted and Actual values.
    compare_df = pd.DataFrame( {"Actual":Y2,"Predicted":y_pred2} )
    print('\n',compare_df,'\n')
    
    '''
    Task 5
        The Predicted and the Actual QBR values are close. The significant features are an accurate way to approximately predict
        the QBR values.  
    '''

    # Calculating the difference QBR and Rate
    rating_diff =  pd.Series(Y2-Y1)
    print(rating_diff)
    
    # Creating and Fitting the model.
    diff_model = sm.OLS(rating_diff,X).fit()
    extract_features(diff_model)

    bott_5_qbs = rating_diff.sort_values(ascending=True).head(5)
    print(bott_5_qbs)
    indices_ls = bott_5_qbs.index.to_list() 

    bott_5_qbs = df.iloc[indices_ls]
    print(bott_5_qbs)

    '''
    Task 6
        The players with a low number of pass attempts have a low QBR. Because these players have a low number of pass attempts,
        they also have a low Y/C. Hence their low QBR. 
    '''



main()
