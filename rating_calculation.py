def calc_passer_rating(row):
    num_attempts = row['Att']
    pass_completions = row['Cmp']
    pass_yds = row['Yds']
    pass_tds = row['TD']
    interceptions = row['Int']

    a = min(max((pass_completions / num_attempts - 0.3) * 5, 0), 2.375)
    b = min(max((pass_yds / num_attempts - 3) * 0.25, 0), 2.375)
    c = min(max((pass_tds / num_attempts) * 20, 0), 2.375)
    d = min(max(2.375 - (interceptions / num_attempts * 25), 0), 2.375)

    passer_rating = ((a + b + c + d) / 6) * 100
    return passer_rating

def calculate_all_ratings(df, passer_rating_name):
    for index in range(len(df)):
        row = df.iloc[index]
        qb_rating = calc_passer_rating(row)
        print("Name:", row['Player'], "  Rating: ", row[passer_rating_name], "   Calc Rating:", qb_rating)
