import pandas as pd
from sklearn.linear_model import LinearRegression

# Base path for the files
base_path = "C:\\Users\\DA\\Desktop\\wrds\\new-data\\6.15\\"

# List of models to process
models = ["bert", "roberta", "distilroberta", "finbert", "distilbert"]

for model in models:
    # File paths for the current model
    input_file_path = f"{base_path}{model}\\Ticker-Week-ndirection-nret-keydevid-headline-Year-score.csv"
    output_file_path_1 = f"{base_path}{model}\\Ticker-Week-ndirection-nret-Year-wscore.csv"
    output_file_path_2 = f"{base_path}{model}\\Ticker-Week-ndirection-nret-Year-wscore-pnret.csv"

    # Read the original data file
    data = pd.read_csv(input_file_path)

    # Set weekly sentiment score (wscore)
    data['wscore'] = data['score']

    # Create a new table with the specified columns
    output_data_1 = data[['Ticker', 'Week', 'ndirection', 'nret', 'Year', 'wscore']]
    output_data_1.to_csv(output_file_path_1, index=False)
    print(f"File saved to {output_file_path_1}")

    # Perform rolling window prediction to generate pnret
    iterations = {
        1: (2005, 2015, 2016),
        2: (2006, 2016, 2017),
        3: (2007, 2017, 2018),
        4: (2008, 2018, 2019),
        5: (2009, 2019, 2020),
        6: (2010, 2020, 2021),
        7: (2011, 2021, 2022),
        8: (2012, 2022, 2023)
    }

    results = []

    for ticker in data['Ticker'].unique():
        ticker_data = data[data['Ticker'] == ticker]
        
        for i in range(1, 9):
            train_start, train_end, test_year = iterations[i]
            train_data = ticker_data[(ticker_data['Year'] >= train_start) & (ticker_data['Year'] <= train_end)]
            test_data = ticker_data[ticker_data['Year'] == test_year]
            
            if train_data.empty or test_data.empty:
                continue
            
            # Train a simple linear regression model to predict pnret
            X_train = train_data[['wscore']].values
            y_train = train_data['nret'].values
            X_test = test_data[['wscore']].values

            model = LinearRegression()
            model.fit(X_train, y_train)
            pnret = model.predict(X_test)
            
            # Store the prediction results
            test_data = test_data.copy()  # Avoid SettingWithCopyWarning
            test_data['pnret'] = pnret
            
            results.append(test_data)

    # Combine results into a DataFrame and save to CSV
    results_df = pd.concat(results)
    results_df = results_df[(results_df['Year'] >= 2016) & (results_df['Year'] <= 2023)]
    results_df.to_csv(output_file_path_2, index=False)
    print(f"File saved to {output_file_path_2}")
