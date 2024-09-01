import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Load data file
input_file_path = "C:\\Users\\DA\\Desktop\\wrds\\Ticker-Year-Week-wheadline-ndirection-nret.csv"
output_folder_path = "C:\\Users\\DA\\Desktop\\wrds\\new-data\\6.15\\distilbert"  # Updated output folder path

# Read data
data = pd.read_csv(input_file_path)

# Initialize sentiment analysis model with DistilBERT
distilbert_model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(distilbert_model_name)
model = AutoModelForSequenceClassification.from_pretrained(distilbert_model_name)

# Ensure all values in the wheadline column are strings and handle missing values
data['wheadline'] = data['wheadline'].astype(str).fillna('')

# Define a function to get sentiment scores using binary classification
def get_sentiment_scores(headlines, model, tokenizer):
    nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    sentiments = []
    for headline in headlines:
        result = nlp(headline)
        sentiments.append(result[0]['score'] if result[0]['label'] == 'POSITIVE' else -result[0]['score'])
    return sentiments

# Calculate sentiment scores using DistilBERT
start_time = time.time()
data['wscore'] = get_sentiment_scores(data['wheadline'].tolist(), model, tokenizer)
print(f"Sentiment calculation time: {time.time() - start_time} seconds")

# Save file with scores
score_output_file = f"{output_folder_path}\\Ticker-Year-Week-wheadline-ndirection-nret-score.csv"
data.to_csv(score_output_file, index=False)

# Remove the last week's missing values
data = data.dropna(subset=['nret', 'ndirection'])

# Define rolling window iterations
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
second_stage_model_info = []

# Perform rolling window prediction and calculate accuracy
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
        
        # Predict ndirection
        test_data = test_data.copy()  # Avoid SettingWithCopyWarning
        test_data['pnret'] = pnret
        test_data['pndirection'] = test_data['pnret'].apply(lambda x: 1 if x > 0 else 0)
        
        # Calculate accuracy
        accuracy = accuracy_score(test_data['ndirection'], test_data['pndirection'])
        
        results.append({
            'Ticker': ticker,
            'Year': test_year,
            'Accuracy': accuracy
        })
        
        # Save model parameters and performance
        second_stage_model_info.append({
            'Ticker': ticker,
            'Year': test_year,
            'Intercept': model.intercept_,
            'Coefficients': model.coef_.tolist(),
            'R^2 Score': model.score(X_train, y_train),
            'Accuracy': accuracy
        })

# Save results
results_df = pd.DataFrame(results)
accuracy_output_file = f"{output_folder_path}\\Ticker-Accuracy-Year.csv"
results_df.to_csv(accuracy_output_file, index=False)


# Save second stage model information
second_stage_model_info_output_file = f"{output_folder_path}\\second_stage_model_info.json"
with open(second_stage_model_info_output_file, 'w') as f:
    json.dump(second_stage_model_info, f, indent=4)

# Plotting results

# Read CSV file
file_path = f'{output_folder_path}\\Ticker-Accuracy-Year.csv'
df = pd.read_csv(file_path)

# Set color palette
palette = sns.color_palette("tab20", len(df['Ticker'].unique()))

# Create plot
plt.figure(figsize=(14, 8))

for i, ticker in enumerate(df['Ticker'].unique()):
    ticker_data = df[df['Ticker'] == ticker]
    plt.plot(ticker_data['Year'], ticker_data['Accuracy'], marker='o', label=ticker, color=palette[i])

# Add a dashed line for accuracy = 0.5
plt.axhline(y=0.5, color='gray', linestyle='--', label='Accuracy = 0.5')

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Set y-axis scale from 0 to 1
plt.title('Model accuracy through time for each ticker: distilbert')
plt.legend(title='Ticker', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Save plot
output_path = f'{output_folder_path}\\Model_accuracy_through_time_for_each_ticker_distilbert.png'
plt.savefig(output_path, bbox_inches='tight')
plt.show()

# Calculate summary statistics for each Ticker
ticker_summary_stats = df.groupby('Ticker')['Accuracy'].agg(['count', 'mean', 'std', 'min', 'max'])

# Round to three decimal places
ticker_summary_stats = ticker_summary_stats.round(3)

# Save results to CSV file
output_path = f'{output_folder_path}\\Ticker_Accuracy_Summary_Stats.csv'
ticker_summary_stats.to_csv(output_path)

# Print summary statistics
print(ticker_summary_stats)
