import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import json
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data file
input_file_path = "C:\\Users\\DA\\Desktop\\wrds\\Ticker-Week-ndirection-nret-keydevid-headline-Year.csv"
output_folder_path = "C:\\Users\\DA\\Desktop\\wrds\\new-data\\6.15\\distilroberta"

# Read data
data = pd.read_csv(input_file_path)

# Ensure all values in the headline column are strings and handle missing values
data['headline'] = data['headline'].astype(str).fillna('')

# Initialize sentiment analysis model
pipe = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

# Define a function to get sentiment scores using softmax
def get_sentiment_scores(headlines, model, tokenizer):
    sentiments = []
    for headline in headlines:
        inputs = tokenizer(headline, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1).detach().numpy()[0]
        
        # Assuming the order of labels is ['negative', 'neutral', 'positive']
        negative_prob = probabilities[0]
        neutral_prob = probabilities[1]
        positive_prob = probabilities[2]
        
        # Assign a score based on the probabilities
        score = positive_prob - negative_prob
        sentiments.append(score)
    return sentiments

# Calculate sentiment scores (score)
start_time = time.time()
data['score'] = get_sentiment_scores(data['headline'], model, tokenizer)
print(f"Sentiment calculation time: {time.time() - start_time} seconds")

# Save file with scores
score_output_file = f"{output_folder_path}\\Ticker-Week-ndirection-nret-keydevid-headline-Year-score.csv"
data.to_csv(score_output_file, index=False)
print(f"File saved to {score_output_file}")

# Calculate weekly sentiment scores for each Ticker (wscore)
data['wscore'] = data['score']  # No need to aggregate since each Week has only one headline

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
            'Intercept': float(model.intercept_),  # Convert to standard float
            'Coefficients': [float(coef) for coef in model.coef_],  # Convert to standard float
            'R^2 Score': float(model.score(X_train, y_train)),  # Convert to standard float
            'Accuracy': float(accuracy)  # Convert to standard float
        })

# Save results
results_df = pd.DataFrame(results)
accuracy_output_file = f"{output_folder_path}\\Ticker-Accuracy-Year.csv"
results_df.to_csv(accuracy_output_file, index=False)

# Convert all elements in second_stage_model_info to standard serializable types
def convert_to_serializable(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

second_stage_model_info_serializable = convert_to_serializable(second_stage_model_info)

# Save second stage model information
second_stage_model_info_output_file = f"{output_folder_path}\\second_stage_model_info.json"
with open(second_stage_model_info_output_file, 'w') as f:
    json.dump(second_stage_model_info_serializable, f, indent=4)

print(f"Results saved to {accuracy_output_file}")
print(f"Second stage model info saved to {second_stage_model_info_output_file}")

# Plotting results
df = pd.read_csv(accuracy_output_file)

# Ensure we have enough unique colors
palette = sns.color_palette("tab20", 20) + sns.color_palette("tab20b", 20) + sns.color_palette("tab20c", 20)

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
plt.title('Model accuracy through time for each ticker: distilroberta')
plt.legend(title='Ticker', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Save plot
output_path = f'{output_folder_path}\\Model_accuracy_through_time_for_each_ticker_distilroberta.png'
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
