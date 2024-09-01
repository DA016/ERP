import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, pipeline
import torch
from sklearn.linear_model import LinearRegression
import numpy as np
import time
import os
import joblib

# Function to reset the results dataframe
def reset_data():
    global all_test_results
    all_test_results = pd.DataFrame()

# Reset data before running the code
reset_data()

# Base paths for the model and data
base_path = "C:\\Users\\DA\\Desktop\\wrds\\new-data\\7.8\\"
model_path = "Distilbert"

# Load the input data
file_path = f"{base_path}Ticker-Year-Week-wheadline-ndirection-nret.csv"
data = pd.read_csv(file_path)

# Ensure all values in the 'wheadline' column are strings and handle missing values
data['wheadline'] = data['wheadline'].astype(str).fillna('')

# Get all unique tickers from the data
tickers = data['Ticker'].unique()

# Initialize the DistilBERT model
distilbert_model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(distilbert_model_name)

# Set the device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom dataset class for sentiment analysis
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].to(device) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long).to(device)
        return item

    def __len__(self):
        return len(self.labels)

# Function to encode the batch for model input
def encode_batch(texts, tokenizer):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# Function to train the model
def train_model(train_texts, train_labels, model, tokenizer, training_args, model_name, ticker, year_range):
    train_encodings = encode_batch(train_texts, tokenizer)
    train_dataset = SentimentDataset(train_encodings, train_labels)
    
    # Add data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer, padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    trainer.train()
    
    # Save the fine-tuned model
    ticker_dir = f"{base_path}{model_path}\\test\\{ticker}"
    model_save_path = f"{ticker_dir}\\{model_name}_{ticker}_{year_range}_fine_tuned_model"
    os.makedirs(model_save_path, exist_ok=True)
    trainer.save_model(model_save_path)
    
    # Clean up
    del trainer
    del model
    torch.cuda.empty_cache()

    # Reload the model
    model = AutoModelForSequenceClassification.from_pretrained(model_save_path, num_labels=2).to(device)
    return model

# Function to get sentiment scores
def get_sentiment_scores(headlines, model, tokenizer, max_length):
    nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    sentiments = []
    for headline in headlines:
        result = nlp(headline, truncation=True, max_length=max_length)[0]
        sentiments.append(result['score'] if result['label'] == 'POSITIVE' else -result['score'])
    return sentiments

# Training arguments
training_args = TrainingArguments(
    output_dir=f'{base_path}{model_path}\\test\\results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    warmup_steps=1000,
    weight_decay=0.01,
    logging_dir=f'{base_path}{model_path}\\test\\logs',
    learning_rate=3e-5,
    save_steps=20000,
    save_total_limit=2
)

# Define the max sequence length for the model
max_length = 512

# Define the train-test splits for fine-tuning
train_test_splits = [
    (range(2005, 2016), [2016, 2017, 2018]),
    (range(2008, 2019), [2019, 2020, 2021]),
    (range(2011, 2022), [2022, 2023])
]

# Start the fine-tuning and prediction process
start_time = time.time()

for ticker in tickers:
    ticker_all_test_results = pd.DataFrame()
    ticker_data = data[data['Ticker'] == ticker]
    
    for train_years, test_years in train_test_splits:
        train_data = ticker_data[ticker_data['Year'].isin(train_years)]
        test_data = ticker_data[ticker_data['Year'].isin(test_years)]
        
        # Prepare training data using sign of nret as labels
        train_texts = train_data['wheadline'].tolist()
        train_labels = np.sign(train_data['nret']).astype(int).tolist()
        
        # Initialize the model
        model = AutoModelForSequenceClassification.from_pretrained(distilbert_model_name, num_labels=2).to(device)

        # Train the model
        year_range = f"{min(train_years)}-{max(train_years)}"
        model = train_model(train_texts, train_labels, model, tokenizer, training_args, model_path, ticker, year_range)
        
        # Get sentiment scores for all years
        all_years_texts = ticker_data['wheadline'].tolist()
        ticker_data.loc[:, 'fscore'] = get_sentiment_scores(all_years_texts, model, tokenizer, max_length)
        
        # Re-fetch train and test data to include fscore
        train_data = ticker_data[ticker_data['Year'].isin(train_years)]
        test_data = ticker_data[ticker_data['Year'].isin(test_years)]
        
        # Save data with fscore
        output_path = f"{base_path}{model_path}\\test\\Ticker-Year-Week-wheadline-ndirection-nret-fscore-{min(train_years)}-{max(test_years)}.csv"
        ticker_data.to_csv(output_path, index=False)
        
        # Train a linear regression model using fscore
        train_fscore = train_data['fscore'].values.reshape(-1, 1)
        train_nret = train_data['nret'].values
        lr_model = LinearRegression()
        lr_model.fit(train_fscore, train_nret)
        
        # Predict pnret for the test set
        test_fscore = test_data['fscore'].values.reshape(-1, 1)
        predictions = lr_model.predict(test_fscore)
        test_data.loc[:, 'pnret'] = predictions
        test_data.loc[:, 'pndirection'] = test_data['pnret'].apply(lambda x: 1 if x > 0 else 0)
        
        # Append test results
        ticker_all_test_results = pd.concat([ticker_all_test_results, test_data], ignore_index=True)
        
        # Save test results
        test_output_path = f"{base_path}{model_path}\\test\\{ticker}_Ticker-Year-Week-wheadline-ndirection-nret-fscore-pnret-pndirection-{min(test_years)}-{max(test_years)}.csv"
        test_data.to_csv(test_output_path, index=False)
    
    # Save all results for the current ticker
    ticker_dir = f"{base_path}{model_path}\\test\\{ticker}"
    os.makedirs(ticker_dir, exist_ok=True)
    distilbert_model_path = f"{ticker_dir}\\distilbert_model_{ticker}_{min(train_years)}-{max(train_years)}.bin"
    lr_model_path = f"{ticker_dir}\\lr_model_{ticker}_{min(train_years)}-{max(train_years)}.joblib"
    model.save_pretrained(distilbert_model_path)
    joblib.dump(lr_model, lr_model_path)
    
    # Append ticker results to all test results
    all_test_results = pd.concat([all_test_results, ticker_all_test_results], ignore_index=True)

# Filter results for the years 2016-2023 and save
filtered_results = all_test_results[all_test_results['Year'].isin(range(2016, 2024))]
filtered_output_path = f"{base_path}{model_path}\\test\\Ticker-Year-Week-wheadline-ndirection-nret-fscore-pnret-pndirection16-23.csv"
filtered_results.to_csv(filtered_output_path, index=False)

# Record the elapsed time
end_time = time.time()
elapsed_time = end_time - start
print(f"Filtered results saved to {filtered_output_path}")
print(f"Elapsed time: {elapsed_time} seconds")
