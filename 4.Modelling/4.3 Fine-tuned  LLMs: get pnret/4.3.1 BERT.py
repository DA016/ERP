import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, pipeline
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import time
import os
import joblib

# Base path and model path
base_path = "C:\\Users\\DA\\Desktop\\wrds\\new-data\\7.8\\"
model_path = "Bert"

# Load data
file_path = f"{base_path}{model_path}\\Ticker-Year-Week-wheadline-ndirection-nret.csv"
data = pd.read_csv(file_path)

# Ensure all values in the 'wheadline' column are strings and handle missing values
data['wheadline'] = data['wheadline'].astype(str).fillna('')

# Get all unique Tickers
tickers = data['Ticker'].unique()

# Initialize BERT model
bert_model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

# Select device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset class for sentiment analysis
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

# Function to get BERT inputs
def encode_batch(texts, tokenizer):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# Function to train the model
def train_model(train_texts, train_labels, model, tokenizer, training_args, model_name, ticker, year_range):
    train_encodings = encode_batch(train_texts, tokenizer)
    train_dataset = SentimentDataset(train_encodings, train_labels)
    
    # Adding a data collator
    data_collator = DataCollatorWithPadding(tokenizer, padding=True)

    trainer = Trainer(
        model=model,                         # Model to fine-tune
        args=training_args,                  # Training arguments
        train_dataset=train_dataset,         # Training dataset
        data_collator=data_collator          # Adding data collator
    )

    trainer.train()
    # Save the model weights, file name includes model, ticker, and year_range
    ticker_dir = f"{base_path}{model_path}\\{ticker}"
    model_save_path = f"{ticker_dir}\\{model_name}_{ticker}_{year_range}_fine_tuned_model"
    os.makedirs(model_save_path, exist_ok=True)  # Ensure directory exists
    trainer.save_model(model_save_path)
    
    # Release model and clear cache
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
    output_dir=f'{base_path}{model_path}\\results',          
    num_train_epochs=3,              
    per_device_train_batch_size=8,  
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir=f'{base_path}{model_path}\\logs',            
    learning_rate=3e-5,
    save_steps=20_000,  
    save_total_limit=2  
)

# Create an empty dataframe to store results for all test years
all_test_results = pd.DataFrame()

start_time = time.time()

# Set max sequence length for BERT model
max_length = 512

# Define train and test splits, fine-tuning every three years
train_test_splits = [
    (range(2005, 2016), [2016, 2017, 2018]),
    (range(2008, 2019), [2019, 2020, 2021]),
    (range(2011, 2022), [2022, 2023])
]

for ticker in tickers:
    # Filter data for the specific Ticker
    ticker_data = data[data['Ticker'] == ticker]
    
    for train_years, test_years in train_test_splits:
        train_data = ticker_data[ticker_data['Year'].isin(train_years)]
        test_data = ticker_data[ticker_data['Year'].isin(test_years)]
        
        # Prepare training data and labels, use sign of nret as the label
        train_texts = train_data['wheadline'].tolist()
        train_labels = np.sign(train_data['nret']).apply(lambda x: 1 if x > 0 else 0).tolist()  # Sign of nret as label
        
        # Reinitialize the model
        model = AutoModelForSequenceClassification.from_pretrained(bert_model_name, num_labels=2).to(device)
        
        # Train the model
        year_range = f"{min(train_years)}-{max(train_years)}"
        model = train_model(train_texts, train_labels, model, tokenizer, training_args, model_path, ticker, year_range)
        
        # Get sentiment scores for all years
        all_years_texts = ticker_data['wheadline'].tolist()
        ticker_data.loc[:, 'fscore'] = get_sentiment_scores(all_years_texts, model, tokenizer, max_length)
        
        # Re-fetch train_data and test_data, ensure they contain the fscore column
        train_data = ticker_data[ticker_data['Year'].isin(train_years)]
        test_data = ticker_data[ticker_data['Year'].isin(test_years)]
        
        # Save data with sentiment scores
        output_path = f"{base_path}{model_path}\\Ticker-Year-Week-wheadline-nret-fscore-{min(train_years)}-{max(test_years)}.csv"
        ticker_data.to_csv(output_path, index=False)
        
        # Prepare training data and labels using fscore as input feature
        train_fscore = train_data['fscore'].values.reshape(-1, 1)
        train_nret = train_data['nret'].values
        
        # Train linear regression model
        lr_model = LinearRegression()
        lr_model.fit(train_fscore, train_nret)
        
        # Predict on the test set
        test_fscore = test_data['fscore'].values.reshape(-1, 1)
        predictions = lr_model.predict(test_fscore)
        test_data.loc[:, 'pnret'] = predictions
        test_data.loc[:, 'pndirection'] = test_data['pnret'].apply(lambda x: 1 if x > 0 else 0)
        
        # Append results to all_test_results dataframe
        all_test_results = pd.concat([all_test_results, test_data], ignore_index=True)
        
        # Save test results
        test_output_path = f"{base_path}{model_path}\\Ticker-Year-Week-wheadline-nret-fscore-pnret-pndirection-{min(test_years)}-{max(test_years)}.csv"
        test_data.to_csv(test_output_path, index=False)
        
        # Save model information
        ticker_dir = f"{base_path}{model_path}\\{ticker}"
        os.makedirs(ticker_dir, exist_ok=True)
        bert_model_path = f"{ticker_dir}\\bert_model_{ticker}_{min(train_years)}-{max(train_years)}.bin"
        lr_model_path = f"{ticker_dir}\\lr_model_{ticker}_{min(train_years)}-{max(train_years)}.joblib"
        model.save_pretrained(bert_model_path)
        joblib.dump(lr_model, lr_model_path)

# Filter results for years 2016-2023
filtered_results = all_test_results[all_test_results['Year'].isin(range(2016, 2024))]
filtered_output_path = f"{base_path}{model_path}\\Ticker-Year-Week-wheadline-nret-fscore-pnret-pndirection16-23.csv"
filtered_results.to_csv(filtered_output_path, index=False)

# Record end time and compute elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Filtered results saved to {filtered_output_path}")
print(f"Elapsed time: {elapsed_time} seconds")
