import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import joblib

# Base path and model path
base_path = "C:\\Users\\DA\\Desktop\\wrds\\new-data\\7.8\\"
model_path = "Roberta"

# Load the data
file_path = f"{base_path}{model_path}\\Ticker-Year-Week-wheadline-ndirection-nret.csv"
data = pd.read_csv(file_path)

# Ensure all values in the headline column are strings and handle missing values
data['wheadline'] = data['wheadline'].astype(str).fillna('')

# Get all unique tickers
tickers = data['Ticker'].unique()

# Initialize Roberta model
roberta_model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)


# Select device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom dataset class
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

# Function to encode the input for the Roberta model
def encode_batch(texts, tokenizer):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# Function to train the model
def train_model(train_texts, train_labels, model, tokenizer, training_args, model_name, ticker, year_range):
    train_encodings = encode_batch(train_texts, tokenizer)
    train_dataset = SentimentDataset(train_encodings, train_labels)
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer, padding=True)

    trainer = Trainer(
        model=model,                         # Model to fine-tune
        args=training_args,                  # Training parameters
        train_dataset=train_dataset,         # Training dataset
        data_collator=data_collator          # Data collator
    )

    trainer.train()
    # Save model weights, filename includes model, ticker, and year_range
    ticker_dir = f"{base_path}{model_path}\\test\\{ticker}"
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

# Set training arguments
training_args = TrainingArguments(
    output_dir=f'{base_path}{model_path}\\test\\results',
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=16,  # Batch size for training
    warmup_steps=500,  # Number of warmup steps
    weight_decay=0.01,  # Weight decay
    logging_dir=f'{base_path}{model_path}\\test\\logs',
    learning_rate=3e-5,  # Learning rate
    save_steps=20000,  # Save interval
    save_total_limit=2  # Limit on the number of saved checkpoints
)

# Set the maximum sequence length for the Roberta model
max_length = 512

# Define training and testing splits, with fine-tuning every three years
train_test_splits = [
    (range(2005, 2016), [2016, 2017, 2018]),
    (range(2008, 2019), [2019, 2020, 2021]),
    (range(2011, 2022), [2022, 2023])
]

# Initialize a DataFrame to save all test results
all_test_results = pd.DataFrame()

start_time = time.time()

for ticker in tickers:
    # Filter data for the specific ticker
    ticker_data = data[data['Ticker'] == ticker]
    
    for train_years, test_years in train_test_splits:
        train_data = ticker_data[ticker_data['Year'].isin(train_years)]
        test_data = ticker_data[ticker_data['Year'].isin(test_years)]
        
        # Prepare training data and labels, using sign of nret as the label
        train_texts = train_data['wheadline'].tolist()
        train_labels = train_data['nret'].apply(lambda x: 1 if x > 0 else 0).tolist()
        
        # Reinitialize the model
        model = AutoModelForSequenceClassification.from_pretrained(roberta_model_name, num_labels=2).to(device)
        
        # Train the model
        year_range = f"{min(train_years)}-{max(train_years)}"
        model = train_model(train_texts, train_labels, model, tokenizer, training_args, model_path, ticker, year_range)
        
        # Save results
        output_path = f"{base_path}{model_path}\\test\\{ticker}_Ticker-Year-Week-wheadline-ndirection-nret-fscore-{min(train_years)}-{max(test_years)}.csv"
        ticker_data.to_csv(output_path, index=False)

        # Prepare training data and labels using fscore as the input feature
        train_fscore = train_data['fscore'].values.reshape(-1, 1)
        train_nret = train_data['nret'].values
        
        # Train the linear regression model
        lr_model = LinearRegression()
        lr_model.fit(train_fscore, train_nret)
        
        # Predict on the test set
        test_fscore = test_data['fscore'].values.reshape(-1, 1)
        predictions = lr_model.predict(test_fscore)
        test_data.loc[:, 'pnret'] = predictions
        test_data.loc[:, 'pndirection'] = test_data['pnret'].apply(lambda x: 1 if x > 0 else 0)
        
        # Append the results to the overall test results DataFrame
        all_test_results = pd.concat([all_test_results, test_data], ignore_index=True)
        
        # Save the test results
        test_output_path = f"{base_path}{model_path}\\test\\{ticker}_Ticker-Year-Week-wheadline-ndirection-nret-fscore-pnret-pndirection-{min(test_years)}-{max(test_years)}.csv"
        test_data.to_csv(test_output_path, index=False)

# Filter results for the years 2016-2023
filtered_results = all_test_results[all_test_results['Year'].isin(range(2016, 2024))]
filtered_output_path = f"{base_path}{model_path}\\test\\Ticker-Year-Week-wheadline-ndirection-nret-fscore-pnret-pndirection16-23.csv"
filtered_results.to_csv(filtered_output_path, index=False)

# Log time taken
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Filtered results saved to {filtered_output_path}")
print(f"Elapsed time: {elapsed_time} seconds")


