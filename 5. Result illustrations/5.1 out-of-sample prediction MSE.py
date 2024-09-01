#out-of-sample prediction MSE



import os
import pandas as pd
from sklearn.metrics import mean_squared_error

# Define base paths and model paths
base_path1 = "C:\\Users\\DA\\Desktop\\wrds\\new-data2\\6.15\\"
base_path2 = "C:\\Users\\DA\\Desktop\\wrds\\new-data2\\7.8\\"
output_path1 = f"{base_path1}\\compare\\"
output_path2 = f"{base_path2}\\compare\\"
model_paths = ["BERT", "RoBERTa", "DistilRoBERTa", "FinBERT", "DistilBERT"]

# Ensure output directories exist
os.makedirs(output_path1, exist_ok=True)
os.makedirs(output_path2, exist_ok=True)

# Prepare lists to hold the results
mse_results_pretrained = []
mse_results_finetuned = []

# Function to process data and calculate MSE for each ticker
def calculate_mse_for_model(base_path, output_path, model_paths, result_list):
    for model_path in model_paths:
        # Construct input file path
        input_file_path = f"{base_path}{model_path}\\Ticker-Week-ndirection-nret-Year-wscore-pnret-pndirection.csv" if '6.15' in base_path else f"{base_path}{model_path}\\Ticker-Year-Week-wheadline-ndirection-nret-fscore-pnret-pndirection16-23.csv"
        
        # Read the data
        data = pd.read_csv(input_file_path)
        
        # Filter data for the years 2016-2023
        data = data[(data['Year'] >= 2016) & (data['Year'] <= 2023)]
        
        # Drop rows with NaN values in 'nret' or 'pnret' columns
        data_filtered_mse = data.dropna(subset=['nret', 'pnret'])
        
        # Calculate MSE for each ticker
        mse_ticker = data_filtered_mse.groupby('Ticker').apply(lambda x: mean_squared_error(x['nret'], x['pnret']))
        mse_ticker_df = mse_ticker.reset_index()
        mse_ticker_df.columns = ['Ticker', model_path]
        
        # Append results to the list
        result_list.append(mse_ticker_df)

# Calculate MSE for pre-trained models
calculate_mse_for_model(base_path1, output_path1, model_paths, mse_results_pretrained)

# Calculate MSE for fine-tuned models
calculate_mse_for_model(base_path2, output_path2, model_paths, mse_results_finetuned)

# Merge the results for pre-trained models
merged_mse_pretrained = pd.concat(mse_results_pretrained, axis=1)
merged_mse_pretrained = merged_mse_pretrained.loc[:, ~merged_mse_pretrained.columns.duplicated()]

# Merge the results for fine-tuned models
merged_mse_finetuned = pd.concat(mse_results_finetuned, axis=1)
merged_mse_finetuned = merged_mse_finetuned.loc[:, ~merged_mse_finetuned.columns.duplicated()]

# Save results to CSV files
mse_output_file_pretrained = f"{output_path1}MSE_ticker_Pre-trained.csv"
mse_output_file_finetuned = f"{output_path2}MSE_ticker_Fine-tuned.csv"
merged_mse_pretrained.to_csv(mse_output_file_pretrained, index=False)
merged_mse_finetuned.to_csv(mse_output_file_finetuned, index=False)

print(f"MSE results for pre-trained models saved to {mse_output_file_pretrained}")
print(f"MSE results for fine-tuned models saved to {mse_output_file_finetuned}")




import pandas as pd
from sklearn.metrics import mean_squared_error
import os

# Define base paths and model paths
base_path1 = "C:\\Users\\DA\\Desktop\\wrds\\new-data2\\6.15\\"
base_path2 = "C:\\Users\\DA\\Desktop\\wrds\\new-data2\\7.8\\"
output_path1 = f"{base_path1}\\compare\\"
output_path2 = f"{base_path2}\\compare\\"
model_paths = ["BERT", "RoBERTa", "DistilRoBERTa", "FinBERT", "DistilBERT"]

# Ensure output directories exist
os.makedirs(output_path1, exist_ok=True)
os.makedirs(output_path2, exist_ok=True)

# Prepare lists to hold the results
mse_year_results_pretrained = []
mse_year_results_finetuned = []

# Function to process data and calculate MSE for each year
def calculate_mse_for_years(base_path, model_paths, mse_year_results, output_file_year):
    for model_path in model_paths:
        # Construct input file path
        input_file_path = f"{base_path}{model_path}\\Ticker-Week-ndirection-nret-Year-wscore-pnret-pndirection.csv" if '6.15' in base_path else f"{base_path}{model_path}\\Ticker-Year-Week-wheadline-ndirection-nret-fscore-pnret-pndirection16-23.csv"
        
        # Read the data
        data = pd.read_csv(input_file_path)
        
        # Filter data for the years 2016-2023
        data = data[(data['Year'] >= 2016) & (data['Year'] <= 2023)]
        
        # Drop rows with NaN values in 'nret' or 'pnret' columns
        data_filtered_mse = data.dropna(subset=['nret', 'pnret'])
        
        # Calculate MSE for each year
        mse_year = data_filtered_mse.groupby('Year').apply(lambda x: mean_squared_error(x['nret'], x['pnret']))
        mse_year_df = mse_year.reset_index()
        mse_year_df.columns = ['Year', model_path]
        
        # Append results to the list
        mse_year_results.append(mse_year_df)

    # Merge the results for each model
    merged_mse_year = pd.concat(mse_year_results, axis=1)
    merged_mse_year = merged_mse_year.loc[:, ~merged_mse_year.columns.duplicated()]

    # Save results to CSV file
    merged_mse_year.to_csv(output_file_year, index=False)

# Calculate MSE for years for pre-trained models
calculate_mse_for_years(base_path1, model_paths, mse_year_results_pretrained, f"{output_path1}MSE_year_Pre-trained.csv")

# Calculate MSE for years for fine-tuned models
calculate_mse_for_years(base_path2, model_paths, mse_year_results_finetuned, f"{output_path2}MSE_year_Fine-tuned.csv")

print("MSE calculations for years complete.")




import pandas as pd
import matplotlib.pyplot as plt
import os

# Define input file paths
input_files = {
    "MSE_year_Fine_tuned": "C:\\Users\\DA\\Desktop\\wrds\\new-data2\\7.8\\compare\\MSE_year_Fine-tuned.csv",
    "MSE_ticker_Fine_tuned": "C:\\Users\\DA\\Desktop\\wrds\\new-data2\\7.8\\compare\\MSE_ticker_Fine-tuned.csv",
    "MSE_year_Pre_trained": "C:\\Users\\DA\\Desktop\\wrds\\new-data2\\6.15\\compare\\MSE_year_Pre-trained.csv",
    "MSE_ticker_Pre_trained": "C:\\Users\\DA\\Desktop\\wrds\\new-data2\\6.15\\compare\\MSE_ticker_Pre-trained.csv"
}

# Define output paths
output_dir = "C:\\Users\\DA\\Desktop\\wrds\\new-data2\\7.8\\compare\\7.31"
os.makedirs(output_dir, exist_ok=True)
output_image_year = os.path.join(output_dir, "Average_MSE_Year.png")
output_image_ticker = os.path.join(output_dir, "Average_MSE_Ticker.png")

# Load data
mse_year_fine_tuned = pd.read_csv(input_files["MSE_year_Fine_tuned"])
mse_ticker_fine_tuned = pd.read_csv(input_files["MSE_ticker_Fine_tuned"])
mse_year_pre_trained = pd.read_csv(input_files["MSE_year_Pre_trained"])
mse_ticker_pre_trained = pd.read_csv(input_files["MSE_ticker_Pre_trained"])

# Calculate average MSE for each year (excluding the 'Year' column)
average_mse_year_pre_trained = mse_year_pre_trained.drop(columns=['Year']).mean(axis=1)
average_mse_year_fine_tuned = mse_year_fine_tuned.drop(columns=['Year']).mean(axis=1)

# Calculate average MSE for each ticker (excluding the 'Ticker' column)
average_mse_ticker_pre_trained = mse_ticker_pre_trained.drop(columns=['Ticker']).mean(axis=1)
average_mse_ticker_fine_tuned = mse_ticker_fine_tuned.drop(columns=['Ticker']).mean(axis=1)

# Plot MSE per year
plt.figure(figsize=(12, 6), dpi=300)
plt.plot(mse_year_pre_trained['Year'], average_mse_year_pre_trained, color='darkred', linestyle='-', linewidth=1, label='Average MSE Pre-trained')
plt.plot(mse_year_fine_tuned['Year'], average_mse_year_fine_tuned, color='blue', linestyle='--', linewidth=1, label='Average MSE Fine-tuned')
plt.xlabel('Year')
plt.ylabel('Average MSE')
plt.title('Average MSE per Year')
plt.legend()
plt.grid(True)
plt.savefig(output_image_year)
plt.close()

# Plot MSE per ticker
plt.figure(figsize=(12, 6), dpi=300)
plt.plot(mse_ticker_pre_trained['Ticker'], average_mse_ticker_pre_trained, color='darkred', linestyle='-', linewidth=1, label='Average MSE Pre-trained')
plt.plot(mse_ticker_fine_tuned['Ticker'], average_mse_ticker_fine_tuned, color='blue', linestyle='--', linewidth=1, label='Average MSE Fine-tuned')
plt.xlabel('Ticker')
plt.ylabel('Average MSE')
plt.title('Average MSE per Ticker')
plt.xticks(rotation=90, fontsize=8)  # Rotate and reduce the font size of ticker names
plt.legend()
plt.grid(True)
plt.savefig(output_image_ticker)
plt.close()

print(f"Average MSE per year plot saved to {output_image_year}")
print(f"Average MSE per ticker plot saved to {output_image_ticker}")





