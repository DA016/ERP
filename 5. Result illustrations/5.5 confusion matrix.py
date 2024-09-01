#confusion matrix


import pandas as pd
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Define the model paths and base paths
model_paths = ["BERT", "RoBERTa", "DistilRoBERTa", "FinBERT", "DistilBERT"]

# Define base paths for pre-trained and fine-tuned models
base_path1 = "C:\\Users\\DA\\Desktop\\wrds\\new-data2\\6.15\\"
base_path2 = "C:\\Users\\DA\\Desktop\\wrds\\new-data2\\7.8\\"

# Define comparison paths to save results
compare_path1 = os.path.join(base_path1, "compare")
compare_path2 = os.path.join(base_path2, "compare")

# Ensure the compare directories exist
os.makedirs(compare_path1, exist_ok=True)
os.makedirs(compare_path2, exist_ok=True)

# Function to process data, calculate metrics, and save confusion matrix
def process_data_and_save_metrics(base_path, model_path, compare_path, output_suffix):
    # Define input data path based on the base path
    input_data_path = (
        f"{base_path}{model_path}\\Ticker-Year-Week-wheadline-ndirection-nret-fscore-pnret-pndirection16-23.csv" 
        if '7.8' in base_path 
        else f"{base_path}{model_path}\\Ticker-Week-ndirection-nret-Year-wscore-pnret-pndirection.csv"
    )
    
    # Load the dataset
    df = pd.read_csv(input_data_path)
    
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    
    # Filter data for the years 2016-2023
    df = df[(df['Year'] >= 2016) & (df['Year'] <= 2023)]
    
    # Drop rows with NaN values in 'ndirection' or 'pndirection'
    df.dropna(subset=['ndirection', 'pndirection'], inplace=True)
    
    # Calculate classification metrics
    y_true = df['ndirection']
    y_pred = df['pndirection']
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    # Store the metrics in a dictionary
    metrics = {
        'Model': model_path,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_path} {output_suffix}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Save the confusion matrix plot
    output_file = os.path.join(compare_path, f'Confusion_Matrix_for_{model_path}_{output_suffix}.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    return metrics

# Process each model for both pre-trained and fine-tuned datasets and collect metrics
all_metrics = []

for model_path in model_paths:
    # Process and save metrics for pre-trained models
    metrics_pretrained = process_data_and_save_metrics(base_path1, model_path, compare_path1, 'Pre-trained')
    
    # Process and save metrics for fine-tuned models
    metrics_finetuned = process_data_and_save_metrics(base_path2, model_path, compare_path2, 'Fine-tuned')
    
    # Collect metrics for comparison
    all_metrics.append(metrics_pretrained)
    all_metrics.append(metrics_finetuned)

# Convert the collected metrics into a DataFrame
metrics_df = pd.DataFrame(all_metrics)

# Save the metrics to CSV files in both comparison directories
metrics_df.to_csv(os.path.join(compare_path1, "model_metrics.csv"), index=False)
metrics_df.to_csv(os.path.join(compare_path2, "model_metrics.csv"), index=False)

print("Metrics calculated and saved.")
