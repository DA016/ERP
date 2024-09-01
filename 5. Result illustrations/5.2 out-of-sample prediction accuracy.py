#out-of-sample prediction accuracy



import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Define input file paths for accuracy data (pre-trained and fine-tuned models)
input_files = {
    "pre_trained": "C:\\Users\\DA\\Desktop\\wrds\\new-data2\\6.15\\compare\\accuracy_year.csv",
    "fine_tuned": "C:\\Users\\DA\\Desktop\\wrds\\new-data2\\7.8\\compare\\updated_accuracy_year.csv"
}

# Define output paths for saving the accuracy plots
output_dir_pre_trained = "C:\\Users\\DA\\Desktop\\wrds\\new-data2\\6.15\\compare\\"
output_dir_fine_tuned = "C:\\Users\\DA\\Desktop\\wrds\\new-data2\\7.8\\compare\\"
output_image_pre_trained = os.path.join(output_dir_pre_trained, "Average_Accuracy_Pre_trained.png")
output_image_fine_tuned = os.path.join(output_dir_fine_tuned, "Average_Accuracy_Fine_tuned.png")

# Load the accuracy data from CSV files
accuracy_pre_trained = pd.read_csv(input_files["pre_trained"])
accuracy_fine_tuned = pd.read_csv(input_files["fine_tuned"])

# Plot accuracy per year for pre-trained models
plt.figure(figsize=(12, 6), dpi=300)
for column in accuracy_pre_trained.columns[1:]:  # Iterate through model columns
    plt.plot(accuracy_pre_trained['Year'], accuracy_pre_trained[column], label=column)
plt.axhline(y=0.5, color='gray', linestyle='--')  # Add a horizontal line at 0.5 accuracy
plt.xlabel('Year')
plt.ylabel('Average Accuracy')
plt.title('Average Accuracy per Year (Pre-trained)')
plt.legend()
plt.grid(True)
plt.savefig(output_image_pre_trained)  # Save the plot
plt.close()

# Plot accuracy per year for fine-tuned models
plt.figure(figsize=(12, 6), dpi=300)
for column in accuracy_fine_tuned.columns[1:]:  # Iterate through model columns
    plt.plot(accuracy_fine_tuned['Year'], accuracy_fine_tuned[column], label=column)
plt.axhline(y=0.5, color='gray', linestyle='--')  # Add a horizontal line at 0.5 accuracy
plt.xlabel('Year')
plt.ylabel('Average Accuracy')
plt.title('Average Accuracy per Year (Fine-tuned)')
plt.legend()
plt.grid(True)
plt.savefig(output_image_fine_tuned)  # Save the plot
plt.close()

print(f"Average accuracy per year plot for pre-trained models saved to {output_image_pre_trained}")
print(f"Average accuracy per year plot for fine-tuned models saved to {output_image_fine_tuned}")


# Define input file paths for accuracy data by ticker (pre-trained and fine-tuned models)
input_files = {
    "pre_trained": "C:\\Users\\DA\\Desktop\\wrds\\new-data2\\6.15\\compare\\accuracy_ticker.csv",
    "fine_tuned": "C:\\Users\\DA\\Desktop\\wrds\\new-data2\\7.8\\compare\\updated_accuracy_ticker.csv"
}

# Define output paths for saving the accuracy plots by ticker
output_dir_pre_trained = "C:\\Users\\DA\\Desktop\\wrds\\new-data2\\6.15\\compare\\"
output_dir_fine_tuned = "C:\\Users\\DA\\Desktop\\wrds\\new-data2\\7.8\\compare\\"
output_image_pre_trained = os.path.join(output_dir_pre_trained, "accuracy_ticker_pre_trained.png")
output_image_fine_tuned = os.path.join(output_dir_fine_tuned, "accuracy_ticker_fine_tuned.png")

# Load the accuracy data by ticker from CSV files
accuracy_pre_trained = pd.read_csv(input_files["pre_trained"])
accuracy_fine_tuned = pd.read_csv(input_files["fine_tuned"])

# Set colors for the bar plots
colors = ['#d5b2d1', '#6a96ca', '#fb96a3', '#91e6d9', '#c5d9ee']

# Function to plot accuracy per ticker
def plot_accuracy_per_ticker(df, title, output_path):
    tickers = df['Ticker']
    models = df.columns[1:]  # Exclude 'Ticker' column
    
    x = np.arange(len(tickers))  # Label locations
    width = 0.15  # Width of the bars
    
    fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
    
    for i, model in enumerate(models):  # Plot each model's accuracy as a bar
        ax.bar(x + i * width, df[model], width, label=model, color=colors[i])
    
    # Set labels and title
    ax.set_xlabel('Ticker')
    ax.set_ylabel('Average Accuracy')
    ax.set_title(title)
    ax.set_xticks(x + width * (len(models) - 1) / 2)  # Center the tick labels
    ax.set_xticklabels(tickers, rotation=90)  # Rotate ticker labels for readability
    ax.set_ylim(0.4, 0.65)  # Set y-axis limits
    ax.axhline(y=0.5, color='#1f77b4', linestyle='--')  # Horizontal line at 0.5 accuracy
    ax.legend()
    ax.grid(axis='y')  # Add grid lines
    
    fig.tight_layout()  # Adjust layout
    
    plt.savefig(output_path, dpi=300)  # Save the plot
    plt.close()

# Plot accuracy per ticker for pre-trained models
plot_accuracy_per_ticker(accuracy_pre_trained, 'Average Accuracy per Ticker (Pre-trained)', output_image_pre_trained)

# Plot accuracy per ticker for fine-tuned models
plot_accuracy_per_ticker(accuracy_fine_tuned, 'Average Accuracy per Ticker (Fine-tuned)', output_image_fine_tuned)

print(f"Average accuracy per ticker plot for pre-trained models saved to {output_image_pre_trained}")
print(f"Average accuracy per ticker plot for fine-tuned models saved to {output_image_fine_tuned}")

