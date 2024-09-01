#Sharpe ratio


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the data from the provided Excel files
pre_trained_sharpe_path = 'C:/Users/DA/Desktop/wrds/new-data/7.8/compare/pre-trained sharpe.xlsx'
fine_tuned_sharpe_path = 'C:/Users/DA/Desktop/wrds/new-data/7.8/compare/fine-tuned sharpe.xlsx'

pre_trained_sharpe_df = pd.read_excel(pre_trained_sharpe_path)
fine_tuned_sharpe_df = pd.read_excel(fine_tuned_sharpe_path)

# Filter numeric columns only
numeric_cols = pre_trained_sharpe_df.select_dtypes(include=[np.number]).columns.tolist()

# Calculate the mean Sharpe ratio for pre-trained and fine-tuned strategies
pre_trained_mean = pre_trained_sharpe_df[numeric_cols].mean()
fine_tuned_mean = fine_tuned_sharpe_df[numeric_cols].mean()

# Create a bar plot with dark blue and dark red colors
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)  # Increased DPI for higher resolution
bar_width = 0.35
index = np.arange(len(numeric_cols))

# Define dark blue and dark red colors
dark_blue = '#1f77b4'
dark_red = '#d62728'

# Plot bars for pre-trained and fine-tuned mean Sharpe ratios
bar1 = ax.bar(index, pre_trained_mean, bar_width, label='Pre-trained', color=dark_blue)
bar2 = ax.bar(index + bar_width, fine_tuned_mean, bar_width, label='Fine-tuned', color=dark_red)

# Add labels, title, and legend
ax.set_xlabel('Models')
ax.set_ylabel('Mean Sharpe Ratio')
ax.set_title('Mean Sharpe Ratio for Pre-trained and Fine-tuned Models')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(numeric_cols, rotation=45)
ax.legend()

# Add grid lines for better readability
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show plot
plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import os

# Define file paths
pre_trained_path = r"C:\Users\DA\Desktop\wrds\new-data2\sharpe pre-trained.xlsx"
fine_tuned_path = r"C:\Users\DA\Desktop\wrds\new-data2\sharpe fine-tuned.xlsx"
output_directory = os.path.dirname(pre_trained_path)  # Save output in the same directory

# Load the data from the Excel files
pre_trained_df = pd.read_excel(pre_trained_path)
fine_tuned_df = pd.read_excel(fine_tuned_path)

# Rename the first column to 'strategy' for clarity
pre_trained_df.rename(columns={'Unnamed: 0': 'strategy'}, inplace=True)
fine_tuned_df.rename(columns={'Unnamed: 0': 'strategy'}, inplace=True)

# Set 'strategy' as the index for easier plotting
pre_trained_df.set_index('strategy', inplace=True)
fine_tuned_df.set_index('strategy', inplace=True)

# Define the market Sharpe ratio
market_sharpe_ratio = 0.754

# Create the figure for plotting
plt.figure(figsize=(14, 10))

# Plot Pre-trained Sharpe ratios
plt.subplot(2, 1, 1)
pre_trained_df.plot(kind='bar', ax=plt.gca())
plt.axhline(y=market_sharpe_ratio, color='navy', linestyle='--', linewidth=2, label='Market=0.754')
plt.text(len(pre_trained_df) - 0.5, market_sharpe_ratio + 0.05, 'Market', color='navy', fontsize=10, ha='right')
plt.title('Sharpe Ratios per Strategy (Pre-trained)')
plt.ylabel('Sharpe Ratio')
plt.xticks(rotation=45)
plt.legend()

# Plot Fine-tuned Sharpe ratios
plt.subplot(2, 1, 2)
fine_tuned_df.plot(kind='bar', ax=plt.gca())
plt.axhline(y=market_sharpe_ratio, color='navy', linestyle='--', linewidth=2, label='Market=0.754')
plt.text(len(fine_tuned_df) - 0.5, market_sharpe_ratio + 0.05, 'Market', color='navy', fontsize=10, ha='right')
plt.title('Sharpe Ratios per Strategy (Fine-tuned)')
plt.ylabel('Sharpe Ratio')
plt.xticks(rotation=45)
plt.legend()

plt.tight_layout()

# Save the figure
output_file = os.path.join(output_directory, "Sharpe_Ratios_Comparison.png")
plt.savefig(output_file, dpi=300)

# Display the plot
plt.show()

print(f"Output saved to: {output_file}")



