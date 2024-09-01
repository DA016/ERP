#transaction costs

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Define model paths
model_paths = ["BERT", "RoBERTa", "DistilRoBERTa", "FinBERT", "DistilBERT"]

# Define base paths for datasets
base_path1 = "C:\\Users\\DA\\Desktop\\wrds\\new-data2\\6.15\\"
base_path2 = "C:\\Users\\DA\\Desktop\\wrds\\new-data2\\7.8\\"

# Define transaction cost
transaction_cost = 0.001121

# Function to update portfolio returns considering transaction costs
def update_returns(file_path, output_path):
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Subtract transaction cost from Returns
        df['Returns'] = df['Returns'] - transaction_cost
        
        # Calculate Log_Returns and Cumulative_Log_Return
        df['Log_Returns'] = np.log1p(df['Returns'])
        df['Cumulative_Log_Return'] = df.groupby('Strategy')['Log_Returns'].cumsum()
        
        # Save the updated data
        df.to_csv(output_path, index=False)
        print(f"Updated file saved to {output_path}")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# Update portfolio returns for each model in base_path1
for model_path in model_paths:
    input_file_path1 = f"{base_path1}{model_path}\\portfolio_returns_weekly.csv"
    output_file_path1 = f"{base_path1}{model_path}\\8.23portfolio_returns_weekly.csv"
    update_returns(input_file_path1, output_file_path1)

# Update portfolio returns for each model in base_path2
for model_path in model_paths:
    input_file_path2 = f"{base_path2}{model_path}\\portfolio_returns_weekly.csv"
    output_file_path2 = f"{base_path2}{model_path}\\8.23portfolio_returns_weekly.csv"
    update_returns(input_file_path2, output_file_path2)

# Function to calculate annual returns
def calculate_annual_returns(df):
    return df.groupby(['Year', 'Strategy'])['Returns'].apply(lambda x: np.prod(1 + x) - 1).reset_index().rename(columns={'Returns': 'Yret'})

# Function to calculate Sharpe ratio
def compute_sharpe_ratio(annual_returns, risk_free_rate):
    results = []
    for strategy in annual_returns['Strategy'].unique():
        strategy_data = annual_returns[annual_returns['Strategy'] == strategy]
        avg_yret = strategy_data['Yret'].mean()
        stddev = strategy_data['Yret'].std()
        sharpe_ratio = (avg_yret - risk_free_rate) / stddev
        results.append({'Strategy': strategy, 'AvgYret': avg_yret, 'StdDev': stddev, 'Sharpe ratio': sharpe_ratio})
    return pd.DataFrame(results)

# Function to process each model in a base path
def process_base_path(base_path):
    for model_path in model_paths:
        input_file_path = f"{base_path}{model_path}\\8.23portfolio_returns_weekly.csv"
        output_annual_path = f"{base_path}{model_path}\\8.23portfolio_annual_returns.csv"
        output_sharpe_path = f"{base_path}{model_path}\\8.23Strategy-AvgYret-StdDev-Sharpe ratio-{model_path}.csv"

        try:
            # Load the updated returns data
            df = pd.read_csv(input_file_path)

            # Calculate annual returns and save the results
            annual_returns = calculate_annual_returns(df)
            annual_returns.to_csv(output_annual_path, index=False)
            print(f"Annual returns saved to {output_annual_path}")

            # Calculate Sharpe ratio and save the results
            sharpe_ratios_df = compute_sharpe_ratio(annual_returns, risk_free_rate)
            sharpe_ratios_df.to_csv(output_sharpe_path, index=False)
            print(f"Sharpe ratios saved to {output_sharpe_path}")

        except Exception as e:
            print(f"Error processing file {input_file_path}: {e}")

# Process data for base_path1
process_base_path(base_path1)

# Process data for base_path2
process_base_path(base_path2)

# Function to combine Sharpe ratio results from all model paths
def combine_sharpe_ratios(base_path, output_combined_path):
    combined_df = pd.DataFrame()
    
    for model_path in model_paths:
        sharpe_file_path = f"{base_path}{model_path}\\8.23Strategy-AvgYret-StdDev-Sharpe ratio-{model_path}.csv"
        print(f"Processing file: {sharpe_file_path}")
        
        try:
            model_df = pd.read_csv(sharpe_file_path)
            model_df.rename(columns={
                'AvgYret': f'{model_path}AvgYret',
                'StdDev': f'{model_path}StdDev',
                'Sharpe ratio': f'{model_path}Sharpe ratio'
            }, inplace=True)
            
            if combined_df.empty:
                combined_df = model_df
            else:
                combined_df = pd.concat([combined_df, model_df.drop(columns=['Strategy'])], axis=1)
        
        except FileNotFoundError:
            print(f"File not found: {sharpe_file_path}")
        except pd.errors.EmptyDataError:
            print(f"Empty data in file: {sharpe_file_path}")
        except Exception as e:
            print(f"Error processing file {sharpe_file_path}: {e}")

    combined_df.to_csv(output_combined_path, index=False)
    print(f"Combined Sharpe ratios saved to {output_combined_path}")

# Combine Sharpe ratio results for base_path1
output_combined_path1 = os.path.join(base_path1, "compare", "8.23Strategy-AvgYret-StdDev-Sharpe ratio-all.csv")
os.makedirs(os.path.dirname(output_combined_path1), exist_ok=True)
combine_sharpe_ratios(base_path1, output_combined_path1)

# Combine Sharpe ratio results for base_path2
output_combined_path2 = os.path.join(base_path2, "compare", "8.23Strategy-AvgYret-StdDev-Sharpe ratio-all.csv")
os.makedirs(os.path.dirname(output_combined_path2), exist_ok=True)
combine_sharpe_ratios(base_path2, output_combined_path2)

# Plot the Sharpe Ratios considering transaction costs
def plot_sharpe_ratios(file_path, output_file):
    df = pd.read_excel(file_path)
    df.rename(columns={'Unnamed: 0': 'strategy'}, inplace=True)
    df.set_index('strategy', inplace=True)

    plt.figure(figsize=(14, 7))
    ax = df.plot(kind='bar', ax=plt.gca())

    # Set y-axis limits to be symmetrical around 0
    y_lim = max(abs(df.min().min()), abs(df.max().max()))
    ax.set_ylim(-y_lim, y_lim)
    ax.axhline(0, color='black', linewidth=1)

    plt.title('Sharpe Ratios per Strategy (Fine-tuned - Considering Transaction Costs)')
    plt.ylabel('Sharpe Ratio')
    plt.xticks(rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()

    print(f"Output saved to: {output_file}")

# Define the new file path and output path
fine_tuned_path = r"C:\Users\DA\Desktop\wrds\new-data2\sharpe fine-tuned - consider transaction cost.xlsx"
output_file = os.path.join(os.path.dirname(fine_tuned_path), "Fine-Tuned_Sharpe_Ratios_Considering_Transaction_Costs_Centered.png")

# Plot the Sharpe Ratios
plot_sharpe_ratios(fine_tuned_path, output_file)
