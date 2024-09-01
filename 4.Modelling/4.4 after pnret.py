#after pnret
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Base path and model paths
base_path = "C:\\Users\\DA\\Desktop\\wrds\\new-data\\7.8\\"

model_paths = ["Bert", "Distilbert", "Finbert", "Distilroberta", "Roberta"]

# Function to calculate metrics
def calculate_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    
    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
    specificity = tn / (tn + fp)
    
    return accuracy, precision, recall, specificity, f1

# Function to read data and calculate metrics
def process_model(data_path):
    data = pd.read_csv(data_path)
    data = data.dropna(subset=['ndirection', 'pndirection'])
    
    results = []
    for year in range(2016, 2024):
        year_data = data[data['Year'] == year]
        true_labels = year_data['ndirection']
        predicted_labels = year_data['pndirection']
        
        metrics = calculate_metrics(true_labels, predicted_labels)
        mse = mean_squared_error(year_data['nret'], year_data['pnret'])
        results.append([year, *metrics, mse])
    
    # Calculate Overall metrics
    true_labels = data['ndirection']
    predicted_labels = data['pndirection']
    metrics = calculate_metrics(true_labels, predicted_labels)
    mse = mean_squared_error(data['nret'], data['pnret'])
    results.append(['Overall', *metrics, mse])
    
    return pd.DataFrame(results, columns=['Year', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score', 'MSE']), data

# Process all models and save results
for model_path in model_paths:
    pre_trained_data_path = f"C:\\Users\\DA\\Desktop\\wrds\\new-data\\6.15\\{model_path}\\Ticker-Week-ndirection-nret-Year-wscore-pnret-pndirection.csv"
    fine_tuned_data_path = f"{base_path}{model_path}\\test\\Ticker-Year-Week-wheadline-ndirection-nret-fscore-pnret-pndirection16-23.csv"
    
    pre_trained_results, pre_trained_data = process_model(pre_trained_data_path)
    fine_tuned_results, fine_tuned_data = process_model(fine_tuned_data_path)
    
    combined_results = pd.DataFrame()
    combined_results['Year'] = pre_trained_results['Year']
    combined_results[f'{model_path} Accuracy Pre-trained'] = pre_trained_results['Accuracy']
    combined_results[f'{model_path} Precision Pre-trained'] = pre_trained_results['Precision']
    combined_results[f'{model_path} Recall Pre-trained'] = pre_trained_results['Recall']
    combined_results[f'{model_path} Specificity Pre-trained'] = pre_trained_results['Specificity']
    combined_results[f'{model_path} F1 Score Pre-trained'] = pre_trained_results['F1 Score']
    combined_results[f'{model_path} MSE Pre-trained'] = pre_trained_results['MSE']
    
    combined_results[f'{model_path} Accuracy Fine-tuned'] = fine_tuned_results['Accuracy']
    combined_results[f'{model_path} Precision Fine-tuned'] = fine_tuned_results['Precision']
    combined_results[f'{model_path} Recall Fine-tuned'] = fine_tuned_results['Recall']
    combined_results[f'{model_path} Specificity Fine-tuned'] = fine_tuned_results['Specificity']
    combined_results[f'{model_path} F1 Score Fine-tuned'] = fine_tuned_results['F1 Score']
    combined_results[f'{model_path} MSE Fine-tuned'] = fine_tuned_results['MSE']
    
    output_path = f"{base_path}{model_path}\\metrics_comparison.csv"
    combined_results.to_csv(output_path, index=False)
    print(f"Metrics comparison saved to {output_path}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File paths
base_path = "C:\\Users\\DA\\Desktop\\wrds\\new-data\\7.8\\"
benchmark_file_path = "C:\\Users\\DA\\Desktop\\wrds\\new-data\\6.15\\benchmark_with_returns_log_returns.csv"
annual_benchmark_file_path = "C:\\Users\\DA\\Desktop\\wrds\\new-data\\6.15\\benchmark_annual_returns.csv"
cap_weight_file_path = "C:\\Users\\DA\\Desktop\\wrds\\new-data\\Ticker-Year-Cap-weight.csv"
# Load the processed benchmark data
benchmark_data = pd.read_csv(benchmark_file_path)
annual_benchmark_data = pd.read_csv(annual_benchmark_file_path)
cap_weights = pd.read_csv(cap_weight_file_path)

model_paths = ["Bert", "Distilbert", "Finbert", "Distilroberta", "Roberta"]

# Define the investment strategies
def compute_portfolio_returns(predictions, cap_weights, strategy, top_n=5):
    portfolio_returns = []
    for year in range(2016, 2024):
        for week in predictions[predictions['Year'] == year]['Week'].unique():
            weekly_data = predictions[(predictions['Year'] == year) & (predictions['Week'] == week)]
            sorted_data = weekly_data.sort_values(by='pnret', ascending=False)

            if strategy == 'EW_L':
                # Equal Weight Long
                portfolio = sorted_data.head(top_n)
                returns = portfolio['nret'].mean()
            elif strategy == 'EW_S':
                # Equal Weight Short
                portfolio = sorted_data.tail(top_n)
                returns = -portfolio['nret'].mean()  # Short returns are negative
            elif strategy == 'EW_LS':
                # Equal Weight Long-Short
                long_portfolio = sorted_data.head(top_n)
                short_portfolio = sorted_data.tail(top_n)
                returns = long_portfolio['nret'].mean() - short_portfolio['nret'].mean()
            elif strategy == 'VW_L':
                # Value Weight Long
                portfolio = sorted_data.head(top_n)
                merged_portfolio = portfolio.merge(cap_weights, on=['Ticker', 'Year'], how='left')
                returns = (merged_portfolio['nret'] * merged_portfolio['weight']).sum() / merged_portfolio['weight'].sum()
            elif strategy == 'VW_S':
                # Value Weight Short
                portfolio = sorted_data.tail(top_n)
                merged_portfolio = portfolio.merge(cap_weights, on=['Ticker', 'Year'], how='left')
                returns = -(merged_portfolio['nret'] * merged_portfolio['weight']).sum() / merged_portfolio['weight'].sum()  # Short returns are negative
            elif strategy == 'VW_LS':
                # Value Weight Long-Short
                long_portfolio = sorted_data.head(top_n)
                short_portfolio = sorted_data.tail(top_n)
                merged_long_portfolio = long_portfolio.merge(cap_weights, on=['Ticker', 'Year'], how='left')
                merged_short_portfolio = short_portfolio.merge(cap_weights, on=['Ticker', 'Year'], how='left')
                long_returns = (merged_long_portfolio['nret'] * merged_long_portfolio['weight']).sum() / merged_long_portfolio['weight'].sum()
                short_returns = (merged_short_portfolio['nret'] * merged_short_portfolio['weight']).sum() / merged_short_portfolio['weight'].sum()
                returns = long_returns - short_returns
            portfolio_returns.append({'Year': year, 'Week': week, 'Strategy': strategy, 'Returns': returns})
    return portfolio_returns

# Set x-axis labels to show years
year_week_map = {
    2016: 575,
    2017: 627,
    2018: 679,
    2019: 731,
    2020: 783,
    2021: 836,
    2022: 888,
    2023: 940
}
year_labels = list(year_week_map.keys())
week_ticks = list(year_week_map.values())

for model_path in model_paths:
    # Read the prediction file
    prediction_file_path = f"{base_path}{model_path}\\test\\Ticker-Year-Week-wheadline-ndirection-nret-fscore-pnret-pndirection16-23.csv"

    predictions = pd.read_csv(prediction_file_path)

    # Filter data for the years 2016-2023
    predictions = predictions[predictions['Year'].between(2016, 2023)]

    # Calculate returns for six strategies
    strategies = ['EW_L', 'EW_S', 'EW_LS', 'VW_L', 'VW_S', 'VW_LS']
    all_returns = []
    for strategy in strategies:
        all_returns.extend(compute_portfolio_returns(predictions, cap_weights, strategy))

    # Convert to DataFrame and compute cumulative log returns
    returns_df = pd.DataFrame(all_returns)
    returns_df['Log_Returns'] = np.log1p(returns_df['Returns'])
    returns_df['Cumulative_Log_Return'] = returns_df.groupby('Strategy')['Log_Returns'].cumsum()

    # Calculate annual returns for each strategy
    annual_returns = returns_df.groupby(['Year', 'Strategy'])['Returns'].apply(lambda x: np.prod(1 + x) - 1).reset_index()
    annual_returns = annual_returns.rename(columns={'Returns': 'Yret'})

    # Save weekly returns and annual returns to CSV
    output_file_path = f"{base_path}{model_path}\\test\\portfolio_returns_weekly-{model_path}.csv"
    annual_output_file_path = f"{base_path}{model_path}\\test\\portfolio_annual_returns-{model_path}.csv"
    returns_df.to_csv(output_file_path, index=False)
    annual_returns.to_csv(annual_output_file_path, index=False)

    print(f"Weekly portfolio returns saved to {output_file_path}")
    print(f"Annual portfolio returns saved to {annual_output_file_path}")

    # Combine all results
    final_returns_df = pd.concat([returns_df, benchmark_data])

    # Plot cumulative log returns
    plt.figure(figsize=(14, 8))
    for strategy in final_returns_df['Strategy'].unique():
        strategy_data = final_returns_df[final_returns_df['Strategy'] == strategy]
        if 'Market' in strategy:
            plt.plot(strategy_data['Week'], strategy_data['Cumulative_Log_Return'], label=strategy, linestyle='dashed')
        else:
            plt.plot(strategy_data['Week'], strategy_data['Cumulative_Log_Return'], label=strategy)

    plt.xticks(ticks=week_ticks, labels=year_labels)
    plt.xlabel('Year')
    plt.ylabel('Cumulative Log Return')
    plt.title(f'Cumulative Log Return of Different Strategies - {model_path} Fine-tuned')
    plt.legend(title='Strategy')
    plt.grid(True)

    # Save plot
    output_plot_path = f"{base_path}{model_path}\\test\\cumulative_log_return_plot_weekly-{model_path} Fine-tuned.png"
    plt.savefig(output_plot_path, bbox_inches='tight')
    plt.show()

    print(f"Plot saved to {output_plot_path}")

# Annual Sharpe ratio calculation and combining the results for all models
import pandas as pd
import numpy as np

# File paths
base_path = "C:\\Users\\DA\\Desktop\\wrds\\new-data\\7.8\\"

model_paths = ["Bert", "Distilbert", "Finbert", "Distilroberta", "Roberta"]

risk_free_rate = 0.01  # Fixed risk-free rate

# Function to calculate annual Sharpe ratio
def compute_sharpe_ratio(annual_returns, risk_free_rate):
    results = []
    for strategy in annual_returns['Strategy'].unique():
        strategy_data = annual_returns[annual_returns['Strategy'] == strategy]
        avg_yret = strategy_data['Yret'].mean()
        stddev = strategy_data['Yret'].std()
        sharpe_ratio = (avg_yret - risk_free_rate) / stddev
        results.append({'Strategy': strategy, 'AvgYret': avg_yret, 'StdDev': stddev, 'Sharpe ratio': sharpe_ratio})
    return pd.DataFrame(results)

# Loop over each model, compute Sharpe ratios and save results
for model_path in model_paths:
    # Load the annual returns data
    annual_returns_file_path = f"{base_path}{model_path}\\test\\portfolio_annual_returns-{model_path}.csv"
    annual_returns = pd.read_csv(annual_returns_file_path)

    # Calculate Sharpe ratio
    sharpe_ratios_df = compute_sharpe_ratio(annual_returns, risk_free_rate)

    # Save the Sharpe ratio results to a new CSV file
    output_sharpe_file_path = f"{base_path}{model_path}\\test\\Strategy-AvgYret-StdDev-Sharpe ratio-{model_path}.csv"
    sharpe_ratios_df.to_csv(output_sharpe_file_path, index=False)

    print(f"Sharpe ratios saved to {output_sharpe_file_path}")

# Combine Sharpe ratios from all models into one CSV file
import os
import pandas as pd

# Initialize an empty DataFrame to store combined results
combined_df = pd.DataFrame()

# Iterate over each model and load the respective Sharpe ratio file
for model_path in model_paths:
    sharpe_file_path = os.path.join(base_path, model_path, f"Strategy-AvgYret-StdDev-Sharpe ratio-{model_path}.csv")
    print(f"Processing file: {sharpe_file_path}")  # Print the path to verify
    
    try:
        model_df = pd.read_csv(sharpe_file_path)
        
        # Rename columns by adding the model name as a prefix
        model_df.rename(columns={
            'AvgYret': f'{model_path}AvgYret',
            'StdDev': f'{model_path}StdDev',
            'Sharpe ratio': f'{model_path}Sharpe ratio'
        }, inplace=True)
        
        # Combine into the main DataFrame
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

# Save the combined DataFrame to a new CSV file
output_combined_file_path = os.path.join(base_path, "compare", "Strategy-AvgYret-StdDev-Sharpe ratio-all.csv")
os.makedirs(os.path.dirname(output_combined_file_path), exist_ok=True)
combined_df.to_csv(output_combined_file_path, index=False)

print(f"Combined Sharpe ratios saved to {output_combined_file_path}")

