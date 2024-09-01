# Process the downloaded stock return data and text data
#Merge the datasets

import pandas as pd
import numpy as np

# Load the daily returns data
daily_returns = pd.read_csv(r"C:\Users\DA\Desktop\wrds\new-data\stock-download.csv")
daily_returns['DlyCalDt'] = pd.to_datetime(daily_returns['DlyCalDt'])

# Check for missing values and duplicates
print("Missing values in daily returns data:", daily_returns.isnull().sum().sum())
daily_returns.drop_duplicates(inplace=True)

# Load the week ranges data
week_ranges = pd.read_csv(r"C:\Users\DA\Desktop\wrds\new-data\merged_05-23week.csv")
week_ranges['Start Date'] = pd.to_datetime(week_ranges['Start Date'])
week_ranges['End Date'] = pd.to_datetime(week_ranges['End Date'])

# Assign week and year columns to daily returns based on the date ranges
def assign_week_and_year(date):
    for index, row in week_ranges.iterrows():
        if row['Start Date'] <= date <= row['End Date']:
            return row['Week'], row['Start Date'].year
    return None, None

daily_returns['Week'], daily_returns['Year'] = zip(*daily_returns['DlyCalDt'].apply(assign_week_and_year))

# Calculate weekly returns using the given formula
def calculate_weekly_return(group):
    return np.exp(np.sum(np.log(1 + group))) - 1

weekly_returns = daily_returns.groupby(['PERMNO', 'Week', 'Year'])['DlyRet'].apply(calculate_weekly_return).reset_index()
weekly_returns.rename(columns={'DlyRet': 'WeeklyRet'}, inplace=True)

# Create Direction variable
weekly_returns['Direction'] = (weekly_returns['WeeklyRet'] > 0).astype(int)

# Generate a new table
stock_return_table = weekly_returns[['PERMNO', 'Week', 'Year', 'WeeklyRet', 'Direction']]

# Save the stock return table
stock_return_table.to_csv(r'C:\Users\DA\Desktop\wrds\new-data\stock_return_table.csv', index=False)

# Process the downloaded text data
text_data = pd.read_csv(r"C:\Users\DA\Desktop\wrds\new-data\headline-download.csv")
text_data['announcedate'] = pd.to_datetime(text_data['announcedate'])

# Check for missing values and duplicates
print("Missing values in text data:", text_data.isnull().sum().sum())
text_data.drop_duplicates(inplace=True)

# Assign week and year columns to text data based on the date ranges
text_data['Week'], text_data['Year'] = zip(*text_data['announcedate'].apply(assign_week_and_year))

# Generate a new table
text_data_table = text_data[['GVKEY', 'Week', 'Year', 'keydevid', 'headline']]

# Save the text data table
text_data_table.to_csv(r'C:\Users\DA\Desktop\wrds\new-data\text_data_table.csv', index=False)

# Merge stock return data and text data
# Define file paths
gvkey_permno_file = r"C:\Users\DA\Desktop\wrds\new-data\GVKEY-PERMNO.csv"
stock_return_table_file = r"C:\Users\DA\Desktop\wrds\new-data\stock_return_table.csv"
text_data_table_file = r"C:\Users\DA\Desktop\wrds\new-data\text_data_table.csv"

# Read CSV files
gvkey_permno_df = pd.read_csv(gvkey_permno_file)
stock_return_table = pd.read_csv(stock_return_table_file)
text_data_table = pd.read_csv(text_data_table_file)

# Merge GVKEY with stock return table
stock_return_table = pd.merge(stock_return_table, gvkey_permno_df, on='PERMNO', how='left')
stock_return_table.drop(columns=['PERMNO'], inplace=True)
stock_return_table.rename(columns={'GVKEY': 'PERMNO'}, inplace=True)

# Merge stock return data with text data
merged_data = pd.merge(stock_return_table, text_data_table, on=['PERMNO', 'Week', 'Year'], how='left')

# Filter the columns to keep
filtered_columns = ['Ticker', 'Week', 'WeeklyRet', 'Direction', 'keydevid', 'headline', 'Year']
filtered_data = merged_data[filtered_columns]

# Save the merged data
output_file_path = r'C:\Users\DA\Desktop\wrds\Ticker-Week-WeeklyRet-direction-keydevid-headline-Year.csv'
filtered_data.to_csv(output_file_path, index=False)








#Create dataset that includes Ticker, Week, Year, w-Headline, n-Ret, and n-Direction

import pandas as pd

# Load the input CSV file
input_file_path = r'C:\Users\DA\Desktop\wrds\Ticker-Week-WeeklyRet-direction-keydevid-headline-Year.csv'
df = pd.read_csv(input_file_path)

# Sort the dataframe by Ticker and Week
df.sort_values(by=['Ticker', 'Week'], inplace=True)

# Combine headlines by Ticker and Week into a single column and fill missing headlines with an empty string
df_combined = df.groupby(['Ticker', 'Year', 'Week'], as_index=False).agg({
    'headline': lambda x: ' '.join(x) if len(x) > 0 else '',  # Combine all headlines into a single string or use an empty string if there are no headlines
    'WeeklyRet': 'first',               # Just take the first occurrence (as all values should be the same for the same Week)
    'direction': 'first',               # Same for direction
    'keydevid': 'first'                 # Same for keydevid
})

# Create the ndirection and nret columns by shifting within each Ticker group
df_combined['nret'] = df_combined.groupby('Ticker')['WeeklyRet'].shift(-1)
df_combined['ndirection'] = df_combined.groupby('Ticker')['direction'].shift(-1)

# Drop the last Week for each Ticker where nret and ndirection would be NaN
df_combined.dropna(subset=['nret', 'ndirection'], inplace=True)

# Rename the headline column to wheadline
df_combined.rename(columns={'headline': 'wheadline'}, inplace=True)

# Save the result to a new CSV file
output_file_path = r"C:\Users\DA\Desktop\wrds\Ticker-Year-Week-wheadline-ndirection-nret.csv"
df_combined.to_csv(output_file_path, index=False)
