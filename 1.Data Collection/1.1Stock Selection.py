#Stock Selection

#(Main steps of Stock Selection are done through WRDS website and Excel. See details in additional material document  or Chapter 3 of dissertation)


#LPERMNO in CRSP/Compustat Merged Database is the same as PERMNO in CRSP. To simplify the expression, they are uniformly referred to as "PERMNO" in report. 


import pandas as pd

# File paths
input_file_path = r"C:\Users\DA\Desktop\wrds\new-data\GSECTOR=45.csv"
output_file_path = r"C:\Users\DA\Desktop\wrds\new-data\Top25_LPERMNOs.csv"

# Read the input data
data = pd.read_csv(input_file_path)

# Convert DlyCalDt to datetime format
data['DlyCalDt'] = pd.to_datetime(data['DlyCalDt'])

# Ensure DlyCap column is numeric
data['DlyCap'] = pd.to_numeric(data['DlyCap'], errors='coerce')

# Filter data for the period 2005-2015
data_filtered = data[(data['DlyCalDt'] >= '2005-01-01') & (data['DlyCalDt'] <= '2015-12-31')]

# Calculate mean DlyCap for the period 2005-2015 for each LPERMNO
mean_dlycap_2005_2015 = data_filtered.groupby('LPERMNO')['DlyCap'].mean().reset_index()
mean_dlycap_2005_2015.columns = ['LPERMNO', 'Mean_DlyCap_2005_2015']

# Sort LPERMNOs by mean DlyCap during 2005-2015
sorted_permnos = mean_dlycap_2005_2015.sort_values(by='Mean_DlyCap_2005_2015', ascending=False)

# Extract the top 25 LPERMNOs
top_25_permnos = sorted_permnos.head(25)

# Save the top 25 LPERMNOs to a new CSV file
top_25_permnos.to_csv(output_file_path, index=False)

print(f"Top 25 LPERMNOs with highest mean DlyCap during 2005-2015 have been saved to {output_file_path}")
