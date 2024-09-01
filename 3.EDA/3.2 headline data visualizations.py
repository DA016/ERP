# headline data visualizations

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.ticker import LogLocator, ScalarFormatter

# Read data
output_file_path = r"C:\Users\DA\Desktop\wrds\new-data\keydevid-headline-keydeveventtypeid-announcedate-Ticker.csv"
df = pd.read_csv(output_file_path)

# Extract year from the announcedate column
df['year'] = pd.to_datetime(df['announcedate']).dt.year

# Calculate the number of headlines per year for each Ticker
ticker_year_counts = df.groupby(['year', 'Ticker']).size().unstack(fill_value=0)

# Use TABLEAU_COLORS to generate 25 different colors
color_list = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173"
]

# Plot the number of headlines per year for each Ticker
plt.figure(figsize=(14, 8), dpi=300)
for i, ticker in enumerate(ticker_year_counts.columns):
    plt.plot(ticker_year_counts.index, ticker_year_counts[ticker], label=ticker, color=color_list[i], linewidth=1.5)

plt.xlabel('Year')
plt.ylabel('Total Number of Headlines')
plt.title('Total Number of Headlines by Ticker and Year')
plt.legend(loc='upper left', fontsize='small', ncol=2)
plt.grid(True)
plt.yscale('log')  # Display y-axis with a logarithmic scale
plt.xticks(range(2005, 2024, 1))  # Set x-axis labels to show years from 2005 to 2023

# Configure detailed labels for the logarithmic scale
ax = plt.gca()
ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_minor_formatter(ScalarFormatter())
ax.yaxis.set_tick_params(which='both', right=False)

# Save the image
output_image_path = r"C:\Users\DA\Desktop\wrds\new-data\headlines_by_ticker_and_year_detailed.png"
plt.savefig(output_image_path, bbox_inches='tight')

# Display the plot
plt.show()

print(f"Image has been saved to {output_image_path}")

# Second part of the script

import pandas as pd
import matplotlib.pyplot as plt

# Read data
output_file_path = r"C:\Users\DA\Desktop\wrds\new-data\keydevid-headline-keydeveventtypeid-announcedate-Ticker.csv"
df = pd.read_csv(output_file_path)

# Ensure the announcedate column is in datetime format
df['announcedate'] = pd.to_datetime(df['announcedate'], errors='coerce')

# Extract year and month information
df['year'] = df['announcedate'].dt.year
df['month'] = df['announcedate'].dt.month

# Calculate the total number of headlines per year
yearly_counts = df.groupby('year').size()

# Calculate the average number of headlines per day, divided by the number of unique years
daily_avg_counts = df.groupby(df['announcedate'].dt.strftime('%m-%d')).size() / df['year'].nunique()

# Compute the average values
average_yearly_counts = yearly_counts.mean()
average_daily_counts = daily_avg_counts.mean()

# Create the figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot the total number of headlines per year
ax1.bar(yearly_counts.index, yearly_counts.values, color='skyblue')
ax1.axhline(y=average_yearly_counts, color='dodgerblue', linestyle='--')
ax1.text(yearly_counts.index[-1], average_yearly_counts + 100, f'Average = {average_yearly_counts:.0f}', color='dodgerblue', ha='right')
ax1.set_xlabel('Year')
ax1.set_ylabel('Total Number of Headlines')
ax1.set_title('Total Number of Headlines by Year')
ax1.set_xticks(range(2005, 2024, 1))  # Set x-axis labels to show years from 2005 to 2023
ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgrey')

# Plot the average number of headlines per day, with months on the x-axis
ax2.bar(daily_avg_counts.index, daily_avg_counts.values, color='lightblue')
ax2.axhline(y=average_daily_counts, color='dodgerblue', linestyle='--')
ax2.text(len(daily_avg_counts) - 1, average_daily_counts + 0.5, f'Average = {average_daily_counts:.1f}', color='dodgerblue', ha='right')
ax2.set_xlabel('Month')
ax2.set_ylabel('Average Number of Headlines')
ax2.set_title('Average Number of Headlines by Day of Year')
ax2.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgrey')

# Set the first day of each month as x-axis labels
month_starts = [f"{month:02}-01" for month in range(1, 13)]
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax2.set_xticks(month_starts)
ax2.set_xticklabels(month_labels)

# Save the image
output_image_path = r"C:\Users\DA\Desktop\wrds\new-data\7.26\Headlines_by_year_and_day_of_year3.png"
plt.savefig(output_image_path, dpi=300, bbox_inches='tight')

# Display the plot
plt.show()

print(f"Image has been saved to {output_image_path}")






import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
file_path = r"C:\Users\DA\Desktop\wrds\new-data\7.26\keydeveventtypeid-count-eventtype new data.csv"
data = pd.read_csv(file_path)

# Sort the data by count in descending order and select the top 20 event types
top_20_events = data.sort_values(by='count', ascending=False).head(20)

# Plotting the data with a different color palette
plt.figure(figsize=(12, 8))
sns.barplot(x='count', y='eventtype', data=top_20_events, palette='Blues_d')
plt.title('Top 20 Distribution of Event Types')
plt.xlabel('Event Count')
plt.ylabel('Event Type')
plt.tight_layout()

# Save the plot to the same folder
output_file_path = r"C:\Users\DA\Desktop\wrds\new-data\7.26\top_20_eventtypes_distribution2.png"
plt.savefig(output_file_path, dpi=300)

# Show the plot
plt.show()

