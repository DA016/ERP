# stock return data visualizations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read data
file_path = r"C:\Users\DA\Desktop\wrds\new-data\Ticker-Week-WeeklyRet-direction-keydevid-headline-Year.csv"
df = pd.read_csv(file_path)

# Convert WeeklyRet to percentage
df['WeeklyRet'] = df['WeeklyRet'] * 100

# Calculate yearly average of WeeklyRet
yearly_avg = df.groupby(['Year', 'Ticker'])['WeeklyRet'].mean().unstack()

# Plot yearly average line plot
plt.figure(figsize=(14, 10))
palette = sns.color_palette("husl", 25)
yearly_avg.plot(ax=plt.gca(), colormap='tab20c')
plt.title('Average Weekly Returns by Year (%)')
plt.xlabel('Year')
plt.ylabel('Average Weekly Return (%)')
plt.legend(title='Ticker', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(ticks=range(df['Year'].min(), df['Year'].max() + 1), rotation=45)  # Set x-axis to show all years in data
plt.tight_layout()
lineplot_path = r"C:\Users\DA\Desktop\wrds\new-data\7.26\average_weekly_returns-by-year_lineplot.png"
plt.savefig(lineplot_path, dpi=300)
plt.show()

# Calculate descriptive statistics for continuous returns
continuous_stats = df.groupby("Ticker")["WeeklyRet"].describe()

# Calculate frequency statistics for discrete returns (direction)
discrete_stats = df.groupby("Ticker")["direction"].value_counts().unstack(fill_value=0)

# Save descriptive statistics to CSV files
continuous_stats_path = r"C:\Users\DA\Desktop\wrds\new-data\6.15\Ticker-Week-WeeklyRet-direction-keydevid-descriptive.csv"
discrete_stats_path = r"C:\Users\DA\Desktop\wrds\new-data\6.15\Ticker-Week-WeeklyRet-direction-keydevid-descriptive2.csv"
continuous_stats.to_csv(continuous_stats_path)
discrete_stats.to_csv(discrete_stats_path)

from tabulate import tabulate

# Generate LaTeX table for descriptive statistics
latex_table = tabulate(continuous_stats, headers='keys', tablefmt='latex', floatfmt='.4f')
with open(r"C:\Users\DA\Desktop\wrds\new-data\6.15\continuous_stats_table.tex", 'w') as f:
    f.write(latex_table)

# Create a heatmap for the discrete stats
plt.figure(figsize=(12, 8))
sns.heatmap(discrete_stats, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Frequency of Weekly Return Directions by Ticker')
plt.xlabel('Direction')
plt.ylabel('Ticker')
heatmap_path = r"C:\Users\DA\Desktop\wrds\new-data\7.26\discrete_stats_heatmap.png"
plt.savefig(heatmap_path, dpi=300)
plt.show()
