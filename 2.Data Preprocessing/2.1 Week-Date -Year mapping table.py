
#Create a Week-Date Range-Year mapping table


import csv
from datetime import datetime, timedelta

# Define start and end dates
start_date = datetime(2005, 1, 1)
end_date = datetime(2023, 12, 31)

# Create a CSV file and write the header
output_file_path = r"C:\Users\DA\Desktop\wrds\new-data\merged_05-23week.csv"
with open(output_file_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Week", "Start Date", "End Date", "Year"])

    # Calculate the start and end dates of each week and write to the CSV file
    current_date = start_date
    week_number = 1
    while current_date <= end_date:
        # Find the first day of the current week (Monday)
        while current_date.weekday() != 0:
            current_date += timedelta(days=1)
        
        # Find the last day of the current week (Sunday)
        end_of_week = current_date + timedelta(days=6)
        
        # Determine the year based on which year contains more days of the week
        start_year = current_date.year
        end_year = end_of_week.year
        if start_year == end_year:
            week_year = start_year
        else:
            start_days = (datetime(start_year + 1, 1, 1) - current_date).days
            end_days = (end_of_week - datetime(end_year, 1, 1)).days + 1
            week_year = start_year if start_days > end_days else end_year
        
        # Write the information of the current week to the CSV file
        writer.writerow([week_number, current_date.strftime("%Y/%m/%d"), end_of_week.strftime("%Y/%m/%d"), week_year])
        
        # Update the date and week number
        current_date += timedelta(days=7)
        week_number += 1

print(f"Week-Date Range-Year mapping table has been saved to {output_file_path}")
