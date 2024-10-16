import wrds
import pandas as pd

# Connect to WRDS
db = wrds.Connection()

# Define the start and end date for the data
start_date = '2024-01-01'
end_date = '2024-10-15'

# Query to find Russell 2000 Index from CRSP MSI dataset
query = f"""
SELECT caldt AS date, vwretd AS return
FROM crsp.msi
WHERE caldt >= '{start_date}' 
AND caldt <= '{end_date}' 
AND indxnam = 'Russell 2000';
"""

# Execute the query and retrieve the data
russell_2000_data = db.raw_sql(query)

# Close WRDS connection
db.close()

# Save the data to a CSV file
russell_2000_data.to_csv('russell_2000_returns_2024.csv', index=False)

# Display the first few rows of the data
print(russell_2000_data.head())


