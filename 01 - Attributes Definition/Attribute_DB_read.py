###############################################################
# Attribute_DB_read.py
#
# This is an example script to read and display the contents of the
# Attribute_DB.csv file using pandas.
###############################################################

# Import 3rd party libraries
import pandas as pd

# Data
file_path = '01 - Attributes Definition/Attribute_DB.csv'

# read data from Attribute_DB.csv
indicator_database = pd.read_csv(file_path, delimiter=';')
print(indicator_database.head())
print("\n")

# Number of Indicators
print(f"Total number of indicators: {len(indicator_database)}\n")

# List all values in High-Level Group (if there are multiple entries, they are separated by comma)
high_level_groups = set(
    group.strip()
    for entry in indicator_database['High-Level Group'].dropna()
    for group in entry.split(',')
)
print("High-Level Groups:")
print("\n ".join(high_level_groups))

# Print all entries of one High-Level Group
selected_group = 'Nuclear Design'
print(f"\nEntries in High-Level Group: {selected_group}")
group_entries = indicator_database[
    indicator_database['High-Level Group'].str.contains(selected_group, na=False)
]
print(group_entries)