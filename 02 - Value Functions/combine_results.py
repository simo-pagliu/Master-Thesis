# USER INPUT: Folders to look for partial results
common_folders = ["simple"]
specific_folders = ["ghg"]

# FIRST PART: Combine results from common_folders
import os
import re
import pandas as pd

# Directory of this script; aggregated outputs will live under
# <script_dir>/aggregatedresults
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AGG_DIR = os.path.join(SCRIPT_DIR, 'aggregatedresults')

# Each folder contains a criteria.csv and value_functions.csv file
# Load and concat all common folders once (assume at least one common folder exists)
combined_common_criteria = []
combined_common_value_functions = []
for folder in common_folders:
    criteria_path = os.path.join(SCRIPT_DIR, folder, 'criteria.csv')
    value_functions_path = os.path.join(SCRIPT_DIR, folder, 'value_functions.csv')
    combined_common_criteria.append(pd.read_csv(criteria_path))
    combined_common_value_functions.append(pd.read_csv(value_functions_path))

# Concatenate collected common DataFrames (no need to check emptiness)
combined_common_criteria_df = pd.concat(combined_common_criteria, ignore_index=True)
combined_common_value_functions_df = pd.concat(combined_common_value_functions, ignore_index=True)

# SECOND PART: Read results from all specific_folders and aggregate per country
# We expect criteria 'group' values to contain a '-COUNTRY' suffix (e.g. GORPUCODE-COUNTRY)
# Value functions also may carry the same `group` convention.

# Collect country-specific rows across all specific_folders
country_criteria = {}
country_value_functions = {}
if specific_folders:
    for folder in specific_folders:
        criteria_path = os.path.join(SCRIPT_DIR, folder, 'criteria.csv')
        value_functions_path = os.path.join(SCRIPT_DIR, folder, 'value_functions.csv')
        if os.path.exists(criteria_path):
            df_criteria = pd.read_csv(criteria_path)
            for _, row in df_criteria.iterrows():
                group = str(row.get('group', ''))
                if '-' in group:
                    country_code = group.split('-')[-1].strip()
                    country_criteria.setdefault(country_code, []).append(row)
                else:
                    country_criteria.setdefault('GLOBAL', []).append(row)
        # value functions may or may not exist; skip silently if missing
        if os.path.exists(value_functions_path):
            df_value_functions_all = pd.read_csv(value_functions_path)
            for _, row in df_value_functions_all.iterrows():
                group = str(row.get('group', '')) if 'group' in df_value_functions_all.columns else ''
                if group and '-' in group:
                    country_code = group.split('-')[-1].strip()
                    country_value_functions.setdefault(country_code, []).append(row)
                else:
                    country_value_functions.setdefault('GLOBAL', []).append(row)

# Determine all country codes found across specific folders (exclude GLOBAL)
country_codes = set(country_criteria.keys()) | set(country_value_functions.keys())
country_codes.discard('GLOBAL')

# Ensure aggregated output dir exists
os.makedirs(AGG_DIR, exist_ok=True)

if not country_codes:
    # No specific country codes found: write single pair of aggregated files at AGG_DIR
    combined_common_criteria_df.to_csv(os.path.join(AGG_DIR, 'criteria.csv'), index=False)
    combined_common_value_functions_df.to_csv(os.path.join(AGG_DIR, 'value_functions.csv'), index=False)
else:
    # For each country, create a single aggregated result under aggregatedresults/<COUNTRY>/
    for country_code in sorted(country_codes):
        out_dir = os.path.join(AGG_DIR, country_code)
        os.makedirs(out_dir, exist_ok=True)

        # Build criteria: common + all specific rows for this country + GLOBAL rows
        criteria_parts = [combined_common_criteria_df, pd.DataFrame(country_criteria.get(country_code, [])), pd.DataFrame(country_criteria.get('GLOBAL', []))]
        country_criteria_df = pd.concat(criteria_parts, ignore_index=True)

        # Strip "-COUNTRY" suffix for this country in group column
        suffix_pattern = re.compile(r"-" + re.escape(country_code) + r"$")
        if 'group' in country_criteria_df.columns:
            country_criteria_df['group'] = country_criteria_df['group'].astype(str).apply(lambda g: suffix_pattern.sub('', g))

        country_criteria_df.to_csv(os.path.join(out_dir, 'criteria.csv'), index=False)

        # Build value functions: common + country-specific + matched GLOBALs
        vf_parts = [combined_common_value_functions_df, pd.DataFrame(country_value_functions.get(country_code, []))]

        # Include GLOBAL VFs that reference criteria names
        global_vf_df = pd.DataFrame(country_value_functions.get('GLOBAL', []))
        if not global_vf_df.empty and 'name' in country_criteria_df.columns:
            names_set = set(country_criteria_df['name'].astype(str).tolist())
            matched_global_vf = global_vf_df[global_vf_df['name'].astype(str).isin(names_set)]
            if not matched_global_vf.empty:
                vf_parts.append(matched_global_vf)

        country_value_functions_df = pd.concat(vf_parts, ignore_index=True)

        # Strip "-COUNTRY" suffix in value functions group column as well
        if 'group' in country_value_functions_df.columns:
            country_value_functions_df['group'] = country_value_functions_df['group'].astype(str).apply(lambda g: suffix_pattern.sub('', g))

        country_value_functions_df.to_csv(os.path.join(out_dir, 'value_functions.csv'), index=False)