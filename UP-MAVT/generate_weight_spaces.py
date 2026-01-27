#!/usr/bin/env python3
#################################################################################
# Generate weight spaces for all countries in parallel
#
# Simone Pagliuca, 2025-2026
#
# Description:
# This script generates weight space files for all 4 countries (IT, FR, CH, PO)
# in parallel to avoid manual execution and stopping of MC simulations.
#################################################################################

import os
import sys
from multiprocessing import Pool
import numpy as np

# Ensure all relative file accesses resolve relative to this script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

from auxiliary import (
    load_criteria_file,
    verify_criteria_consistency,
    combine_alternatives_by_country,
    convert_qualitative_indicators_in_folders,
    remap_bwt_results_for_country,
    startup,
)
from pile_bwt import bwt, constraints_func, define_weight_spaces

#################################################################################
# USER INPUTS
# Countries to process
COUNTRIES = ["IT", "FR", "CH", "PO"]
COUNTRIES = ["CH"]  # TESTING ONLY


# Elicitation run numbers (folders containing results)
elicitation_numbers = [1, 2, 4, 9]
elicitation_numbers = [9]  # TESTING ONLY

# Folders Containing quick elicitation results (only QIs)
QI_elicitation_numbers = [3, 5, 6, 7, 8]

# Weight space generation parameters
required_weight_solutions = 1  # Target number of unique weight combinations

# Number of parallel workers (None = use all available cores)
n_workers = None
#################################################################################


#################################################################################


def process_country(country):
    """Process a single country: setup data and generate weight spaces."""
    print(f"\n{'='*60}")
    print(f"Processing country: {country}")
    print(f"{'='*60}")
    
    try:
        # NOTE: Automatic conversion disabled - criteria.csv must be manually prepared
        folders_to_convert = sorted(set(elicitation_numbers + (QI_elicitation_numbers or [])))
        if folders_to_convert:
            print(f"⚠ Warning: Ensure qualitative indicators are properly converted in: {folders_to_convert}")
            print(f"⚠ Warning: criteria.csv files should be manually prepared if needed")
            # convert_qualitative_indicators_in_folders(folders_to_convert)  # Disabled

        # Load and verify criteria from elicitation results
        elicitation_criteria_paths = []
        weight_elicitations = []
        value_functions = []

        for elicit_num in elicitation_numbers:
            elicit_dir = os.path.join(SCRIPT_DIR, "elicitation_results", str(elicit_num))
            
            # Criteria file path
            crit_path = os.path.join(elicit_dir, "criteria.csv")
            if not os.path.exists(crit_path):
                raise FileNotFoundError(f"Criteria file not found: {crit_path}")
            elicitation_criteria_paths.append(crit_path)
            
            # Weight elicitation file (BWT results): use country-adjusted comparisons when possible
            weight_file = remap_bwt_results_for_country(elicit_num, country, script_dir=SCRIPT_DIR)
            if weight_file is None:
                raise ValueError(f"Could not find BWT results for elicitation {elicit_num} and country {country}")
            weight_elicitations.append(weight_file)
            
            # Value functions file for selected country
            vf_file = os.path.join(elicit_dir, country, "value_functions.csv")
            if not os.path.exists(vf_file):
                raise FileNotFoundError(f"Value functions file not found: {vf_file}")
            value_functions.append(vf_file)

        print(f"Loading elicitation results for country: {country}")
        print(f"Elicitations: {elicitation_numbers}")

        # Load and verify all criteria are the same
        print("Loading and verifying criteria files...")
        criteria_dfs = [load_criteria_file(path) for path in elicitation_criteria_paths]
        criteria_verified = verify_criteria_consistency(criteria_dfs)
        print(f"✓ All criteria files are consistent. Using common criteria with {len(criteria_verified)} criteria.")

        # Use or create criteria.csv in SCRIPT_DIR
        file_path_criteria = os.path.join(SCRIPT_DIR, "criteria.csv")
        if os.path.exists(file_path_criteria):
            print(f"⚠ Using existing criteria.csv (not overwriting manual changes)")
        else:
            print(f"Creating new criteria.csv from elicitation data")
            criteria_verified.to_csv(file_path_criteria, index=False)

        print(f"Combining alternatives for {country} from {len(elicitation_numbers)} elicitation(s)...")
        elicitation_dirs = [os.path.join(SCRIPT_DIR, "elicitation_results", str(num)) for num in elicitation_numbers]
        qi_elicitation_dirs = [os.path.join(SCRIPT_DIR, "elicitation_results", str(num)) for num in QI_elicitation_numbers] if QI_elicitation_numbers else None
        combine_alternatives_by_country(elicitation_dirs, country, SCRIPT_DIR, qi_elicitation_dirs=qi_elicitation_dirs)

        # Use country-specific combined alternatives file
        file_path_alternatives = os.path.join(SCRIPT_DIR, f"alternatives_{country}.csv")
        if not os.path.exists(file_path_alternatives):
            raise FileNotFoundError(f"Alternative file not found after combining: {file_path_alternatives}")

        # Startup: Load all data
        dict_data_list, crit_index, vf_list, conf_list, alternatives = startup(
            file_path_criteria, weight_elicitations, value_functions, file_path_alternatives
        )
        n_alternatives = len(alternatives)
        print(f"Loaded {len(dict_data_list)} elicitation(s) with {n_alternatives} alternatives.")

        # PILE-BWT Method + Weight Space Definition
        print(f"Running BWT for {country} and defining weight spaces...")
        bwt_results = []

        for i, dict_data in enumerate(dict_data_list):
            print(f"  [{country}] Running BWT for elicitation {elicitation_numbers[i]}...")
            bwt_result = bwt(dict_data)
            bwt_results.append(bwt_result)

        print(f"✓ BWT solver result: {bwt_results[-1]['solver_result']['x']}")
        constraint_value = constraints_func(bwt_results[-1]["solver_result"]["x"], dict_data_list[-1])
        print(f"✓ Constraint values for last BWT result: {constraint_value}")

        # Define or load weight spaces using constraint-based optimization
        list_of_weight_space_points = define_weight_spaces(
            dict_data_list, 
            elicitation_numbers, 
            SCRIPT_DIR, 
            required_solutions=required_weight_solutions,
            country=country
        )

        print(f"✓ Generated weight spaces for {len(list_of_weight_space_points)} elicitations.")
        print(f"✓ Successfully completed for country: {country}")
        return country, True, None

    except Exception as e:
        print(f"✗ Error processing country {country}: {e}")
        import traceback
        traceback.print_exc()
        return country, False, str(e)


def main():
    """Main function to process all countries in parallel."""
    print("\n" + "="*60)
    print("Weight Space Generation for All Countries")
    print("="*60)
    print(f"Countries: {COUNTRIES}")
    print(f"Elicitations: {elicitation_numbers}")
    print(f"Number of workers: {n_workers or 'all available cores'}")
    print("="*60 + "\n")

    # Process countries in parallel
    with Pool(processes=n_workers) as pool:
        results = pool.map(process_country, COUNTRIES)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for country, success, error in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{country}: {status}")
        if error:
            print(f"  Error: {error}")
    print("="*60 + "\n")

    # Return exit code
    all_success = all(success for _, success, _ in results)
    return 0 if all_success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
