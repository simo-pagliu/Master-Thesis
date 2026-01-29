#################################################################################
# Main script to run UP-MAVT analysis
#
# Simone Pagliuca, 2025-2026
#
# Description:
# TBC....
#################################################################################

#################################################################################
# Import third party libraries
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import os
import pandas as pd
import csv
from datetime import datetime
#################################################################################

#################################################################################
# Import internal modules
from pile_bwt import bwt, constraints_func, define_weight_spaces
from up_mavt import mc_simulation
from aggregation_methods import weighted_sum, harmonic_mean, geometric_mean
from auxiliary import (
    startup,
    combine_alternatives_by_country,
    load_criteria_file,
    verify_criteria_consistency,
    convert_qualitative_indicators_in_folders,
    remap_bwt_results_for_country
)
from plotting import create_heatmap, create_histograms, update_plots, finalize_layout
#################################################################################

# Ensure all relative file accesses resolve relative to this script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

#################################################################################
# USER INPUTS - Base Configuration
# Base country (used for Phase 1)
base_country = "IT"

# Elicitation run numbers (folders containing results)
elicitation_numbers = [1, 2, 4, 9]

# Folders Containing quick elicitation results (only QIs)
QI_elicitation_numbers = [3, 5, 6, 7, 8]

# Weight space generation parameters
required_weight_solutions = 1  # Target number of unique weight combinations

# Montecarlo Parameters
n_runs = 10000  # Number of Montecarlo simulation runs
plot_bins = 100  # Number of bins for histograms
opinion_weights = np.ones(len(elicitation_numbers))/len(elicitation_numbers)  # Equal weights for each elicitation

# Countries to analyze (Phases 2 and 3)
countries_to_analyze = ["IT", "FR", "CH", "PO"]

# Aggregation methods to cycle through
aggregation_methods = [
    ("weighted_sum", weighted_sum),
    ("harmonic_mean", harmonic_mean),
    ("geometric_mean", geometric_mean),
]
#################################################################################

#################################################################################
# Helper function to save plots as PDF
def save_figures_as_pdf(fig, fig_hist, phase_name, country, method_name=None):
    """Save heatmap and histograms as PDFs with descriptive names."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_dir = "./results/pdfs"
    os.makedirs(pdf_dir, exist_ok=True)
    
    if method_name:
        base_name = f"{phase_name}_{country}_{method_name}_{timestamp}"
    else:
        base_name = f"{phase_name}_{country}_{timestamp}"
    
    if fig:
        pdf_path = os.path.join(pdf_dir, f"{base_name}_heatmap.pdf")
        fig.savefig(pdf_path, format='pdf', dpi=150, bbox_inches='tight')
        print(f"  → Saved heatmap: {pdf_path}")
    
    if fig_hist:
        pdf_path = os.path.join(pdf_dir, f"{base_name}_histograms.pdf")
        fig_hist.savefig(pdf_path, format='pdf', dpi=150, bbox_inches='tight')
        print(f"  → Saved histograms: {pdf_path}")

#################################################################################
# Main simulation function
def run_monte_carlo_simulation(
    selected_country,
    aggregation_method,
    strict_mode,
    random_weight_analysis,
    phase_name="phase",
    method_name=None,
    UPDATE_EVERY=10000,
):
    """
    Run a complete Monte Carlo simulation with the specified configuration.
    
    Parameters:
    - selected_country: Country code (IT, FR, CH, PO)
    - aggregation_method: Function to use for aggregation (weighted_sum, harmonic_mean, geometric_mean)
    - strict_mode: Boolean for STRICT mode
    - random_weight_analysis: Boolean for random weight analysis
    - phase_name: Identifier for the phase (used in filenames)
    - method_name: Name of aggregation method (used in filenames)
    - UPDATE_EVERY: Update frequency for plots
    """
    print(f"\n{'='*80}")
    print(f"Phase: {phase_name} | Country: {selected_country} | Strict: {strict_mode}")
    if method_name:
        print(f"Aggregation Method: {method_name} | Random Weights: {random_weight_analysis}")
    print(f"{'='*80}\n")
    
    #################################################################################
    # Load and verify criteria from elicitation results
    elicitation_criteria_paths = []
    weight_elicitations = []
    value_functions = []

    for elicit_num in elicitation_numbers:
        elicit_dir = os.path.join(SCRIPT_DIR, "elicitation_results", str(elicit_num))
        
        crit_path = os.path.join(elicit_dir, "criteria.csv")
        elicitation_criteria_paths.append(crit_path)
        
        weight_file = remap_bwt_results_for_country(elicit_num, selected_country, script_dir=SCRIPT_DIR)
        weight_elicitations.append(weight_file)
        
        vf_file = os.path.join(elicit_dir, selected_country, "value_functions.csv")
        value_functions.append(vf_file)

    print(f"Loading elicitation results for country: {selected_country}")
    print(f"Elicitations: {elicitation_numbers}")

    # Load and verify criteria
    print("Loading and verifying criteria files...")
    criteria_dfs = [load_criteria_file(path) for path in elicitation_criteria_paths]
    criteria_verified = verify_criteria_consistency(criteria_dfs)
    print(f"✓ All criteria files are consistent. Using common criteria with {len(criteria_verified)} criteria.")

    file_path_criteria = os.path.join(SCRIPT_DIR, "criteria.csv")
    if os.path.exists(file_path_criteria):
        print(f"⚠ Using existing criteria.csv (not overwriting manual changes)")
    else:
        print(f"Creating new criteria.csv from elicitation data")
        criteria_verified.to_csv(file_path_criteria, index=False)

    print(f"Combining alternatives for {selected_country} from {len(elicitation_numbers)} elicitation(s)...")
    elicitation_dirs = [os.path.join(SCRIPT_DIR, "elicitation_results", str(num)) for num in elicitation_numbers]
    qi_elicitation_dirs = [os.path.join(SCRIPT_DIR, "elicitation_results", str(num)) for num in QI_elicitation_numbers] if QI_elicitation_numbers else None
    combine_alternatives_by_country(elicitation_dirs, selected_country, SCRIPT_DIR, qi_elicitation_dirs=qi_elicitation_dirs)

    file_path_alternatives = os.path.join(SCRIPT_DIR, f"alternatives_{selected_country}.csv")

    # Startup: Load all data
    dict_data_list, crit_index, vf_list, conf_list, alternatives = startup(file_path_criteria, weight_elicitations, value_functions, file_path_alternatives)
    n_alternatives = len(alternatives)
    print(f"Loaded {len(dict_data_list)} elicitation(s) with {n_alternatives} alternatives.")

    # Extract alternative names
    alternative_names = pd.read_csv(file_path_alternatives)['name'].tolist()
    #################################################################################

    #################################################################################
    # PILE-BWT Method + Weight Space Definition
    print("Running BWT for each elicitation and defining weight spaces...")
    bwt_results = []

    for i, dict_data in enumerate(dict_data_list):
        print(f"Running BWT for elicitation {i+1}...")
        bwt_result = bwt(dict_data)
        bwt_results.append(bwt_result)

    print(f"BWT solver result: {bwt_results[-1]['solver_result']['x']}")
    constraint_value = constraints_func(bwt_results[-1]["solver_result"]["x"], dict_data_list[-1])
    print(f"Constraint values for last BWT result: {constraint_value}")

    # Define or load weight spaces
    list_of_weight_space_points = define_weight_spaces(
        dict_data_list, 
        elicitation_numbers, 
        SCRIPT_DIR, 
        required_solutions=required_weight_solutions,
        country=selected_country
    )

    print(f"Imported weight spaces for {len(list_of_weight_space_points)} elicitations.")
    #################################################################################

    #################################################################################
    # Preparation for the Montecarlo Simulation
    print("Starting Monte Carlo simulation...")
    mc_code = mc_simulation(alternatives, opinion_weights, vf_list, conf_list, list_of_weight_space_points, dict_data_list, aggregation_method, sim_runs=n_runs, strict=strict_mode, crit_index=crit_index, random_weight_analysis=random_weight_analysis)
    
    n_elicitations = len(dict_data_list)
    
    # Setup plots (always off interactively)
    fig, ax = create_heatmap(alternative_names, n_alternatives, strict_mode)
    fig_hist, axes = create_histograms(n_alternatives, n_elicitations, strict_mode, plot_bins)
    
    # Create output CSV
    output_file = f"./results/results_{phase_name}_{selected_country}"
    if method_name:
        output_file += f"_{method_name}"
    output_file += ".csv"
    
    os.makedirs("./results", exist_ok=True)
    
    def _make_unique_column_names(names):
        """Make a list of CSV column names unique while preserving order."""
        seen = {}
        unique = []
        for idx, raw in enumerate(names):
            name = str(raw).strip()
            if name == "":
                name = f"Alternative_{idx+1}"
            if name in seen:
                seen[name] += 1
                name = f"{name}_{seen[name]}"
            else:
                seen[name] = 1
            unique.append(name)
        return unique

    if strict_mode:
        if len(alternative_names) != n_alternatives:
            raise ValueError(
                f"Mismatch between loaded alternatives ({n_alternatives}) and names in CSV ({len(alternative_names)})."
            )
        header = ["Run", "Elicitation"] + _make_unique_column_names(alternative_names)
    else:
        if len(alternative_names) != n_alternatives:
            raise ValueError(
                f"Mismatch between loaded alternatives ({n_alternatives}) and names in CSV ({len(alternative_names)})."
            )
        header = ["Run"] + _make_unique_column_names(alternative_names)
    
    with open(output_file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
    
    # Prepare to collect simulation statistics
    full_sets = []
    rank_counts = np.zeros((n_alternatives, n_alternatives))
    if strict_mode:
        lists_of_full_sets = [[] for _ in range(n_elicitations)]
        rank_counts_per_el = [np.zeros((n_alternatives, n_alternatives)) for _ in range(n_elicitations)]
    else:
        lists_of_full_sets = None
        rank_counts_per_el = None
    #################################################################################

    #################################################################################
    # MONTECARLO MAIN LOOP
    for i, r in enumerate(mc_code):
        if strict_mode:
            elic_idx, run_results = r
        else:
            run_results = r

        # Compute ranking
        ranking_run = np.argsort(run_results)[::-1]
        for pos, alt in enumerate(ranking_run):
            rank_counts[alt, pos] += 1
            if strict_mode:
                rank_counts_per_el[elic_idx][alt, pos] += 1

        rank_probs = rank_counts / (i + 1)

        # Store results
        if strict_mode:
            lists_of_full_sets[elic_idx].append(run_results.copy())
            all_runs_combined = [item for sub in lists_of_full_sets for item in sub]
            full_sets = all_runs_combined
            distributions = np.array(full_sets).T if len(full_sets) > 0 else np.zeros((n_alternatives, 0))
        else:
            full_sets.append(run_results.copy())
            distributions = np.array(full_sets).T

        # Save to CSV
        with open(output_file, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            run_results_list = run_results.tolist() if hasattr(run_results, 'tolist') else list(run_results)
            if strict_mode:
                writer.writerow([i+1, elic_idx+1] + run_results_list)
            else:
                writer.writerow([i+1] + run_results_list)

        # Update plots
        if (i % UPDATE_EVERY == 0 or i == n_runs - 1):
            update_plots(
                rank_probs,
                distributions,
                i,
                n_runs,
                n_alternatives,
                alternative_names,
                strict=strict_mode,
                n_elicitations=n_elicitations,
                lists_of_full_sets=lists_of_full_sets,
                rank_counts_per_el=rank_counts_per_el,
                fig=fig,
                ax=ax,
                fig_hist=fig_hist,
                axes=axes,
                plot_bins=plot_bins,
            )

        # Print progress
        if strict_mode:
            print(f"[RUNNING] {i+1}/{n_runs*n_elicitations}", end='\r', flush=True)
        else:
            print(f"[RUNNING] {i+1}/{n_runs}", end='\r', flush=True)

    # Finalize and save plots
    finalize_layout(fig, fig_hist)
    save_figures_as_pdf(fig, fig_hist, phase_name, selected_country, method_name)
    
    # Close figures to free memory
    plt.close(fig)
    plt.close(fig_hist)
    
    print(f"\n✓ Simulation complete. Results saved to: {output_file}")
    #################################################################################

#################################################################################
# ORCHESTRATE ALL PHASES
print("\n" + "="*80)
print("MULTI-PHASE MONTE CARLO ANALYSIS")
print("="*80)

# PHASE 1: IT with random_weight_analysis=TRUE, cycling through aggregation methods
print("\n[PHASE 1] Running IT with random weight analysis...")
for method_name, aggregation_func in aggregation_methods:
    run_monte_carlo_simulation(
        selected_country=base_country,
        aggregation_method=aggregation_func,
        strict_mode=False,
        random_weight_analysis=True,
        phase_name="Phase1_RandomWeights",
        method_name=method_name,
    )

# PHASE 2: All countries with geometric_mean in STRICT mode
print("\n[PHASE 2] Running all countries with geometric_mean in STRICT mode...")
for country in countries_to_analyze:
    run_monte_carlo_simulation(
        selected_country=country,
        aggregation_method=geometric_mean,
        strict_mode=True,
        random_weight_analysis=False,
        phase_name="Phase2_GeometricMean_Strict",
    )

# # PHASE 3: All countries with all methods in non-strict mode
# print("\n[PHASE 3] Running all countries with all aggregation methods in non-strict mode...")
# for country in countries_to_analyze:
#     for method_name, aggregation_func in aggregation_methods:
#         run_monte_carlo_simulation(
#             selected_country=country,
#             aggregation_method=aggregation_func,
#             strict_mode=False,
#             random_weight_analysis=False,
#             phase_name="Phase3_AllMethods",
#             method_name=method_name,
#         )

print("\n" + "="*80)
print("ALL PHASES COMPLETED")
print("="*80)
#################################################################################