import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys

# Add parent directory to path for imports
PARENT_DIR = os.path.dirname(SCRIPT_DIR := os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)

from auxiliary import load_criteria, load_value_functions_with_confidence
from pile_bwt import weight_sampler

# Get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UP_MAVT_DIR = os.path.dirname(SCRIPT_DIR)

# Configuration
ELICITATIONS = [1, 2, 4, 9]
COUNTRIES = ["IT", "FR", "CH", "PO"]

# Main loop over all countries and elicitations
for elicit_num in ELICITATIONS:
    for country in COUNTRIES:
        print(f"\n{'='*60}")
        print(f"Processing Elicitation {elicit_num} - Country {country}")
        print(f"{'='*60}")
        
        try:
            # Load BWT results CSV and criteria data
            bwt_results_file = os.path.join(SCRIPT_DIR, f"BWT_results_{elicit_num}.csv")
            bwt_results_country_file = os.path.join(SCRIPT_DIR, f"BWT_results_{elicit_num}_{country}.csv")
            weight_file = glob.glob(os.path.join(SCRIPT_DIR, f"BWT_results_{elicit_num}_weights_{country}.csv"))
            
            if not weight_file:
                print(f"  ⚠ Weight file not found for elicitation {elicit_num}, country {country}")
                continue
            weight_file = weight_file[0]
            
            criteria_file = os.path.join(UP_MAVT_DIR, "criteria.csv")
            dict_data = load_criteria(criteria_file, bwt_results_file)

            # Load value functions and attach them to dict_data
            elicit_dir = os.path.join(UP_MAVT_DIR, "elicitation_results", str(elicit_num))
            vf_file = os.path.join(elicit_dir, country, "value_functions.csv")
            vf_map = {}

            try:
                vf_map, conf_map = load_value_functions_with_confidence(vf_file)
                
                # Attach value functions to criteria in dict_data
                for group_name, group_data in dict_data.items():
                    for crit_name, crit in group_data['criteria'].items():
                        if crit_name in vf_map:
                            crit['value_function'] = vf_map[crit_name]
                
                print(f"  Loaded value functions from {vf_file}")
            except Exception as e:
                print(f"  Warning: Could not load value functions: {e}")
                continue

            # Map criterion -> value function for quick lookup
            criterion_vf_map = {
                crit_name: crit.get('value_function')
                for group_data in dict_data.values()
                for crit_name, crit in group_data['criteria'].items()
                if crit.get('value_function') is not None
            }

            # Read all weights from the file
            # Format: each row is [criterion_name, value1, value2, value3, ...]
            weights_dict = {}
            with open(weight_file, 'r', newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        criterion = row[0]
                        # All columns after the first are weight values
                        weights = [float(v) for v in row[1:]]
                        weights_dict[criterion] = weights

            # Compute min, max, and range for each criterion
            criteria_names = []
            min_weights = []
            max_weights = []
            ranges = []

            for criterion in sorted(weights_dict.keys()):
                weights = weights_dict[criterion]
                min_w = min(weights)
                max_w = max(weights)
                criteria_names.append(criterion)
                min_weights.append(min_w)
                max_weights.append(max_w)
                ranges.append(max_w - min_w)

            # Create horizontal bar plot
            fig, ax = plt.subplots(figsize=(10, max(6, len(criteria_names) * 0.4)))

            y_pos = np.arange(len(criteria_names))

            # Plot horizontal bars from min to max
            for i, (criterion, min_w, max_w, rng) in enumerate(zip(criteria_names, min_weights, max_weights, ranges)):
                # Draw the bar from min to max
                ax.barh(i, rng, left=min_w, height=0.6, 
                        color='#2b78c8', alpha=0.7, edgecolor='black', linewidth=1)
                # Add text showing the range to the right of the bar
                ax.text(max_w + 0.01, i, f'[{min_w:.3f}, {max_w:.3f}]', 
                        ha='left', va='center', fontsize=8, fontweight='bold')

            ax.set_yticks(y_pos)
            ax.set_yticklabels(criteria_names, fontsize=9)
            ax.set_xlabel('Weight Value', fontsize=11, fontweight='bold')
            ax.set_title(f'Weight Space Ranges - Elicitation {elicit_num} ({country})', 
                            fontsize=13, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.grid(axis='x', alpha=0.3)

            plt.tight_layout()

            # Save the plot
            results_dir = os.path.join(SCRIPT_DIR, 'results', str(elicit_num))
            os.makedirs(results_dir, exist_ok=True)
            plt.savefig(os.path.join(results_dir, f'weight_space_ranges_{country}.png'), dpi=300, bbox_inches='tight')
            # plt.show()  # Display before closing
            plt.close()

            # Create second figure: comparison of declared ratios (a values) vs ratios computed from sampled valid weight sets.
            comparisons = []
            declared_ratios = []
            declared_ratios_country = []
            ratio_means = []
            ratio_stds = []
            labels = []

            # Build weight space structure in the same criterion order used by constraints_func (and by weight_sampler)
            criterion_order = [
                crit_name
                for group_data in dict_data.values()
                for crit_name in group_data['criteria'].keys()
            ]
            crit_index_map = {name: i for i, name in enumerate(criterion_order)}

            weight_space_points = [weights_dict[name] for name in criterion_order if name in weights_dict]
            if len(weight_space_points) != len(criterion_order):
                missing = [name for name in criterion_order if name not in weights_dict]
                print(f"  ⚠ Missing criteria in weight space file: {missing}")
                continue

            # Sample valid weight sets once, then reuse them for all ratio computations
            N_WEIGHT_SAMPLES = 200
            sampled_weight_sets = []
            for _ in range(N_WEIGHT_SAMPLES):
                w = np.array(weight_sampler(dict_data, weight_space_points), dtype=float)
                w = w / np.sum(w)
                sampled_weight_sets.append(w)
            sampled_weight_sets = np.vstack(sampled_weight_sets)  # shape: (N, n_criteria)

            # Load country-specific values into a lookup dictionary
            country_a_values = {}
            if os.path.exists(bwt_results_country_file):
                try:
                    with open(bwt_results_country_file, 'r', newline='') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            ref = row.get('Reference', '')
                            other = row.get('Other', '')
                            key = (ref, other)
                            try:
                                country_a_values[key] = float(row.get('a', 0))
                            except Exception:
                                pass
                except Exception as e:
                    print(f"  Warning: Could not load country values: {e}")

            try:
                with open(bwt_results_file, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        comp_type = row.get('Type', '')
                        ref = row.get('Reference', '')
                        other = row.get('Other', '')
                        group = row.get('Group', '')

                        # Read declared ratio directly from CSV
                        try:
                            a_val = float(row.get('a', 0))
                        except Exception:
                            continue

                        if ref not in crit_index_map or other not in crit_index_map:
                            continue

                        ref_idx = crit_index_map[ref]
                        other_idx = crit_index_map[other]

                        # Compute ratio samples from valid sampled weight sets
                        # For best comparisons: ratio is w_ref / w_other
                        # For worst comparisons: ratio is w_other / w_ref (inverted)
                        if comp_type == 'best':
                            denom = sampled_weight_sets[:, other_idx]
                            numer = sampled_weight_sets[:, ref_idx]
                        else:  # worst
                            denom = sampled_weight_sets[:, ref_idx]
                            numer = sampled_weight_sets[:, other_idx]

                        # denom should never be zero (weights are bounded away from 0), but keep safe division.
                        ratio_samples = np.divide(numer, denom, out=np.zeros_like(numer), where=(denom != 0))
                        ratio_mean = float(np.mean(ratio_samples))
                        ratio_std = float(np.std(ratio_samples, ddof=1)) if len(ratio_samples) > 1 else 0.0

                        # Determine label prefix based on group and comparison type
                        if group == "Between-groups-B":
                            prefix = "BB" if comp_type == "best" else "BW"
                        elif group == "Between-groups-W":
                            prefix = "WB" if comp_type == "best" else "WW"
                        else:
                            prefix = "B" if comp_type == "best" else "W"

                        # Get country value for this comparison
                        a_val_country = country_a_values.get((ref, other), None)

                        comparisons.append((comp_type, ref, other))
                        declared_ratios.append(a_val)
                        declared_ratios_country.append(a_val_country if a_val_country is not None else 0)
                        ratio_means.append(ratio_mean)
                        ratio_stds.append(ratio_std)
                        labels.append(f"{prefix}: {ref[:15]} vs {other[:15]}")
            except Exception as e:
                print(f"  Could not create ratio comparison plot: {e}")
                comparisons = []

            if comparisons:
                # Create grouped bar chart
                n_comparisons = len(comparisons)
            if comparisons:
                # Create grouped bar chart
                n_comparisons = len(comparisons)
                fig2, ax2 = plt.subplots(figsize=(max(12, n_comparisons * 0.6), 8))

                x_pos = np.arange(n_comparisons)
                width = 0.25

                # Plot declared ratios, country ratios, and computed ratios side by side
                ax2.bar(
                    x_pos - width,
                    declared_ratios,
                    width,
                    label='Declared Ratio (a)',
                    color='#2b78c8',
                    alpha=0.7,
                    edgecolor='black',
                )

                ax2.bar(
                    x_pos,
                    declared_ratios_country,
                    width,
                    label=f'Country Ratio (a_{country})',
                    color='#78c82b',
                    alpha=0.7,
                    edgecolor='black',
                )

                ax2.bar(
                    x_pos + width,
                    ratio_means,
                    width,
                    label='Sampled Weight Ratio (mean)',
                    color='#c8782b',
                    alpha=0.7,
                    edgecolor='black',
                    yerr=np.vstack([
                        np.minimum(np.array(ratio_stds, dtype=float), np.array(ratio_means, dtype=float)),
                        np.array(ratio_stds, dtype=float),
                    ]),
                    capsize=4,
                )

                ax2.set_xlabel('Comparisons', fontsize=11, fontweight='bold')
                ax2.set_ylabel('Ratio Value', fontsize=11, fontweight='bold')
                ax2.set_title(
                    f'Declared vs Country vs Sampled Weight Ratios (mean ± σ) - Elicitation {elicit_num} ({country})',
                    fontsize=13,
                    fontweight='bold',
                )
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
                ax2.legend()
                ax2.grid(axis='y', alpha=0.3)

                plt.tight_layout()
                
                # Save the plot
                results_dir = os.path.join(SCRIPT_DIR, 'results', str(elicit_num))
                os.makedirs(results_dir, exist_ok=True)
                plt.savefig(os.path.join(results_dir, f'ratio_comparison_{country}.png'), dpi=300, bbox_inches='tight')
                # plt.show()  # Display before closing
                plt.close()
            
            print(f"  ✓ Successfully processed elicitation {elicit_num}, country {country}")
        
        except Exception as e:
            print(f"  ✗ Error processing elicitation {elicit_num}, country {country}: {e}")
            import traceback
            traceback.print_exc()

print(f"\n{'='*60}")
print("Completed processing all elicitations and countries")
print(f"{'='*60}")

