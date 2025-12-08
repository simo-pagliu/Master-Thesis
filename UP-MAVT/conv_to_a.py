"""Utility to convert WBT comparisons into 1/value_function(value) terms.

Reads a WBT results CSV (e.g. ``wbt_results_1.csv``) and the corresponding
value functions CSV (e.g. ``value_functions_1.csv``), evaluates the relevant
value function for each comparison value, and writes a new CSV with the
computed inverse (``1 / v_f(value)``). The criterion whose value function is
used depends on the comparison type:

* ``best`` comparisons use the reference criterion's value function.
* ``worst`` comparisons use the "other" criterion's value function (the one
  marked as worst).

Fail-fast: missing criteria, invalid numbers, or zero-valued value functions
raise so issues surface immediately.
"""

import csv
import os
from typing import Dict, Iterable, List

from auxiliary import load_value_functions

# Instead of criteria name define some short codes:
CRITERION_CODES = {
	"Capital Investment Budgeted": "E1",
	"Construction Complexity": "E2",
	"Design Complexity": "E3",
	"Design Maturity": "F1",
	"Discharge Burnup": "T1",
	"Electricity Market Benefit": "S1",
	"GHGe Benefit": "S2",
	"Licensing Status": "F2",
	"Load following Capabilities": "T2",
	"Net Power Output": "T3",
	"Nuclear Waste": "S3",
	"Operative + Fuel Costs": "E4",
	"Percived Safety": "S4",
	"Supplier Availbility": "F3",
	"Thermal Efficiency": "T4",
}

# Input/output paths (relative to this file)
INPUT_WBT = "wbt_results_1.csv"
INPUT_VALUE_FUNCTIONS = "value_functions_1.csv"
OUTPUT_A_VALUES = "a_values.csv"


def read_wbt_rows(path: str) -> Iterable[Dict[str, str]]:
	"""Yield comparison rows from a WBT results CSV, skipping empty lines."""
	with open(path, mode="r", newline="") as infile:
		reader = csv.DictReader(infile)
		for row in reader:
			if not row:
				continue
			# Skip rows that are entirely empty/whitespace
			if all((val is None) or (str(val).strip() == "") for val in row.values()):
				continue
			yield row


def make_code(comp_type: str, group: str, reference: str, other: str) -> str:
	"""Build compact code like b_E1_E2 based on type/group and criteria codes."""
	comp_type = comp_type.lower()
	group_lower = group.lower()

	if "between-groups-b" in group_lower:
		scope = "between_b"
	elif "between-groups-w" in group_lower:
		scope = "between_w"
	else:
		scope = "intra"

	if comp_type == "best":
		prefix_map = {"intra": "b", "between_b": "bb", "between_w": "bw"}
	elif comp_type == "worst":
		prefix_map = {"intra": "w", "between_b": "wb", "between_w": "ww"}
	else:
		raise ValueError(f"Unsupported comparison type '{comp_type}'")

	prefix = prefix_map.get(scope)
	if prefix is None:
		raise ValueError(f"Unsupported group '{group}'")

	try:
		ref_code = CRITERION_CODES[reference]
		other_code = CRITERION_CODES[other]
	except KeyError as exc:
		raise KeyError(f"Missing criterion code mapping for '{exc.args[0]}'") from exc

	return f"{prefix}_{ref_code}_{other_code}"


def compute_inverse_values(
    wbt_rows: Iterable[Dict[str, str]],
    value_functions: Dict[str, callable],
) -> List[Dict[str, object]]:
    """Compute 1/value_function(value) for each WBT comparison."""
    computed = []
    for row in wbt_rows:
        comp_type = row.get("Type", "").strip().lower()
        reference = row.get("Reference", "").strip()
        other = row.get("Other", "").strip()
        raw_value = row.get("Value", "").strip()
        group = row.get("Group", "").strip()

        if comp_type not in {"best", "worst"}:
            raise ValueError(f"Unsupported comparison type '{comp_type}' in row: {row}")

        try:
            comparison_value = float(raw_value)
        except Exception as exc:
            raise ValueError(f"Invalid numeric value '{raw_value}' for row: {row}") from exc

        code = make_code(comp_type, group, reference, other)

        # For best comparisons, use the reference criterion's value function.
        # For worst comparisons, use the other criterion's value function.
        target_criterion = reference if comp_type == "best" else other
        vf = value_functions.get(target_criterion)
        if vf is None:
            raise KeyError(f"Value function for criterion '{target_criterion}' not found")

        vf_value = float(vf(comparison_value))
        if vf_value == 0:
            raise ZeroDivisionError(
                f"Value function for criterion '{target_criterion}' returned 0 at x={comparison_value}"
            )

        inverse_value = 1.0 / vf_value

        computed.append(
            {
                "code": code,
                "inverse_value_function": inverse_value,
            }
        )

    return computed


def write_inverse_values(path: str, rows: List[Dict[str, object]]) -> None:
    """Write computed inverse value-function results to CSV."""
    fieldnames = ["code", "inverse_value_function"]
    with open(path, mode="w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    wbt_path = os.path.join(base_dir, INPUT_WBT)
    vf_path = os.path.join(base_dir, INPUT_VALUE_FUNCTIONS)
    out_path = os.path.join(base_dir, OUTPUT_A_VALUES)

    # Load value functions once
    value_functions = load_value_functions(vf_path)

    # Read WBT rows and compute inverse values
    wbt_rows = list(read_wbt_rows(wbt_path))
    computed_rows = compute_inverse_values(wbt_rows, value_functions)

    write_inverse_values(out_path, computed_rows)
    print(f"Computed inverse values for {len(computed_rows)} comparisons -> {out_path}")


if __name__ == "__main__":
	main()