from pathlib import Path
import runpy
import sys


BASE_DIR = Path(__file__).resolve().parent
COMPLEX_DIR = BASE_DIR / "common" / "complex"
SIMPLE_DIR = BASE_DIR / "common" / "simple"
SPECIFIC_DIR = BASE_DIR / "specific"
VALUE_FUNCTION_APP_DIR = BASE_DIR.parent / "02 - Value Functions"
COMBINE_RESULTS_SCRIPT = VALUE_FUNCTION_APP_DIR / "combine_results.py"
AGGREGATED_RESULTS_DIR = VALUE_FUNCTION_APP_DIR / "aggregatedresults"


def launch_score_correction_ui(criterion_dir: Path) -> None:
	"""Open the UI that lets the user correct qualitative scores in-place."""
	raise NotImplementedError("Qualitative score correction UI is pending implementation")


def launch_qualitative_value_function_ui(criterion_dir: Path) -> None:
	"""Open the qualitative value function UI preloaded with the updated data and a linear default."""
	raise NotImplementedError("Qualitative value function UI is pending implementation")


def launch_value_function_ui(criteria_path: Path, value_functions_path: Path) -> None:
	"""Open the existing value-function UI preloading the provided criteria and value function CSVs."""
	sys.path.insert(0, str(VALUE_FUNCTION_APP_DIR))
	from PyQt5.QtWidgets import QApplication
	from elicitation_logic import ElicitationProcess
	from main_window import MainWindow

	app = QApplication.instance()
	created_app = app is None
	if created_app:
		app = QApplication([])

	process = ElicitationProcess()
	process.load_data(str(criteria_path))

	window = MainWindow(process)
	window.file_label.setText(f"Loaded: {criteria_path}")
	window.load_elicited_csv(str(value_functions_path))
	window.show()

	app.exec_()

	if created_app:
		app.quit()


def launch_weight_elicitation_ui(criteria_path: Path, output_dir: Path) -> None:
	"""Run the weight elicitation UI on the provided criteria and persist results under output_dir."""
	raise NotImplementedError("Weight elicitation UI hookup is pending implementation")


def run_complex_elicitation() -> None:
	"""Process qualitative criteria first: correct scores, then elicit value functions."""
	for criterion_dir in sorted(p for p in COMPLEX_DIR.iterdir() if p.is_dir()):
		launch_score_correction_ui(criterion_dir)
		launch_qualitative_value_function_ui(criterion_dir)


def run_simple_value_functions() -> None:
	"""Run value function elicitation for the common simple indicators."""
	criteria_path = SIMPLE_DIR / "criteria.csv"
	value_functions_path = SIMPLE_DIR / "value_functions.csv"
	launch_value_function_ui(criteria_path, value_functions_path)


def run_specific_value_functions() -> None:
	"""Sequentially elicit value functions for each specific indicator subfolder."""
	for indicator_dir in sorted(p for p in SPECIFIC_DIR.iterdir() if p.is_dir()):
		criteria_path = indicator_dir / "criteria.csv"
		value_functions_path = indicator_dir / "value_functions.csv"
		launch_value_function_ui(criteria_path, value_functions_path)


def combine_results() -> None:
	"""Combine simple, specific, and complex outputs via the existing aggregation script."""
	common_folders = ["simple"]
	specific_folders = [p.name for p in sorted(SPECIFIC_DIR.iterdir()) if p.is_dir()]
	runpy.run_path(str(COMBINE_RESULTS_SCRIPT), init_globals={
		"common_folders": common_folders,
		"specific_folders": specific_folders,
	})


def run_weight_elicitation() -> None:
	"""Run weight elicitation for each aggregated result (per specifics)."""
	for result_dir in sorted(AGGREGATED_RESULTS_DIR.iterdir()):
		if result_dir.is_dir():
			criteria_path = result_dir / "criteria.csv"
			launch_weight_elicitation_ui(criteria_path, result_dir)


def run() -> None:
	"""Full elicitation pipeline as outlined in the high-level overview."""
	# run_complex_elicitation()
	run_simple_value_functions()
	run_specific_value_functions()
	combine_results()
	run_weight_elicitation()


if __name__ == "__main__":
	run()