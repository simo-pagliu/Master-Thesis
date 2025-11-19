################################################
# Import 3rd party libraries
################################################
import tkinter as tk
import os
################################################
# Import local modules
################################################
from ui import WBT_ui
from best_worst_tradeoff import bwt
from auxiliary import import_criteria

################################################
# User data input
################################################
DEBUG = True
working_directory = "03 - Weighting"

################################################
# Generate needed aux. vars
################################################
criteria = import_criteria(os.path.join(working_directory, "criteria.csv"))
criteria_count = len(criteria)

################################################
# Use of auxiliary functions
################################################
#value_function_plot(os.path.join(working_directory, "criteria.csv"))

################################################
# Run the UI
################################################
root = tk.Tk()
app = WBT_ui(root, criteria)
root.mainloop()

################################################
# Run BWT model
################################################
#result = bwt(criteria, app.worst_comparison_results, app.best_comparison_results)

################################################
# Run Auxiliary functions
################################################
#plot_results(result, app.reordered_criteria_names)
#save_to_file(result, app.reordered_criteria_names, working_directory)

################################################
# DEBUG OUTPUT
################################################
# if DEBUG:
#     print("\033[93mDebugging Information:\033[0m")
#     print("Criteria:\n", criteria)
#     print("--------------------------------")
#     print("Criteria count:\n", criteria_count)
#     print("--------------------------------")
#     print("Best comparison results: \n", app.best_comparison_results)
#     print("--------------------------------")
#     print("Worst comparison results: \n", app.worst_comparison_results)
#     print("--------------------------------")
#     print("Full Optimization Output: \n", result)
#     print("\033[93mEnd Debugging Information\033[0m")
