<!-- .github/copilot-instructions.md for the Master-Thesis repo -->
# Quick instructions for AI coding agents

Purpose: give immediate, actionable context so an AI assistant can be productive working in this repository.

- **Big picture**: This repository contains a Master's thesis analysis implemented as a set of Python scripts, data CSVs and a LaTeX project. The numbered top-level directories correspond to pipeline stages: `01 - Attributes Definition`, `02 - Value Functions`, `03 - Weighting`, `04 - Elicitation`. A separate analysis folder `UP-MAVT` contains an alternative aggregation and Monte Carlo workflow.

- **Entry points & UIs**:
  - `02 - Value Functions/main.py` — PyQt5 GUI that initializes `ElicitationProcess` and `MainWindow`. Note: it contains a Wayland compatibility restart helper that sets `QT_QPA_PLATFORM=xcb` before Qt loads.
  - `03 - Weighting/main.py` — Tkinter UI; it sets `working_directory = "03 - Weighting"` and imports `criteria.csv` from that folder. Many scripts assume they are executed with the repository root as CWD or with working-directory variables set accordingly.
  - `UP-MAVT/main.py` — batch Monte Carlo + plotting flow that forces `matplotlib.use('TkAgg')` and expects `criteria.csv`, `alternatives.csv`, and `wbt_results_*.csv`/`value_functions_*.csv` files in the same folder.
  - `LaTex_Project/compile.bat` and `compile.sh` — LaTeX compilation helpers (README lists `pdflatex` and `biber` sequence).

- **Data & configuration patterns**:
  - CSV-driven: `criteria.csv`, `alternatives.csv`, `value_functions.csv`, `wbt_results*.csv` are the main inputs. Scripts commonly read CSVs using `csv.DictReader` or helper functions.
  - Folder names use numeric prefixes to indicate pipeline stage; code sometimes uses relative paths (e.g., the weighting main sets `working_directory` explicitly).
  - `02 - Value Functions/qualitative/main.py` exists but is empty — treat as a placeholder if you plan to implement or refactor qualitative/value-function flows.

- **Common conventions the AI should follow when editing or adding code**:
  - Preserve existing scripting style: small single-file scripts with procedural `if __name__ == '__main__'` entry points.
  - Keep GUI code separate from core logic. Look for classes like `ElicitationProcess`, `MainWindow`, or helper modules (`framework.py`) and modify core logic there rather than editing the top-level UI glue unless the change is UI-specific.
  - Avoid changing runtime backends (Qt/Tk/matplotlib) unless the change is deliberate; scripts sometimes set backends/platforms at import-time for compatibility.
  - **Fail-fast preference:** Do not add extensive `try/except` wrappers, numerous `if file exists` guards, or other defensive scaffolding solely to "fool-proof" execution. The repository owner prefers code that fails visibly at runtime so they can inspect and fix issues directly.

- **How to run things (examples for Windows PowerShell)**:
  - Use the repository virtualenv if present (the project has been run with `.venv` in the past). Example (PowerShell):
    ```powershell
    .\.venv\Scripts\Activate.ps1; python "02 - Value Functions\main.py"
    ```
  - Run weighting GUI:
    ```powershell
    .\.venv\Scripts\Activate.ps1; python "03 - Weighting\main.py"
    ```
  - Run UP-MAVT Monte Carlo (non-GUI batch):
    ```powershell
    .\.venv\Scripts\Activate.ps1; python "UP-MAVT\main.py"
    ```
  - LaTeX compile (from `LaTex_Project`):
    ```powershell
    cd "LaTex_Project"; pdflatex main; biber main; pdflatex main; pdflatex main
    ```

- **Files and locations an agent will commonly edit or inspect**:
  - `02 - Value Functions/elicitation_logic.py`, `main_window.py`, `framework.py` — business logic and UI for value elicitation.
  - `03 - Weighting/*` — Best-Worst Tradeoff tooling (`best_worst_tradeoff.py`, `ui.py`, `auxiliary.py`).
  - `UP-MAVT/*` — Monte Carlo, sampling and aggregation (`up_mavt.py`, `pile_bwt.py`, `aggregation_methods.py`).
  - `01 - Attributes Definition/*` — raw datasets and attribute definitions used as inputs to higher-level scripts.

- **Integration points & external deps**:
  - GUI frameworks: `PyQt5` (Value Functions) and `tkinter` (Weighting). Avoid unifying these unless explicitly requested.
  - Plotting: `matplotlib`, `seaborn` (UP-MAVT). Some files force backends (TkAgg) or platform env vars for Qt.
  - LaTeX: uses `pdflatex` + `biber` for bibliography. Overleaf sync is referenced in README.

- **Developer workflow notes discovered in the repo**:
  - Scripts expect certain CSVs to exist and may create `results/` directories (UP-MAVT writes `./results/results.csv`). Ensure tests/tools do not overwrite user results without prompting.
  - Many scripts use simple print-based debug output and have `DEBUG` flags. Keep these if you add instrumentation; don't remove them silently.

- **When merging changes**:
  - Prefer editing modular files (`elicitation_logic.py`, `auxiliary.py`, `up_mavt.py`) rather than top-level `main.py` unless the change affects process orchestration.
  - Update or add CSV examples under the same directory if the change needs sample data.

If anything in this guidance is unclear or you'd like me to expand a specific section (for example, describe `ElicitationProcess` internals or the Weighting UI flow), tell me which area and I'll update the file.
