# Evaluating Nuclear Reactor Designs for Deployment in Europe

**A Multi-Criteria Decision Support Framework with Uncertainty Propagation**

**Master's Thesis — Simone Pagliuca**
*Politecnico di Milano, Nuclear Engineering*
*Carried out at the Paul Scherrer Institute (PSI), Sep 2025 – Feb 2026*
Advisor: Prof. Stefano Lorenzi · Co-advisors: Prof. Russell McKenna, Dr. Peter Burgherr, Dr. River Huang

[![Overleaf Project - Read Only](https://img.shields.io/badge/Overleaf-Project-blue)](https://www.overleaf.com/read/nvnzzqkztwfd#c543f9)

This repository contains the full pipeline behind the thesis: the indicator/attribute database, the expert-elicitation tools used to build value functions and weights, the **UP-MAVT** (Uncertainty-Propagated Multi-Attribute Value Theory) Monte Carlo evaluation engine, the resulting figures/data, and the LaTeX source of the thesis and its executive summary.

---

## How the pieces fit together

The repository mirrors the thesis workflow, in order:

```
01 - Attributes Definition   →  02 - Value Functions   →  03 - Weighting / UP-MAVT  →  LaTex_Project
   (indicators & raw data)      (expert elicitation:        (expert elicitation:         (thesis text, pulling
                                  value functions per          criteria weights via         figures directly from
                                  criterion)                    PILE-BWT)                    the folders above)
                                                                       ↓
                                                            UP-MAVT Monte Carlo simulation
                                                                       ↓
                                                    UP-MAVT/results, weight_spaces  →  Complete Results (export)
```

No `requirements.txt`/`environment.yml` is provided yet; see [Python dependencies](#python-dependencies) below for what to install.

---

## Repository structure

### `01 - Attributes Definition/`
Defines and collects all indicators (attributes) used in the MCDA framework.

- **`Attributes_Draft.ods`**, **`Indicators_E1.ods`**, **`Indicators_Zdenko.ods`**, **`Capex eval.ods`** — spreadsheet drafts of indicators, data sources, and calculations.
- **`Attribute_DB.csv`** / **`Attribute_DB_read.py`** — the consolidated indicator database and an example `pandas` script to load and summarize it.
- **`criteria.csv`**, **`alternatives.csv`** — the criteria and reactor-design alternatives used downstream.

Data sub-folders (each with its own analysis notebook/script, `framework.py`):
- **`Electricity Data/`** — electricity markets and load profiles, cross-border exchange (`Border Flows/`), carbon intensity, `Electricity Market Analysis.ipynb`.
- **`Reactors Data/`** — technical/performance data sourced from **IAEA-PRIS** and **ARIS**, `Reactor Analysis.ipynb`, `capex_Eval.ipynb`, and plots (capacity factors, CAPEX distribution, reactors by country).
- **`Social and Political Indicators Data/`** — socio-economic and political-stability/public-acceptance indicators sourced from **IMF**, **World Bank**, and **V-Dem**.

### `02 - Value Functions/`
Two PyQt5 desktop applications used to elicit expert **value functions** (mapping raw indicator values to a 0–1 value scale), plus a script that aggregates their outputs.

| Script | Purpose |
|---|---|
| `main.py` | Entry point for the **quantitative** elicitation GUI (`python "main.py"`). Lets an expert pick monotonic/non-monotonic behavior, thresholds and indifference points, a confidence level, and a curve fit (piecewise linear, polynomial, monotone spline/PCHIP, Gaussian, sigmoid), with a live `matplotlib` preview. |
| `main_window.py` | The `MainWindow` implementation behind the quantitative GUI. |
| `elicitation_logic.py` | `ElicitationProcess`: loads criteria/guideline CSVs, stores elicited points, fits curves (`scipy.optimize.curve_fit`, `PchipInterpolator`, isotonic regression), and writes `value_functions.csv`. |
| `qualitative_elicitation.py` | A second, standalone GUI (`python qualitative_elicitation.py`) for **qualitative** criteria: experts rank or place alternatives on a spectrum, then assign values per rank via sliders. |
| `plotter.py` | Stateless `matplotlib` plotting helper shared by the quantitative GUI. |
| `combine_results.py` | Non-GUI batch script (`python combine_results.py`) that merges the `quantitative/` and `qualitative/` outputs, expands per-country overrides, infers criterion polarity, and writes the results to `aggregatedresults/`. Run after both GUIs have produced data. |

Data folders: **`quantitative/`** and **`qualitative/`** (inputs: `criteria.csv`, `alternatives.csv`, `guideline.txt`; output: `value_functions.csv`); **`aggregatedresults/`** (output of `combine_results.py`, one sub-folder per country: `CH`, `FR`, `IT`, `PO`).

### `03 - Weighting/`
A Tkinter application implementing the **Best-Worst Tradeoff (BWT)** interview used to elicit criteria weights — the precursor to the fuller PILE-BWT method used in `UP-MAVT/`.

| Script | Purpose |
|---|---|
| `main.py` | Entry point (`python main.py`). Loads `criteria.csv`, shows a sanity-check plot of the value functions per criterion group, then launches the `WBT_ui` interview. |
| `ui.py` | `WBT_ui`: the Best-Worst comparison interview — the expert marks the best/worst criterion per group, then indicates the data value at which each other criterion becomes equally preferable to the best/worst. |
| `best_worst_tradeoff.py` | `bwt()`: solves for criteria weights via `scipy.optimize.minimize` (SLSQP) from the elicited tradeoffs. |
| `auxiliary.py` | Helpers to import criteria/value functions, plot results, and save `criterion_weights_N.csv`. |

Files: `criteria.csv`, `value_functions.csv`, `BWT_results.csv` (pairwise best/worst comparisons and confidence per expert).

### `UP-MAVT/`
The thesis's core method: **Uncertainty-Propagated MAVT**, a Monte Carlo engine that propagates uncertainty in raw data, elicited value functions, and elicited weights through a Multi-Attribute Value Theory aggregation.

| Script | Purpose |
|---|---|
| `main.py` | Orchestrates the experiment phases (Phase 1: Italy only, random Dirichlet weights, 3 aggregation methods; Phase 2: all 4 countries, geometric-mean aggregation; Phase 3: all methods × all countries — commented out by default). For each run: loads elicitation data, solves PILE-BWT weights per expert, runs `n_runs = 10000` Monte Carlo simulations, and writes result CSVs and heatmap/histogram PDFs. |
| `up_mavt.py` | Core Monte Carlo mechanics: samples a criterion value from its uncertainty distribution (Normal, Uniform, Discrete, Triangular, Trapezoidal, and confidence-weighted qualitative distributions), evaluates it through the elicited value function, and scores each alternative per draw. |
| `pile_bwt.py` | **PILE-BWT** optimizer: solves for weights via SLSQP (with trust-constr/COBYQA fallback) subject to intra- and inter-group best/worst tradeoff constraints; also defines/samples the weight spaces. |
| `aggregation_methods.py` | Aggregation operators: weighted sum, geometric mean, harmonic mean, minimum, maximum. |
| `constraints.py` | Builds the SLSQP constraint set used by `pile_bwt.bwt()`. |
| `generate_weight_spaces.py` | Standalone, parallelized (`multiprocessing`) precomputation of weight-space samples for all four countries. |
| `plot_results.py` / `plotting.py` | Reload saved `results_*.csv` files and (re)generate heatmap/histogram figures without rerunning the simulation. |
| `auxiliary.py` | Data loading/consistency-checking glue (`startup`, `load_criteria_file`, `verify_criteria_consistency`, `combine_alternatives_by_country`, …). |

Country data: **`alternatives_CH.csv`**, **`alternatives_FR.csv`**, **`alternatives_IT.csv`**, **`alternatives_PO.csv`** (Switzerland, France, Italy, Poland — the four case-study countries), each cell encoding an uncertainty distribution for one reactor design × criterion.

Output folders:
- **`elicitation_results/`** — one folder per expert-elicitation session (`1`–`9`), each with per-country `alternatives.csv`/`criteria.csv`/`value_functions.csv` and plots.
- **`results/`** — raw Monte Carlo output (`results_Phase{1,2,3}_*.csv`) and rendered figures in `pdfs/`.
- **`weight_spaces/`** — PILE-BWT weight solutions and diagnostic plots (`ratio_comparison_*`, `weight_space_ranges_*`) per expert and country.

### `Complete Results/`
A curated, flattened export of the final figures used in the thesis (no code): `Outputs from UP-MAVT Code/` (heatmaps/histograms), `Value Functions/{CH,FR,IT,PO}/` (final elicited value-function plots and rankings), `Weight Spaces from PILE-BWT/{1,2,4,9}/`. Note that `LaTex_Project/main.tex` actually pulls figures directly from `UP-MAVT/results/pdfs/`, `UP-MAVT/weight_spaces/results/`, and `UP-MAVT/elicitation_results/plots/` — this folder is a convenience copy, not the primary source.

### `Executive_Summary/`
A standalone LaTeX executive summary sharing the PoliMi template assets. See [Compilation](#compilation).

### `LaTex_Project/`
The thesis itself, built on the official Politecnico di Milano MSc thesis template (`polimi_template_classic/`, "PoliMi3i"). Chapters (`\input` in `main.tex`): `00 - Introduction`, `01 - Literature Review`, `02 - Methods`, `03 - Selection` (with sub-chapters `03a` Attributes, `03b` Technical, `03c` Feasibility, `03d` Economic, `03e` Public, `03f` Removed, `03g` Elicitation Process, `03h` Digital Tools), `04a` Results, `04b` Discussion, `04c` Suggestions, and appendices `A - Code`, `B - Survey`, `C - Suggestions`. Bibliography via `biblatex`/`biber` (`references.bib`); acronyms via `glossaries`/`makeglossaries` (`acronyms.tex`). `ad_hoc_plots/` contains small standalone scripts used to generate illustrative (non-pipeline) methodology figures.

---

## Python dependencies

There is no `requirements.txt` yet; install the following to run the elicitation tools and the UP-MAVT engine (Python 3.12 was used during development):

```bash
pip install PyQt5 numpy pandas scipy matplotlib seaborn
```

`03 - Weighting/` uses `tkinter`, which ships with most Python installations but may need a separate OS package on Linux (e.g. `python3-tk`).

## Running the pipeline

```bash
# 1. Elicit value functions (per criterion)
cd "02 - Value Functions"
python main.py                     # quantitative criteria (GUI)
python qualitative_elicitation.py  # qualitative criteria (GUI)
python combine_results.py          # merge into aggregatedresults/

# 2. (Optional/legacy) simple Best-Worst weighting
cd "../03 - Weighting"
python main.py

# 3. PILE-BWT weighting + Monte Carlo UP-MAVT evaluation
cd "../UP-MAVT"
python generate_weight_spaces.py   # optional: precompute weight spaces
python main.py                     # runs the configured phase(s), writes results/ and weight_spaces/
python plot_results.py             # re-render figures from saved results without rerunning the simulation
```

## Compilation

Both LaTeX documents use the Politecnico di Milano thesis template and require `pdflatex` + `biber` (and `makeglossaries` for the acronym list).

**Thesis** (`LaTex_Project/`):
```bash
# Linux/macOS
bash compile.sh

# Windows
compile.bat
```
> `compile.bat` does not run `makeglossaries`, so acronyms may not resolve when compiling on Windows — use `compile.sh` (e.g. via WSL or Git Bash) if the glossary needs to be up to date.

**Executive summary** (`Executive_Summary/`):
```bash
bash compile.sh
```

Both scripts build in a `build/` subdirectory and copy the final PDF (`main.pdf` / `executive_summary.pdf`) back to the project root.

---

## License

No license file is currently included; the thesis content and code are not licensed for reuse. The PoliMi thesis template under `LaTex_Project/polimi_template_classic/` carries its own Politecnico di Milano NC-BY notice, which covers only the template, not the thesis content or the analysis code.

## Contact

Simone Pagliuca — Politecnico di Milano — pagliuca.simone01@gmail.com
