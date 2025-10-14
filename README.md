# Multi-Criteria Decision Analysis (MCDA) for Nuclear Reactor Design Evaluation

**Master’s Thesis Project**
*Simone Pagliuca, Politecnico di Milano*
*Internship at Paul Scherrer Institute (PSI), Sep 2025 – Feb 2026*

[![Overleaf Project - Read Only](https://img.shields.io/badge/Overleaf-Project-blue)](https://www.overleaf.com/read/nvnzzqkztwfd#c543f9)

This repository contains the data, analysis, and LaTeX source for my Master’s Thesis.

---

## Repository Structure

### LaTeX Project
- **Main LaTeX files**: Compiled using `pdflatex` and `biber` (see [Compilation](#compilation)).
- **Connected to Overleaf**: For real-time collaboration and editing.

### 01 - Attributes Definition
Defines all indicators used in the MCDA framework.
- **`Attributes_Draft.ods`**: Spreadsheet overview of indicators, data sources, and calculations.

#### Electricity Data
- Contains datasets and analyses related to:
  - Electricity markets and load profiles
  - Cross-border electricity exchange
  - Greenhouse gas (GHG) emissions

#### Reactors Data
- Sourced from **PRIS** and **ARIS** databases:
  - Technical specifications
  - Performance metrics
  - Comparative analyses

#### Social and Political Indicators Data
- Sourced from **IMF**, **World Bank**, and **V-Dem**:
  - Socio-economic factors
  - Political stability and public acceptance metrics

---

## Compilation
To compile the LaTeX project locally:
```bash
pdflatex main
biber main
pdflatex main
pdflatex main
