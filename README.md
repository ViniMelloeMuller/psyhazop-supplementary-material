# Replication Package: Behavioral Evaluation of Large Language Models Under Cognitive Biases in HAZOP Studies

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Reproducibility: Autonomous & Auditable](https://img.shields.io/badge/Reproducibility-Tier%201%20%26%202-brightgreen.svg)](#reproducibility-standards)

Supplementary Computational Material for:
> **"Behavioral Evaluation of Large Language Models Under Cognitive Biases in HAZOP Studies"**  
> *Vinicius M. Müller, Vitor A. O. Silva, Sávio S. V. Vianna, Flávio V. Silva*  
> School of Chemical Engineering, University of Campinas (UNICAMP), Brazil.

---

## Overview and Scientific Purpose

This replication package serves as the autonomous, auditable computational supplementary material for the paper. In accordance with computational reproducibility principles for high-impact journals, this package is self-contained: **any reviewer or reader can deterministically verify and reproduce all empirical statistics, inter-rater reliability metrics, Bayesian GLMM estimates, publication tables, and figures in seconds with a single command**, without incurring API fees or relying on external services.

### Reproducibility Standards

1. **Tier 1 (Deterministic Result Reproduction):** Generates all reported sample totals, Cohen's kappa reliability statistics with Wald 95% confidence intervals, descriptive tables, 2x2 matched transitions, Bayesian hierarchical logistic regression (GLMM) estimands ($\mathcal{H}_1$ and $\mathcal{H}_2$), LaTeX tables (`tab_*.tex`), and TikZ figures (`fig_*.tex`) directly from the frozen experimental data and MCMC traces.
2. **Tier 2 (Qualitative and Adjudication Auditability):** Includes the complete audit workbooks from independent evaluators (`ratings_rater_1.xlsx`, `ratings_rater_2.xlsx`), the documented third-judge tie-break adjudication (`adjudication.csv`), the de-anonymization keys (`BLINDING_KEY_RESTRICTED.csv`), and all 420 verbatim, raw single-pass LLM responses across the three evaluated models.

---

## Directory Structure

```text
supplementary_material/
├── README.md                          # This document
├── pyproject.toml                     # Standard PEP 621 dependencies
├── reproduce.py                       # Self-contained master reproduction script
├── data/
│   ├── analysis_dataset.csv           # Adjudicated N=420 study dataset
│   ├── audit/                         # Independent double-blind rating workbooks
│   │   ├── 01_Planilha_Desempate_Juiz_3_Preenchida.xlsx # Full Judge 3 adjudication workbook
│   │   ├── anthropic__claude-opus-5.5/ # Evaluator 1 & 2 workbooks + keys
│   │   ├── google__gemini-3.8-flash/   # Evaluator 1 & 2 workbooks + keys
│   │   └── meta__muse-spark-1.3/       # Evaluator 1 & 2 workbooks + keys
│   └── events/                        # 420 verbatim model response records (JSON)
│       ├── anthropic__claude-opus-5.5/ # 140 event JSONs
│       ├── google__gemini-3.8-flash/   # 140 event JSONs
│       └── meta__muse-spark-1.3/       # 140 event JSONs
├── traces/                            # Pre-computed Bayesian MCMC posterior traces (ArviZ NetCDF)
│   ├── posterior.nc                   # Primary NUTS trace (4 chains × 2,000 draws = 8,000 draws)
│   └── sensitivity/                   # Traces for the 3 alternative prior specifications
└── outputs/                           # Regenerated replication artifacts
    ├── mydata.dat                     # LaTeX datatool key/value file
    ├── cohen_kappa_summary.csv        # Pre-adjudication Cohen's kappa and Wald 95% CIs
    ├── descriptive_summary.csv        # Observed bias counts and proportions
    ├── statistical_analysis.json      # Complete Bayesian GLMM summary and diagnostics
    ├── tab_cases_conditions.tex       # Table: Cases, conditions, and response counts
    ├── tab_descriptive_bias.tex       # Table: Observed target-bias counts and rates
    ├── tab_reliability.tex            # Table: Pre-adjudication inter-rater reliability
    ├── tab_paired_transitions.tex     # Table: 2x2 matched control-manipulation transitions
    ├── tab_cross_model_differences.tex# Table: Pairwise model susceptibility differences (H2)
    ├── tab_global.tex                 # Table: GLMM baseline, manipulation effect, and sigma_u
    ├── tab_bias_effects.tex           # Table: Conditional Odds Ratios by Bias Family (H1)
    ├── tab_model_variability.tex      # Table: Relative model susceptibility multipliers
    ├── tab_prior_sensitivity.tex      # Table: Prior sensitivity robustness check
    ├── tab_hazop_contrasts.tex        # Table: Verbatim HAZOP excerpt comparison (E-001)
    ├── fig_h1_probability_matrix.tex  # TikZ Figure: Pr(theta > 0 | D) across 15 cells
    ├── fig_h2_pairwise_probabilities.tex # TikZ Figure: Pairwise model comparison
    ├── fig_forest_plot_panel_a.tex    # TikZ Figure: Forest plot of bias family ORs
    ├── fig_forest_plot_panel_b.tex    # TikZ Figure: Forest plot of model multipliers
    ├── fig_process_complexity_panel_a.tex # TikZ Figure: Random intercept density (sigma_u)
    └── fig_process_complexity_panel_b.tex # TikZ Figure: Case random intercepts (u_i)
```

---

## Quickstart: 1-Minute Replication

### Prerequisites
- Python 3.10 or higher.
- Standard scientific packages: `numpy`, `pandas`, `scipy`, `statsmodels`, `arviz`, `xarray`, `openpyxl`.

### Step 1: Set up the environment
Using standard `pip`:
```bash
# Create and activate a clean virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# Install replication dependencies
pip install -e .
```

Alternatively, using [`uv`](https://github.com/astral-sh/uv):
```bash
uv sync
```

### Step 2: Run the replication script
```bash
python reproduce.py
```

Execution takes approximately **2 to 3 seconds**. The script:
1. Recalculates inter-rater reliability (Cohen's unweighted $\kappa$ and Wald 95% CIs) from the raw independent rating workbooks.
2. Computes empirical bias prevalence, Wilson 95% score intervals, and the 2x2 matched transition matrix.
3. Loads and processes the full 8,000 post-warmup MCMC draws from `traces/posterior.nc` to evaluate Hypothesis 1 ($\mathcal{H}_1$) and Hypothesis 2 ($\mathcal{H}_2$), including the 95% empirical ellipsoidal Mahalanobis test.
4. Generates and overwrites all 9 LaTeX tables and 6 TikZ figures in `outputs/`.
5. Exports the complete `mydata.dat` database for LaTeX `datatool` integration.
6. Audits all reproduced values against the paper's benchmarks and prints a verification report to stdout.

---

## Verification Benchmarks

When `python reproduce.py` executes, it checks all reproduced quantities against the paper's reported values:

| Metric | Reproduced Value | Published Paper Benchmark | Status |
| :--- | :---: | :---: | :---: |
| **Overall Pre-Adjudication Cohen's $\kappa$** | `0.3347` | $0.33$ | Passed |
| **$\kappa$ 95% Wald Confidence Interval** | `[0.1603, 0.5091]` | $[0.16, 0.51]$ | Passed |
| **Raw Observed Agreement** | `93.10%` (391/420) | $93\%$ (391/420) | Passed |
| **Manipulated Trials Bias Rate ($Z=1$)** | `8 / 252 (3.17%)` | $8 / 252 (3.2\%)$ | Passed |
| **Neutral Control Bias Rate ($Z=0$)** | `0 / 168 (0.00%)` | $0 / 168 (0.0\%)$ | Passed |
| **Matched Transitions ($0 \to 1$)** | `8 / 252 pairs` | $8 / 252 pairs$ | Passed |
| **Baseline Probability $p_0$ (median)** | `0.014` | $0.014$ ($[0.005, 0.036]$) | Passed |
| **Case Heterogeneity $\sigma_u$ (median)** | `0.47` | $0.47$ ($[0.23, 0.93]$) | Passed |
| **Hypothesis 1 ($\mathcal{H}_1$) Decision** | `Inconclusive` | Inconclusive (15/15 span 1.0) | Passed |
| **Hypothesis 2 ($\mathcal{H}_2$) Decision** | `Inconclusive` | Inconclusive ($d_0^2 < d_{\mathrm{crit}}^2$) | Passed |

---

## Artifact-to-Manuscript Mapping

| Output File (`outputs/`) | Role in Article | Description |
| :--- | :--- | :--- |
| `mydata.dat` | Inline text (`\var{...}`) | Key-value store of all numerical values reported in the text |
| `tab_cases_conditions.tex` | Methodology | Cases, experimental conditions, and response counts |
| `tab_descriptive_bias.tex` | Section 3 | Observed target-bias counts and rates across conditions and LLMs |
| `tab_reliability.tex` | Section 3 | Pre-adjudication Cohen's $\kappa$, Wald 95% CIs, and observed agreement |
| `tab_paired_transitions.tex` | Section 3 | 2x2 matched control-to-manipulation transition matrix |
| `tab_global.tex` | Section 3.1 | GLMM parameters ($\beta_0, \beta_Z$), interpretable estimands ($p_0$), and $\sigma_u$ |
| `tab_bias_effects.tex` | Section 3.1 | Conditional Manipulation Odds Ratios ($\mathrm{OR}_{bm}$) for $\mathcal{H}_1$ |
| `tab_model_variability.tex`| Section 3.2 | Relative model susceptibility multipliers relative to Claude Opus 5.5 |
| `tab_cross_model_differences.tex` | Table 3 / Sec 3.2 | Pairwise model log-odds differences $\Delta_{mm'}$ and sign probabilities |
| `tab_prior_sensitivity.tex`| Section 3.1 | Robustness across 4 prior specifications under structural zeroes |
| `tab_hazop_contrasts.tex` | Table 1 / Sec 3 | Documented qualitative HAZOP contrast excerpt (Gemini vs Claude on E-001) |
| `fig_h1_probability_matrix.tex` | Figure 1 / Sec 3.1 | Directional sign probabilities $\Pr(\theta_{bm} > 0 \mid \mathcal{D})$ heatmap |
| `fig_h2_pairwise_probabilities.tex` | Figure / Sec 3.2 | Pairwise direction probabilities $\Pr(\Delta_{mm'} > 0 \mid \mathcal{D})$ |
| `fig_forest_plot_panel_a.tex` | Figure / Sec 3.1 | TikZ forest plot of bias family Odds Ratios (50% and 95% CrI) |
| `fig_forest_plot_panel_b.tex` | Figure / Sec 3.2 | TikZ forest plot of relative model susceptibility multipliers |
| `fig_process_complexity_panel_a.tex` | Section 3.1 | Posterior density of scenario random intercept scale $\sigma_u$ |
| `fig_process_complexity_panel_b.tex` | Section 3.1 | Caterpillar plot of individual case random intercepts ($u_i$) |

---

## Optional: Resampling the Bayesian GLMM from Scratch

The package ships with the exact posterior draws saved in `traces/posterior.nc` (4 chains × 2,000 warmup + 2,000 retained draws = 8,000 post-warmup draws) sampled using PyMC's No-U-Turn Sampler (NUTS) with random seed `20261017`.

If you wish to re-sample the Markov chains from scratch:
```bash
pip install -e ".[mcmc]"
python reproduce.py --refit-mcmc
```
*Note: Full MCMC sampling with 4 chains and 16,000 total iterations across hierarchical random intercepts takes approximately 10 to 20 minutes depending on CPU core count.*

---

## Auditing the Qualitative LLM Responses

To inspect the raw, verbatim responses generated by the models before human evaluation:
- Browse `data/events/<model_slug>/`:
  - Each JSON file corresponds to one single-pass trial.
  - Contains the verbatim prompt submitted (`request`), the model's technical response with reasoning traces (`response`), token usage statistics, and timestamp metadata.
- Cross-reference with `data/analysis_dataset.csv` using `event_id`.

---

## Citation and Contact

If you use or build upon this replication package, please cite:

```bibtex
@article{muller2026psyhazop,
  title={Behavioral Evaluation of Large Language Models Under Cognitive Biases in HAZOP Studies},
  author={M{\"u}ller, Vinicius M. and Silva, Vitor A. O. and Vianna, S{\'a}vio S. V. and Silva, Fl{\'a}vio V.},
  journal={IFAC-PapersOnLine},
  year={2026}
}
```

For scientific correspondence regarding the study and replication package:
- Vinicius M. Müller: `v195047@dac.unicamp.br`
- Prof. Dr. Flávio V. Silva: `flaviovs@unicamp.br`
