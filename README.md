# BOPS — Batch Bayesian Optimisation with Pareto-Based Selection

Supplementary code repository for peer-review reproducibility verification:

> **"Pareto-Based Selection Strategies for Batch Bayesian Optimization of Expensive Black-Box Functions"**

This self-contained repository provides a functional demonstration of the complete optimization pipeline proposed in the paper. It executes all **5 batch selection methods** across independent trials on standard benchmark functions to facilitate rapid verification by reviewers without requiring any project-specific data paths or heavy infrastructure. The settings are intentionally small and are meant as a reproducibility/demo run, not as a replacement for the full experimental study in the paper.


---

## Folder Structure

```
BOPS/
├── methods.py          # All 5 method definitions (self-contained)
├── run_demo.py         # Optimisation pipeline + demo runner
└── README.md           # This file
```

After running the demo, the following folders are created automatically:

```
BOPS/
├── demo_training_data/ # LHD initial designs (auto-generated)
├── demo_results/       # .npz result files (one per run)
└── demo_summary.txt    # Plain-text results table
```

---

## Requirements

Python 3.8 or later is required. Install all dependencies with:

```bash
pip install numpy scipy GPy scikit-learn pymoo cma pyDOE2
```

| Package | Purpose |
|---|---|
| `numpy`, `scipy` | Numerical computations |
| `GPy` | Gaussian Process surrogate model |
| `scikit-learn` | DBSCAN and KMeans clustering (CSAW, ClusterHC) |
| `pymoo` | Non-dominated sorting for Pareto front |
| `cma` | CMA-ES optimiser (eShotgun exploit step, D≥2) |
| `pyDOE2` | Latin Hypercube sampling (acquisition optimiser multi-start) |


---

## Quick Start

**Step 1** — Download or copy the two files into a folder:

```
BOPS/
├── methods.py
└── run_demo.py
```

**Step 2** — Open a terminal in that folder and run:

```bash
python run_demo.py
```

That is all. No path configuration is needed. The script locates itself automatically.

---

## What the Demo Does

The script runs all 5 methods on two test functions:

| Problem | D | Description | Known optimum |
|---|---|---|---|
| `SixHumpCamel` | 2 | Six-Hump Camel function | f* ≈ −1.032 |
| `Branin` | 2 | Classic 2-D benchmark | f\* ≈ 0.398 |

**Settings** (small for quick verification):

| Parameter | Value |
|---|---|
| Batch size q | 4 |
| Budget | 20 function evaluations (excluding initial design) |
| Initial design | LHD with 5×D points, fixed per problem/run and shared by all methods |
| Independent runs | 2 per method |
| Pareto scenario | Sobol |



### Initial Observations and Fairness

For each `(problem, run)` pair, the script creates one Latin Hypercube Design (LHD) initial data file with `5×D` observations. All methods then reuse exactly the same initial observations for that `(problem, run)`, so differences in the summary table are due to the batch selection methods rather than different starting data.

The default `5×D` initial design is deliberately small to keep the demo fast. For a stronger but slower verification run, increase `N_INIT_FACTOR` in `run_demo.py` from `5` to `10` and optionally increase `N_RUNS` from `2` to `5` or more.

**Methods evaluated:**

| Method | Type | Description |
|---|---|---|
| `eShotgun` | Pareto-based | ε-greedy + Gaussian shotgun batch |
| `eCSAW_Batch` | Pareto-based | ε-greedy first point via CSAW selection |
| `eClusterHC_Batch` | Pareto-based | ε-greedy first point via ClusterHC selection |
| `ClusterHC_PF_Batch` | Pareto-based | DBSCAN clustering + hypervolume contribution |
| `CSAW_PF_Batch` | Pareto-based | DBSCAN clustering + entropy-weighted SAW score |

---

## Expected Output

The terminal will print a results table like:

```
=================================================================
  RESULTS SUMMARY
=================================================================
  Problem        Method                 Mean best    Run 1    Run 2
  -------------- ---------------------- ----------  -------  -------
  SixHumpCamel   eShotgun               -1.0232  -1.0149  -1.0316
  SixHumpCamel   eCSAW_Batch            -1.0254  -1.0281  -1.0228 
  ...
  Branin         eShotgun                0.4791   0.5409   0.4173
  ...

  Total time : ~ 20s
  Results in : demo_results/
  Summary    : demo_summary.txt
=================================================================
```

The full table is also saved to `demo_summary.txt`.

---

## File Descriptions

### `methods.py`

Self-contained module containing all method definitions. No external package (other than those listed above) is required. All helper functions from the original project files are consolidated here.

Exports the `BATCH_METHODS` dictionary which maps method names to callables:

```python
from methods import BATCH_METHODS
func = BATCH_METHODS['eShotgun']
Xnew = func(model, lb, ub, feval_budget, q, cf, epsilon=0.1, pf=True)
```

### `run_demo.py`

Standalone demo script. Contains:
- `SixHumpCamel` and `Branin` test problem definitions (inline — no external data needed)
- GP fitting with GPy (Matérn-5/2 kernel)
- LHD initial design generation
- Optimisation loop (identical to the main BOPS pipeline)
- Results saving and summary table

---



