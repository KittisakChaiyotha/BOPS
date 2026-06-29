"""
run_demo.py  —  BOPS Demo Runner
==================================
One-click demo for the BOPS experiment pipeline.

HOW TO RUN
----------
Place both files in one folder:
    your_folder/
    ├── methods.py      (all method definitions)
    └── run_demo.py     (this file)

Then run:
    python run_demo.py

No path configuration needed. All folders are created automatically.

TEST PROBLEMS
-------------
  SixHumpCamel D=2   fast ~3s/method    classic 2-D function f* ≈ -1.0316
  Branin       D=2   fast ~3s/method    classic benchmark    f* ≈ 0.398

DEMO SETTINGS
-------------
  q      = 4   (batch size)
  budget = 20  (function evaluations, excluding initial design)
  runs   = 2   (independent runs per method)
  init   = 5D  (Latin Hypercube initial observations; shared by all methods)

OUTPUT
------
  demo_results/    .npz result files
  demo_summary.txt plain-text results table

REQUIREMENTS
------------
  pip install numpy scipy GPy scikit-learn pymoo cma pyDOE2
"""

import os
import sys
import time
import numpy as np

# ── locate this file and import methods ──────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    from methods import BATCH_METHODS
except ImportError as e:
    print(f"\n[ERROR] Cannot import methods.py: {e}")
    print("Make sure methods.py is in the same folder as run_demo.py")
    sys.exit(1)

try:
    import GPy as gp
except ImportError:
    print("\n[ERROR] GPy not found.  Run:  pip install GPy")
    sys.exit(1)

# ── test problems (inline — no external package needed) ──────────────────────

class SixHumpCamel:
    """2-D Six-Hump Camel function.  Optimum f* ≈ -1.0316  (fast, D=2)"""
    def __init__(self):
        self.dim  = 2
        self.lb   = np.array([-2., -1.])
        self.ub   = np.array([ 2.,  1.])
        self.cf   = None;  self.yopt = -1.0316
    def __call__(self, x):
        x = np.asarray(x).ravel()
        x1, x2 = x[0], x[1]
        return float((4 - 2.1*x1**2 + x1**4/3)*x1**2
                     + x1*x2 + (-4 + 4*x2**2)*x2**2)


class Branin:
    """2-D Branin function.  Three optima at f*≈0.398"""
    def __init__(self):
        self.dim  = 2
        self.lb   = np.array([-5., 0.])
        self.ub   = np.array([10., 15.])
        self.cf   = None; self.yopt = 0.397887
    def __call__(self, x):
        x = np.asarray(x).ravel()
        x1, x2 = x[0], x[1]
        a,b,c = 1, 5.1/(4*np.pi**2), 5/np.pi
        r,s,t = 6, 10, 1/(8*np.pi)
        return float(a*(x2 - b*x1**2 + c*x1 - r)**2
                     + s*(1-t)*np.cos(x1) + s)


PROBLEMS = {
    'SixHumpCamel': SixHumpCamel,
    'Branin'      : Branin,
}

# ── GP helper ─────────────────────────────────────────────────────────────────

def build_GP(Xtr, Ytr):
    kernel = gp.kern.Matern52(input_dim=Xtr.shape[1], ARD=False)
    model  = gp.models.GPRegression(Xtr, Ytr, kernel, normalizer=True)
    model.constrain_positive('')
    (kv, kl, kn) = model.parameter_names()
    model[kv].constrain_bounded(1e-6, 1e6, warning=False)
    model[kl].constrain_bounded(1e-6, 1e6, warning=False)
    model[kn].constrain_fixed(1e-6, warning=False)
    model.optimize_restarts(optimizer='lbfgs', num_restarts=2,
                            num_processes=1, max_iters=20, verbose=False)
    return model

# ── uniform wrapper ───────────────────────────────────────────────────────────

class _UniformWrapper:
    """Maps a problem to [0,1]^D and exposes the original bounds."""
    def __init__(self, f):
        self._f    = f
        self.real_lb = f.lb.copy(); self.real_ub = f.ub.copy()
        self.dim   = f.dim
        self.lb    = np.zeros(f.dim); self.ub = np.ones(f.dim)
        self.cf    = f.cf
    def __call__(self, x):
        x_real = self.real_lb + np.asarray(x) * (self.real_ub - self.real_lb)
        return float(self._f(x_real))


# ── training data generator ───────────────────────────────────────────────────

def make_training_data(problem_name, run_no, train_dir, n_init=None):
    """
    Create or reuse a fixed LHD initial design.

    The same file is reused by all methods for a given (problem, run), so the
    comparison starts from identical initial observations. If N_INIT_FACTOR is
    changed later, stale files with a different size are regenerated.
    """
    os.makedirs(train_dir, exist_ok=True)
    path = os.path.join(train_dir, f"{problem_name}_{run_no}.npz")

    f_raw  = PROBLEMS[problem_name]()
    D      = f_raw.dim
    n_init = n_init or (5 * D)

    if os.path.exists(path):
        try:
            with np.load(path, allow_pickle=True) as old:
                X_old = old['arr_0']; Y_old = old['arr_1']
            if X_old.shape == (n_init, D) and Y_old.reshape(-1, 1).shape == (n_init, 1):
                return path
            print(f"  [init] regenerating {problem_name} run {run_no}: "
                  f"stored n={X_old.shape[0]}, required n={n_init}")
        except Exception:
            print(f"  [init] regenerating unreadable initial file: {path}")

    np.random.seed(run_no * 42)
    lhd = np.zeros((n_init, D))
    for j in range(D):
        perm      = np.random.permutation(n_init)
        lhd[:, j] = (perm + np.random.uniform(size=n_init)) / n_init
    X_real = f_raw.lb + lhd * (f_raw.ub - f_raw.lb)
    Y      = np.array([f_raw(X_real[i]) for i in range(n_init)]).reshape(-1,1)
    np.savez(path, arr_0=X_real, arr_1=Y, n_init=n_init, seed=run_no * 42)
    print(f"  [init] {problem_name} run {run_no}: "
          f"n={n_init}, D={D}, y_min={float(Y.min()):.4f}")
    return path


# ── main optimisation loop ────────────────────────────────────────────────────

def run_one(problem_name, run_no, method_name, method_args,
            batch_size, budget, train_dir, results_dir, scenario='sobol'):
    """
    Run one (problem, run, method) experiment.
    Returns best objective value found, or np.nan on failure.
    """
    np.random.seed(run_no)
    os.makedirs(results_dir, exist_ok=True)

    # Load training data
    data_path = os.path.join(train_dir, f"{problem_name}_{run_no}.npz")
    with np.load(data_path, allow_pickle=True) as d:
        X_real = d['arr_0']; Y = d['arr_1'].reshape(-1, 1)

    f_raw = PROBLEMS[problem_name]()
    f     = _UniformWrapper(f_raw)
    Xtr   = (X_real - f.real_lb) / (f.real_ub - f.real_lb)   # normalise to [0,1]
    Ytr   = Y.copy()
    n_tr  = Ytr.size

    batch_fn  = BATCH_METHODS[method_name]
    feval_bud = 100 * f.dim   # candidate budget for Pareto-front approximation (100D)

    call_args = dict(method_args)
    call_args['pf_method'] = scenario

    print(f"    Starting {method_name} | {problem_name} | run {run_no}")

    batch_no = 0
    while Xtr.shape[0] < budget + n_tr:
        model = build_GP(Xtr, Ytr)
        Xnew  = batch_fn(model, f.lb, f.ub, feval_bud,
                         batch_size, f.cf, **call_args)
        Ynew  = np.zeros((batch_size, 1))
        for i in range(batch_size):
            while True:
                try:
                    Ynew[i] = f(Xnew[i]); break
                except Exception:
                    Xnew[i] = np.random.uniform(f.lb, f.ub)
        Xtr = np.vstack([Xtr, Xnew])
        Ytr = np.vstack([Ytr, Ynew])
        batch_no += 1
        print(f"      Batch {batch_no:3d}: fmin={float(np.min(Ytr)):.5f}")

    # Save
    suf   = "".join(f"_{v}" for v in method_args.values())
    fname = f"{problem_name}_{run_no}_{batch_size}_{budget}_{method_name}{suf}.npz"
    np.savez(os.path.join(results_dir, fname),
             Xtr=Xtr, Ytr=Ytr, budget=budget,
             batch_method=method_name, batch_size=batch_size)

    return float(np.min(Ytr))


# ── demo configuration ────────────────────────────────────────────────────────

DEMO_PROBLEMS  = ['SixHumpCamel', 'Branin']
N_RUNS         = 2
BATCH_SIZE     = 4
BUDGET         = 20
N_INIT_FACTOR  = 5     # initial observations = N_INIT_FACTOR × D
SCENARIO       = 'sobol'

METHODS_TO_RUN = [
    ('eShotgun',           {'epsilon': 0.1, 'pf': True}),
    ('eCSAW_Batch',        {'epsilon': 0.1, 'pf': True}),
    ('eClusterHC_Batch',   {'epsilon': 0.1, 'pf': True}),
    ('ClusterHC_PF_Batch', {}),  
    ('CSAW_PF_Batch',      {}),  
]

TRAIN_DIR   = os.path.join(HERE, 'demo_training_data')
RESULTS_DIR = os.path.join(HERE, 'demo_results')


# ── main ──────────────────────────────────────────────────────────────────────

import warnings as _warnings
_warnings.filterwarnings('ignore', category=RuntimeWarning)

if __name__ == '__main__':

    print("=" * 65)
    print("  BOPS Demo  —  Pareto-Based Selection Criteria")
    print("=" * 65)
    print(f"  Problems : {DEMO_PROBLEMS}")
    print(f"  Methods  : {[m[0] for m in METHODS_TO_RUN]}")
    print(f"  Runs     : {N_RUNS}  |  Budget: {BUDGET}  |  q: {BATCH_SIZE}")
    print(f"  Initial  : LHD with {N_INIT_FACTOR}D points, shared by all methods")
    print(f"  Output   : demo_results/  +  demo_summary.txt")
    print("=" * 65)

    # Step 1 — training data
    print(f"\n[Step 1] Generating initial training data (LHD, {N_INIT_FACTOR}D points)...")
    for prob in DEMO_PROBLEMS:
        for r in range(1, N_RUNS + 1):
            D = PROBLEMS[prob]().dim
            make_training_data(prob, r, TRAIN_DIR, n_init=N_INIT_FACTOR * D)

    # Step 2 — experiments
    print("\n[Step 2] Running experiments...\n")
    records  = []
    t_total  = time.time()

    for prob in DEMO_PROBLEMS:
        for mname, margs in METHODS_TO_RUN:
            run_bests = []
            t0 = time.time()
            for r in range(1, N_RUNS + 1):
                try:
                    y = run_one(prob, r, mname, margs,
                                BATCH_SIZE, BUDGET,
                                TRAIN_DIR, RESULTS_DIR, SCENARIO)
                    run_bests.append(y)
                except Exception as e:
                    print(f"  [FAIL] {prob} | {mname} run {r}: {e}")
                    run_bests.append(np.nan)

            elapsed   = time.time() - t0
            mean_best = float(np.nanmean(run_bests))
            ok = "✓" if not any(np.isnan(run_bests)) else "!"
            records.append((prob, mname, run_bests, mean_best, elapsed))
            print(f"  {ok}  {prob:<14} {mname:<22} "
                  f"mean best = {mean_best:9.4f}   [{elapsed:.1f}s]")

    total = time.time() - t_total

    # Step 3 — summary table
    hdr = (f"  {'Problem':<14} {'Method':<22} {'Mean best':>10}  "
           + "  ".join([f"Run {r+1}" for r in range(N_RUNS)]))
    print(f"\n{'='*65}\n  RESULTS SUMMARY\n{'='*65}")
    print(hdr); print(f"  {'-'*63}")
    for prob, mname, run_bests, mean_best, _ in records:
        rstr = "  ".join([f"{v:7.4f}" if not np.isnan(v) else "  FAIL "
                          for v in run_bests])
        print(f"  {prob:<14} {mname:<22} {mean_best:10.4f}  {rstr}")
    print(f"\n  Total time : {total:.1f}s")
    print(f"  Results in : demo_results/")

    # Step 4 — save txt
    spath = os.path.join(HERE, 'demo_summary.txt')
    with open(spath, 'w', encoding='utf-8') as fp:
        fp.write("BOPS Demo Results\n" + "="*65 + "\n")
        fp.write(f"Problems : {DEMO_PROBLEMS}\n")
        fp.write(f"Methods  : {[m[0] for m in METHODS_TO_RUN]}\n")
        fp.write(f"Runs: {N_RUNS}  Budget: {BUDGET}  q: {BATCH_SIZE}  Initial: {N_INIT_FACTOR}D LHD\n\n")
        fp.write(hdr + "\n" + f"  {'-'*63}\n")
        for prob, mname, run_bests, mean_best, elapsed in records:
            rstr = "  ".join([f"{v:7.4f}" if not np.isnan(v) else "  FAIL "
                              for v in run_bests])
            fp.write(f"  {prob:<14} {mname:<22} {mean_best:10.4f}  {rstr}\n")
        fp.write(f"\nTotal time: {total:.1f}s\n")
    print(f"  Summary    : {spath}")
    print("=" * 65)
