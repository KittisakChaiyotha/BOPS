"""
methods.py  —  All Batch Selection Methods
===========================================
Self-contained definitions of all 5 batch Bayesian optimisation methods
used in the BOPS paper:

  Pareto-based (5):
    eShotgun, eCSAW_Batch, eClusterHC_Batch, ClusterHC_PF_Batch, CSAW_PF_Batch


  Also exports:
    RandomPareto  — Pareto front approximation (Sobol/LHD/Random)
    _HyperVolume  — hypervolume computation
    BATCH_METHODS — dict mapping method name → callable

Requirements:
    pip install numpy scipy GPy scikit-learn pymoo cma pyDOE2
"""

# ============================================================================
# REFERENCES / ATTRIBUTION
# ============================================================================
# The eShotgun method implemented in this file is the epsilon-Shotgun
# (e-shotgun) acquisition function of De Ath et al. (2020). The implementation
# here follows their formulation; please cite the original work when using it:
#
#   @inproceedings{de2020e,
#     title     = {$\epsilon$-shotgun: $\epsilon$-greedy Batch Bayesian Optimisation},
#     author    = {De Ath, George and Everson, Richard M. and
#                  Fieldsend, Jonathan E. and Rahat, Alma A. M.},
#     booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference},
#     pages     = {787--795},
#     year      = {2020},
#     publisher = {ACM}
#   }
# ============================================================================


# ============================================================================
# IMPORTS
# ============================================================================
import math
import warnings
import numpy as np
import scipy
import scipy.optimize
import cma
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import MinMaxScaler
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.stats import qmc
from pyDOE2.doe_lhs import lhs


# ============================================================================
# I.  HYPERVOLUME  (pyhv.py)
# ============================================================================

class _HyperVolume:
    """Exact hypervolume (minimisation). Fonseca et al. 2006."""

    def __init__(self, referencePoint):
        self.referencePoint = referencePoint
        self.list = []

    def compute(self, front):
        relevantPoints = np.array(front, dtype=float)
        referencePoint = self.referencePoint
        dimensions     = len(referencePoint)
        if any(referencePoint):
            relevantPoints -= referencePoint
        self.preProcess(relevantPoints)
        bounds = [-1.0e308] * dimensions
        return self.hvRecursive(dimensions - 1, len(relevantPoints), bounds)

    def hvRecursive(self, dimIndex, length, bounds):
        hvol     = 0.0
        sentinel = self.list.sentinel
        if length == 0: return hvol
        if dimIndex == 0:
            return -sentinel.next[0].cargo[0]
        if dimIndex == 1:
            q = sentinel.next[1]; h = q.cargo[0]; p = q.next[1]
            while p is not sentinel:
                pCargo = p.cargo
                hvol += h * (q.cargo[1] - pCargo[1])
                if pCargo[0] < h: h = pCargo[0]
                q = p; p = q.next[1]
            hvol += h * q.cargo[1]
            return hvol
        remove = self.list.remove; reinsert = self.list.reinsert
        hvRecursive = self.hvRecursive
        p = sentinel; q = p.prev[dimIndex]
        while q.cargo is not None:
            if q.ignore < dimIndex: q.ignore = 0
            q = q.prev[dimIndex]
        q = p.prev[dimIndex]
        while length > 1 and (q.cargo[dimIndex] > bounds[dimIndex] or
                               q.prev[dimIndex].cargo[dimIndex] >= bounds[dimIndex]):
            p = q; remove(p, dimIndex, bounds); q = p.prev[dimIndex]; length -= 1
        qArea = q.area; qCargo = q.cargo; qPrevDimIndex = q.prev[dimIndex]
        if length > 1:
            hvol = (qPrevDimIndex.volume[dimIndex] +
                    qPrevDimIndex.area[dimIndex] * (qCargo[dimIndex] - qPrevDimIndex.cargo[dimIndex]))
        else:
            qArea[0] = 1
            qArea[1:dimIndex+1] = [qArea[i] * -qCargo[i] for i in range(dimIndex)]
        q.volume[dimIndex] = hvol
        if q.ignore >= dimIndex:
            qArea[dimIndex] = qPrevDimIndex.area[dimIndex]
        else:
            qArea[dimIndex] = hvRecursive(dimIndex - 1, length, bounds)
            if qArea[dimIndex] <= qPrevDimIndex.area[dimIndex]: q.ignore = dimIndex
        while p is not sentinel:
            pCargoDimIndex = p.cargo[dimIndex]
            hvol += q.area[dimIndex] * (pCargoDimIndex - q.cargo[dimIndex])
            bounds[dimIndex] = pCargoDimIndex; reinsert(p, dimIndex, bounds); length += 1
            q = p; p = p.next[dimIndex]; q.volume[dimIndex] = hvol
            if q.ignore >= dimIndex:
                q.area[dimIndex] = q.prev[dimIndex].area[dimIndex]
            else:
                q.area[dimIndex] = hvRecursive(dimIndex - 1, length, bounds)
                if q.area[dimIndex] <= q.prev[dimIndex].area[dimIndex]: q.ignore = dimIndex
        hvol -= q.area[dimIndex] * q.cargo[dimIndex]
        return hvol

    def preProcess(self, front):
        dimensions = len(self.referencePoint)
        nodeList   = _MultiList(dimensions)
        nodes      = [_MultiList.Node(dimensions, point) for point in front]
        for i in range(dimensions):
            self._sortByDim(nodes, i); nodeList.extend(nodes, i)
        self.list = nodeList

    def _sortByDim(self, nodes, i):
        decorated = [(node.cargo[i], node) for node in nodes]
        decorated.sort(); nodes[:] = [n for (_, n) in decorated]


class _MultiList:
    class Node:
        def __init__(self, numberLists, cargo=None):
            self.cargo = cargo; self.next = [None]*numberLists
            self.prev  = [None]*numberLists; self.ignore = 0
            self.area  = [0.0]*numberLists;  self.volume = [0.0]*numberLists
        def __lt__(self, other): return all(self.cargo < other.cargo)

    def __init__(self, numberLists):
        self.numberLists = numberLists
        self.sentinel    = _MultiList.Node(numberLists)
        self.sentinel.next = [self.sentinel]*numberLists
        self.sentinel.prev = [self.sentinel]*numberLists

    def extend(self, nodes, index):
        sentinel = self.sentinel
        for node in nodes:
            last = sentinel.prev[index]; node.next[index] = sentinel
            node.prev[index] = last; sentinel.prev[index] = node; last.next[index] = node

    def remove(self, node, index, bounds):
        for i in range(index):
            pred = node.prev[i]; succ = node.next[i]
            pred.next[i] = succ; succ.prev[i] = pred
            if bounds[i] > node.cargo[i]: bounds[i] = node.cargo[i]
        return node

    def reinsert(self, node, index, bounds):
        for i in range(index):
            node.prev[i].next[i] = node; node.next[i].prev[i] = node
            if bounds[i] > node.cargo[i]: bounds[i] = node.cargo[i]


# ============================================================================
# II.  ACQUISITION OPTIMISERS  (acquisition_optimisers.py)
# ============================================================================

class _CF_INFO:
    def __init__(self, cf):
        self.cf = cf; self.got_cf = cf is not None
        self.bad_value = np.array(np.inf)
    def not_valid(self, X): return self.got_cf and (not self.cf(X))


def _aquisition_CMAES(model, aq_func, cf=None, aq_kwargs={}):
    dim = model.X.shape[1]; cf_info = _CF_INFO(cf)
    def f(X):
        if isinstance(X, list): X = np.reshape(np.array(X), (len(X), dim))
        elif len(X.shape) != 2: X = np.atleast_2d(X)
        pred_mean, pred_var = model.predict(X, full_cov=False)
        pred_std = np.sqrt(pred_var)
        aq_res = -aq_func(pred_mean, pred_std, **aq_kwargs)
        aq_res = aq_res.ravel().tolist()
        for i in range(len(aq_res)):
            if cf_info.not_valid(X[i, :]): aq_res[i] = cf_info.bad_value.flat[0]
        return aq_res[0] if len(aq_res) == 1 else aq_res
    return f


def _aquisition_LBFGSB(model, aq_func, cf=None, aq_kwargs={}):
    dim = model.X.shape[1]; cf_info = _CF_INFO(cf)
    def f(X):
        if cf_info.not_valid(X): return cf_info.bad_value
        X = X.reshape(-1, dim)
        pred_mean, pred_var = model.predict(X, full_cov=False)
        pred_std = np.sqrt(pred_var)
        return -np.squeeze(aq_func(pred_mean, pred_std, **aq_kwargs))
    return f


def _minimise_CMAES(f, lb, ub, maxeval=5000, cf=None, ftol_abs=1e-15):
    cma_options = {'bounds': [list(lb), list(ub)], 'tolfun': ftol_abs,
                   'maxfevals': maxeval, 'verb_disp': 0, 'verb_log': 0,
                   'verbose': -1, 'CMA_stds': np.abs(ub - lb)}
    if cf is None:
        x0 = lambda: np.random.uniform(lb, ub)
    else:
        def _make_x0(cf, lb, ub):
            def wrapper():
                while True:
                    x = np.random.uniform(lb, ub)
                    if np.all(x >= lb) and np.all(x <= ub) and cf(x): return x
            return wrapper
        x0 = _make_x0(cf, lb, ub)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        xopt, _ = cma.fmin2(f, x0=x0, sigma0=0.25, options=cma_options,
                            bipop=True, restarts=9)
    return xopt


def _minimise_LBFGSB(f, lb, ub, maxeval=5000, cf=None, ftol_abs=1e-15):
    dim = lb.size; N_opt = 10; N_lhs = max(maxeval - N_opt*100, N_opt)
    x0s = lhs(dim, samples=N_lhs, criterion='m') * (ub-lb) + lb
    fx  = np.array(f(x0s)).ravel()
    x0s = x0s[np.argsort(fx)[:N_opt], :]
    bounds = [(l, b) for l, b in zip(lb, ub)]
    factr  = ftol_abs / np.finfo(float).eps
    xb = np.zeros((N_opt, dim)); fv = np.zeros((N_opt, 1))
    for i, x0 in enumerate(x0s):
        xb[i], fv[i], _ = scipy.optimize.fmin_l_bfgs_b(
            f, x0=x0, bounds=bounds, approx_grad=True, factr=factr)
    return xb[np.argmin(fv.flat)]


# ============================================================================
# III.  RANDOM PARETO  (random_pareto.py)
# ============================================================================

def RandomPareto(model, fevals, lb, ub, cf=None, method="sobol"):
    """
    Estimate the Pareto front of a GP model in (mu, sigma) space.
    method: 'sobol' | 'lhd' | 'random'
    """
    D = lb.size; lb = np.array(lb); ub = np.array(ub)
    if method == 'sobol':
        sampler = qmc.Sobol(d=D, scramble=True)
        m = int(np.ceil(np.log2(max(fevals, 2))))
        Xs = sampler.random_base2(m)[:fevals]
    elif method == 'lhd':
        Xs = qmc.LatinHypercube(d=D).random(n=fevals)
    elif method == 'random':
        Xs = np.random.rand(fevals, D)
    else:
        raise ValueError(f"Unknown method: {method}")
    Xs = qmc.scale(Xs, lb, ub)
    if cf is not None:
        mask = np.array([cf(x) for x in Xs]); Xs = Xs[mask]
    mu, var   = model.predict(Xs)
    sigma     = np.sqrt(var)
    F         = np.column_stack([mu.ravel(), sigma.ravel()])
    F_nds     = np.column_stack([mu.ravel(), -sigma.ravel()])
    nds       = NonDominatedSorting().do(F_nds, only_non_dominated_front=True)
    X_front   = Xs[nds]; musigma = F[nds]
    return X_front, musigma


# ============================================================================
# IV.  PARETO-BASED SELECTION HELPERS  (egreedy_shotgun.py)
# ============================================================================

def _estimate_L(model, xj, lengthscale, lb, ub):
    def df(x, model):
        x = np.atleast_2d(x); dmdx, _ = model.predictive_gradients(x)
        res = np.sqrt((dmdx * dmdx).sum(1))
        if x.shape[0] == 1: res = res[0]
        return -res
    n_dim = xj.size
    df_lb = np.maximum(xj - lengthscale, lb)
    df_ub = np.minimum(xj + lengthscale, ub)
    bounds = list(zip(df_lb, df_ub))
    samples = np.vstack([np.random.uniform(df_lb, df_ub, size=(500, n_dim)), model.X])
    x0 = samples[np.argmin(df(samples, model))]
    _, minusL, _ = scipy.optimize.fmin_l_bfgs_b(
        df, x0, bounds=bounds, args=(model,), maxiter=2000, approx_grad=True)
    L = -np.squeeze(minusL).item()
    return L if L >= 1e-7 else 10


def _ball_radius(xj, model, lb, ub):
    ls = model.kern.lengthscale[0]
    L  = _estimate_L(model, xj, ls, lb, ub)
    M  = np.min(model.Y)
    mu_xj, sig2_xj = model.predict(np.atleast_2d(xj))
    rj = (np.abs(mu_xj - M) + np.sqrt(sig2_xj)) / L
    return np.squeeze(rj)


def _hv_contributions(points, ref):
    contributions = np.zeros(len(points))
    if len(points) == 0: return contributions
    hv_base   = _HyperVolume(ref)
    base_vol  = hv_base.compute(points)
    for i in range(len(points)):
        rest = np.array(points[:i].tolist() + points[i+1:].tolist())
        if rest.ndim == 1 and len(rest) > 0: rest = rest.reshape(1, -1)
        contributions[i] = base_vol - (hv_base.compute(rest) if len(rest) > 0 else 0)
    return contributions


def _compute_entropy_weights(G: np.ndarray):
    m, n = G.shape; G = G + 1e-12
    G_sum = G.sum(axis=0)
    pij   = G / np.where(G_sum == 0, 1e-18, G_sum)
    k_val = 1 / np.log(m)
    E_term = np.where(pij > 0, pij * np.log(pij), 0.0)
    E = -k_val * E_term.sum(axis=0)
    d = 1 - E
    return d / d.sum()


def _exploit_batch(X, mu, k_batch=4):
    k = min(k_batch, X.shape[0])
    idx = np.argsort(mu.ravel())[:k]
    if len(idx) < k_batch:
        best = X[idx[0]] if len(idx) > 0 else X[np.argmin(mu.ravel())]
        filler = np.tile(best, (k_batch - len(idx), 1))
        return np.vstack([X[idx], filler])
    return X[idx]


def _get_pf_candidates(model, f_lb, f_ub, feval_budget, cf, pf_method='sobol'):
    X_front, musigma = RandomPareto(model, feval_budget, f_lb, f_ub,
                                    cf=cf, method=pf_method)
    if X_front.shape[0] == 0:
        n_dim   = f_lb.size
        X_front = np.random.uniform(f_lb, f_ub, size=(max(200, feval_budget*2), n_dim))
        mu, sig2 = model.predict(X_front)
        musigma  = np.hstack([mu, np.sqrt(sig2)])
    return X_front, musigma[:, 0].reshape(-1,1), musigma[:, 1].reshape(-1,1)


def _shotgun_exploit_batch(model, f_lb, f_ub, feval_budget, q, cf=None):
    """Exploitative Shotgun branch used by Algorithm 6.

    First select an anchor point xj by minimizing the GP predictive mean mu(x).
    CMA-ES is used in dimensions >= 2, while L-BFGS-B is used in 1-D because
    CMA-ES does not support one-dimensional optimization. The remaining batch
    points are generated by local Gaussian sampling around xj with the adaptive
    Shotgun radius.
    """
    f_lb = np.asarray(f_lb, dtype=float).reshape(-1)
    f_ub = np.asarray(f_ub, dtype=float).reshape(-1)

    if f_lb.size == 1:
        # CMA-ES does not support 1-D — use L-BFGS-B instead.
        f_acq = _aquisition_LBFGSB(model, lambda mu, sig: -mu, cf)
        xj    = _minimise_LBFGSB(f_acq, f_lb, f_ub, feval_budget, cf)
    else:
        f_acq = _aquisition_CMAES(model, lambda mu, sig: -mu, cf)
        xj    = _minimise_CMAES(f_acq, f_lb, f_ub, feval_budget, cf)

    xj = np.asarray(xj, dtype=float).reshape(-1)
    rj = _ball_radius(xj, model, f_lb, f_ub)
    rj = np.minimum(rj, np.linalg.norm(f_ub - f_lb) / 2)

    Xnew = [xj]
    while len(Xnew) < q:
        Xt = np.random.normal(loc=xj, scale=rj)
        bad = np.flatnonzero((Xt < f_lb) | (Xt > f_ub))
        while bad.size > 0:
            Xt[bad] = np.random.normal(loc=xj[bad], scale=rj)
            bad = np.flatnonzero((Xt < f_lb) | (Xt > f_ub))
        Xnew.append(Xt)

    return np.array(Xnew)


# ── ClusterHC (k-batch guaranteed) ──────────────────────────────────────────

def _ClusterHC_selection(X, mu, sigma, eps_dbscan=0.05, k_batch=4):
    """ClusterHC: DBSCAN (+ k-means fallback) + hypervolume-contribution selection.

    Follows the ClusterHC algorithm. DBSCAN the objective values into J clusters;
    if J > q, re-cluster the objective values into exactly q clusters with
    k-means; take the maximum-hypervolume-contribution representative of each
    cluster; if fewer than q representatives are obtained, fill the batch from the
    remaining Pareto solutions ranked by global hypervolume contribution.
    """
    q = min(k_batch, X.shape[0])
    if X.shape[0] == 0:
        return X
    fb_idx = np.argsort(mu.ravel())[:q]
    fb     = X[fb_idx]                              # fallback: lowest predicted mean
    pred_obj = np.vstack([mu, sigma]).T
    if np.any(np.max(pred_obj, 0) - np.min(pred_obj, 0) < 1e-9):
        return fb
    try:
        norm   = MinMaxScaler().fit_transform(pred_obj)
        labels = DBSCAN(eps=eps_dbscan, min_samples=1).fit(norm).labels_
        cluster_ids = sorted(set(labels) - {-1})
        J = len(cluster_ids)

        hv_pts = np.vstack([mu, -sigma]).T
        ref    = np.max(hv_pts, axis=0) + 1e-6

        # k-means fallback when DBSCAN finds more clusters than the batch size
        if J > q:
            labels = KMeans(n_clusters=q, n_init=10,
                            random_state=0).fit(norm).labels_
            cluster_ids = sorted(set(labels))

        # one representative (max hypervolume contribution) per cluster
        reps, rep_idx = [], set()
        for cid in cluster_ids:
            ci  = np.where(labels == cid)[0]
            hvs = _hv_contributions(hv_pts[ci], ref)
            bi  = int(ci[np.argmax(hvs)])
            reps.append(X[bi]); rep_idx.add(bi)

        # fill remaining slots from unchosen Pareto solutions by global HC
        if len(reps) < q:
            rem = [i for i in range(X.shape[0]) if i not in rep_idx]
            if rem:
                rh = _hv_contributions(hv_pts[rem], ref)
                for i in np.argsort(rh)[::-1][:q - len(reps)]:
                    reps.append(X[rem[i]]); rep_idx.add(rem[i])

        # safety pad for degenerate fronts
        while len(reps) < q:
            reps.append(fb[len(reps) % len(fb)])

        return np.array(reps[:q])
    except Exception as e:
        print(f"ClusterHC failed: {e}"); return fb


# ── CSAW (entropy SAW + DBSCAN) ──────────────────────────────────────────────

def _CSAW_selection(X, mu, sigma, eps_dbscan=0.05, k_batch=4):
    """CSAW: entropy-weighted SAW + DBSCAN clustering."""
    k    = min(k_batch, X.shape[0])
    fb_idx = np.argsort(mu.ravel())[:k]; fb = X[fb_idx]
    if X.shape[0] == 0: return fb
    pred_obj = np.vstack([mu, sigma]).T
    if np.any(np.max(pred_obj,0) - np.min(pred_obj,0) < 1e-9): return fb
    try:
        # Step 1-3: entropy weights + SAW score
        G_norm = MinMaxScaler().fit_transform(pred_obj)
        w      = _compute_entropy_weights(G_norm); w1, w2 = w[0], w[1]
        mu_f = mu.ravel(); sig_f = sigma.ravel()
        mu_r  = mu_f.max()-mu_f.min() or 1e-12
        sig_r = sig_f.max()-sig_f.min() or 1e-12
        Ai = ((mu_f.max()-mu_f)/mu_r)*w1 + ((sig_f-sig_f.min())/sig_r)*w2
        # Step 4: DBSCAN
        norm   = MinMaxScaler().fit_transform(pred_obj)
        labels = DBSCAN(eps=eps_dbscan, min_samples=1).fit(norm).labels_
        nc     = len(set(labels) - {-1})
        if nc == 0: return fb
        if nc > k:
            labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(norm)
        # Step 5: best SAW per cluster
        selected = []; sel_idx = set()
        for cid in set(labels) - {-1}:
            ci = np.where(labels==cid)[0]
            bi = ci[np.argmax(Ai[ci])]
            selected.append(X[bi]); sel_idx.add(bi)
        if len(selected) < k:
            rem = [i for i in np.argsort(Ai)[::-1] if i not in sel_idx]
            for i in rem:
                if len(selected) >= k: break
                selected.append(X[i]); sel_idx.add(i)
        while len(selected) < k: selected.append(fb[len(selected) % k])
        return np.array(selected[:k])
    except Exception as e:
        print(f"CSAW failed: {e}"); return fb


# ============================================================================
# V.  PARETO-BASED BATCH METHODS
# ============================================================================

def eShotgun(model, f_lb, f_ub, feval_budget, q, cf,
             epsilon=0.1, pf=True, pf_method='sobol', **kwargs):
    """eShotgun: epsilon-greedy anchor + Gaussian shotgun batch.

    The epsilon-Shotgun acquisition of De Ath et al. (2020). With probability
    epsilon, the anchor is sampled from the approximated Pareto set; otherwise,
    the anchor is obtained by minimizing the GP predictive mean. The rest of the
    batch is sampled locally around the anchor using the adaptive Shotgun radius.
    """
    if np.random.uniform() < epsilon:
        if pf:
            X_front, _ = RandomPareto(model, feval_budget, f_lb, f_ub,
                                       cf=cf, method=pf_method)
            xj = (X_front[np.random.choice(X_front.shape[0])]
                  if X_front.shape[0] > 0 else np.random.uniform(f_lb, f_ub))
        else:
            xj = np.random.uniform(f_lb, f_ub)

        xj = np.asarray(xj, dtype=float).reshape(-1)
        rj = _ball_radius(xj, model, f_lb, f_ub)
        rj = np.minimum(rj, np.linalg.norm(f_ub - f_lb) / 2)
        Xnew = [xj]
        while len(Xnew) < q:
            Xt = np.random.normal(loc=xj, scale=rj)
            bad = np.flatnonzero((Xt < f_lb) | (Xt > f_ub))
            while bad.size > 0:
                Xt[bad] = np.random.normal(loc=xj[bad], scale=rj)
                bad = np.flatnonzero((Xt < f_lb) | (Xt > f_ub))
            Xnew.append(Xt)
        return np.array(Xnew)

    return _shotgun_exploit_batch(model, f_lb, f_ub, feval_budget, q, cf)


def eClusterHC_Batch(model, f_lb, f_ub, feval_budget, q, cf,
                     epsilon=0.1, pf=True, pf_method='sobol', **kwargs):
    """eClusterHC following Algorithm 6.

    With probability epsilon, select the whole batch using ClusterHC on the
    approximated Pareto set. Otherwise, use the exploitative Shotgun branch:
    minimize the GP predictive mean for the anchor and sample the remaining
    points locally around that anchor.
    """
    if np.random.rand() < epsilon:
        X_cand, mu, sigma = _get_pf_candidates(model, f_lb, f_ub, feval_budget,
                                                cf, pf_method)
        Xnew = _ClusterHC_selection(X_cand, mu, sigma, k_batch=q)
        Xnew = np.atleast_2d(Xnew)
        if Xnew.shape[0] != q:
            fp = Xnew[0] if Xnew.shape[0] > 0 else np.random.uniform(f_lb, f_ub)
            Xnew = np.vstack([Xnew, np.tile(fp, (q-Xnew.shape[0], 1))])[:q]
        return Xnew

    return _shotgun_exploit_batch(model, f_lb, f_ub, feval_budget, q, cf)


def eCSAW_Batch(model, f_lb, f_ub, feval_budget, q, cf,
                epsilon=0.1, pf=True, pf_method='sobol', **kwargs):
    """eCSAW following Algorithm 6.

    With probability epsilon, select the whole batch using CSAW on the
    approximated Pareto set. Otherwise, use the exploitative Shotgun branch:
    minimize the GP predictive mean for the anchor and sample the remaining
    points locally around that anchor.
    """
    if np.random.rand() < epsilon:
        X_cand, mu, sigma = _get_pf_candidates(model, f_lb, f_ub, feval_budget,
                                                cf, pf_method)
        Xnew = _CSAW_selection(X_cand, mu, sigma, k_batch=q)
        Xnew = np.atleast_2d(Xnew)
        if Xnew.shape[0] != q:
            fp = Xnew[0] if Xnew.shape[0] > 0 else np.random.uniform(f_lb, f_ub)
            Xnew = np.vstack([Xnew, np.tile(fp, (q-Xnew.shape[0], 1))])[:q]
        return Xnew

    return _shotgun_exploit_batch(model, f_lb, f_ub, feval_budget, q, cf)


def ClusterHC_PF_Batch(model, f_lb, f_ub, feval_budget, q, cf,
                        pf_method='sobol', **kwargs):
    """ClusterHC: pure DBSCAN + hypervolume selection on Pareto front."""
    X_cand, mu, sigma = _get_pf_candidates(model, f_lb, f_ub, feval_budget,
                                            cf, pf_method)
    Xnew = _ClusterHC_selection(X_cand, mu, sigma, k_batch=q)
    Xnew = np.atleast_2d(Xnew)
    if Xnew.shape[0] != q:
        fp = Xnew[0] if Xnew.shape[0] > 0 else np.random.uniform(f_lb, f_ub)
        Xnew = np.vstack([Xnew, np.tile(fp, (q-Xnew.shape[0], 1))])[:q]
    return Xnew


def CSAW_PF_Batch(model, f_lb, f_ub, feval_budget, q, cf,
                  pf_method='sobol', **kwargs):
    """CSAW: entropy SAW + DBSCAN selection on Pareto front."""
    X_cand, mu, sigma = _get_pf_candidates(model, f_lb, f_ub, feval_budget,
                                            cf, pf_method)
    Xnew = _CSAW_selection(X_cand, mu, sigma, k_batch=q)
    Xnew = np.atleast_2d(Xnew)
    if Xnew.shape[0] != q:
        fp = Xnew[0] if Xnew.shape[0] > 0 else np.random.uniform(f_lb, f_ub)
        Xnew = np.vstack([Xnew, np.tile(fp, (q-Xnew.shape[0], 1))])[:q]
    return Xnew




# ============================================================================
# VII.  REGISTRY
# ============================================================================

BATCH_METHODS = {
    'eShotgun'          : eShotgun,
    'eCSAW_Batch'       : eCSAW_Batch,
    'eClusterHC_Batch'  : eClusterHC_Batch,
    'ClusterHC_PF_Batch': ClusterHC_PF_Batch,
    'CSAW_PF_Batch'     : CSAW_PF_Batch,
}
