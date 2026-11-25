"""Direct samplers for E7 (docs/E7_direct_sampling_protocol_2026-11-23.md): the measured output distribution of an
optimiser, quantum or classical, used as the partition ensemble with no Metropolis correction.

Every function works on a FIELD-FREE model (E(z) = E(-z); ising.py basis convention, bit k of s <-> z_k = 1 - 2 bit)
and returns a distribution over the 2^(n-1) bipartitions (representatives r < 2^(n-1), i.e. top bit 0), so that
co-assignment probabilities follow from mcmc.coassignment_observables(n) exactly as for the Gibbs target.

Samplers
  optimiser       the exact ground-state bipartition every time (an ideal annealer or exact solver)
  greedy          classical multi-start steepest single-flip descent from uniformly random starts (the classical
                  analogue of repeated optimiser runs); exact distribution by enumerating every start
  anneal(T, dt)   coherent (closed-system) transverse-field annealing H(s) = -(1 - s) sum_k X_k + s alpha H_C,
                  s = t / T, from |+>^n, measured at s = 1; second-order (Strang) splitting with step <= dt
  qaoa(g, b)      |psi> = prod_k exp(-i b_k sum X) exp(-i g_k alpha H_C) |+>^n, measured
alpha = proposals.layden_alpha(J, 0) puts H_C on the scale used by every quantum proposal in this project.
QAOA angles come from optimise_qaoa() with one of three objectives:
  energy   <alpha H_C> (standard QAOA, the optimiser use)
  kl       KL(P_theta || pi_beta) = beta (free energy of P_theta) + log Z: variational Gibbs sampling at the target
           temperature (needs the output entropy, computable in simulation and at these sizes only)
  oracle   worst-pair co-assignment error against the exact answer (not deployable; the floor of the circuit family)
VERIFY (tests/test_e7.py): dense-matrix references for QAOA and annealing, flip symmetry and normalisation, limits
(gamma = 0 and T -> 0 give uniform; slow anneals reach the ground state), greedy end states are local minima.
"""
import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import logsumexp

from . import ising
from .roi_exact import node_uncertainty

EPS_TARGET = 0.02          # E4 precision target for co-assignment probabilities
SHOTS_PER_RUN = 1000       # for the precise-but-wrong index (run-to-run SD with 1,000 shots per run)
NEAR_WINDOW = np.log(10.0)  # near-degenerate set: partitions within a factor 10 of the most probable one


# ----------------------------------------------------------------------------------------------- state vectors
def _plus(n):
    N = 1 << n
    return np.full(N, 1.0 / np.sqrt(N), dtype=complex)


@njit(cache=True)
def _mix_nb(x, c, s, n):
    """x = float64 view (re, im interleaved) of psi. exp(-i b X_k): a' = c a - i s b', b' = c b - i s a."""
    N = x.shape[0] // 2
    for k in range(n):
        step = 1 << k
        for base in range(0, N, 2 * step):
            for j in range(base, base + step):
                l = j + step
                ar = x[2 * j]
                ai = x[2 * j + 1]
                br = x[2 * l]
                bi = x[2 * l + 1]
                x[2 * j] = c * ar + s * bi
                x[2 * j + 1] = c * ai - s * br
                x[2 * l] = c * br + s * ai
                x[2 * l + 1] = c * bi - s * ar


@njit(cache=True)
def _phase_nb(x, aE, g):
    for i in range(aE.shape[0]):
        cs = np.cos(g * aE[i])
        sn = -np.sin(g * aE[i])
        re = x[2 * i]
        im = x[2 * i + 1]
        x[2 * i] = re * cs - im * sn
        x[2 * i + 1] = re * sn + im * cs


def _mix(psi, b, n):
    """psi <- exp(-i b sum_k X_k) psi, in place, by per-qubit 2 x 2 rotations."""
    _mix_nb(psi.view(np.float64), np.cos(b), np.sin(b), n)
    return psi


def _phase(psi, aE, g):
    """psi <- exp(-i g diag(aE)) psi, in place."""
    _phase_nb(psi.view(np.float64), aE, g)
    return psi


def lump(P, n):
    """Distribution over labelled states (2^n) -> distribution over bipartitions (2^(n-1) representatives)."""
    M = 1 << (n - 1)
    return P[:M] + P[M:][::-1]


def flip_asymmetry(P, n):
    M = 1 << (n - 1)
    return float(np.max(np.abs(P[:M] - P[M:][::-1])))


def qaoa_state(aE, gammas, bs):
    n = int(round(np.log2(len(aE))))
    aE = np.ascontiguousarray(aE, dtype=float)
    psi = _plus(n)
    for g, b in zip(gammas, bs):
        _phase(psi, aE, float(g))
        _mix(psi, float(b), n)
    return psi


def qaoa_probs(aE, gammas, bs):
    n = int(round(np.log2(len(aE))))
    return lump(np.abs(qaoa_state(aE, gammas, bs)) ** 2, n)


def anneal_state(aE, T, dt):
    n = int(round(np.log2(len(aE))))
    K = max(1, int(np.ceil(T / dt - 1e-9)))
    h = T / K
    s = (np.arange(K) + 0.5) / K
    aE = np.ascontiguousarray(aE, dtype=float)
    psi = _plus(n)
    _phase(psi, aE, 0.5 * h * s[0])
    for m in range(K):
        _mix(psi, -h * (1.0 - s[m]), n)                     # exp(+i h (1 - s) sum X)
        nxt = s[m + 1] if m + 1 < K else 0.0
        _phase(psi, aE, 0.5 * h * (s[m] + nxt))             # two half steps of exp(-i h s alpha H_C) merged
    return psi


def anneal_probs(aE, T, dt):
    n = int(round(np.log2(len(aE))))
    return lump(np.abs(anneal_state(aE, T, dt)) ** 2, n)


# ----------------------------------------------------------------------------------------------- classical direct
def optimiser_probs(Er, rtol=1e-12):
    """Ideal optimiser: the ground-state bipartition (exact ties, if any, shared equally)."""
    m = Er.min()
    ties = Er <= m + rtol * max(abs(m), 1e-300)
    return ties / ties.sum()


def greedy_probs(J, tol=1e-12):
    """Steepest single-flip descent from every start (uniform over bipartitions; the dynamics commute with the
    global flip, so representatives suffice). Ties between equally steep flips go to the lowest index."""
    n = len(J)
    M = 1 << (n - 1)
    Z = ising.spins(n)[:M].astype(float)
    thr = tol * max(np.abs(J).max(), 1e-300)
    active = np.ones(M, bool)
    while active.any():
        idx = np.where(active)[0]
        Za = Z[idx]
        dE = 2.0 * Za * (Za @ J)                               # energy change of flipping each spin
        k = np.argmin(dE, axis=1)
        better = dE[np.arange(len(idx)), k] < -thr
        Z[idx[better], k[better]] *= -1.0
        active[idx[~better]] = False
    s = (((1 - Z) / 2).astype(np.int64)) @ (1 << np.arange(n, dtype=np.int64))
    s = np.where(s >= M, (1 << n) - 1 - s, s)
    return np.bincount(s, minlength=M) / M


# ----------------------------------------------------------------------------------------------- target and metrics
def gibbs(Er, beta):
    lw = -beta * (Er - Er.min())
    logpi = lw - logsumexp(lw)
    return logpi, np.exp(logpi)


def coassign(P, Fobs):
    return (1.0 + P @ Fobs) / 2.0


def kl(P, logQ):
    """KL(P || Q) in nats, Q given by its log; 0 log 0 = 0."""
    m = P > 0
    return float(np.sum(P[m] * (np.log(P[m]) - logQ[m])))


def calls_to_eps(d, v, eps=EPS_TARGET):
    """Fewest i.i.d. shots S >= 1 with bias^2 + v / S <= eps^2 for EVERY pair; inf if some |bias| >= eps.
    E7 amendment 1 (23 Nov 2026, after the run): S >= 1. A deterministic sampler (v = 0) that is within eps on every
    pair needs one shot; the hashed version returned 0, which crashed the secondary calls table."""
    if np.max(np.abs(d)) >= eps:
        return float("inf")
    return float(max(1.0, np.ceil(np.max(v / (eps ** 2 - d ** 2)))))


def qemcmc_calls(gap, s2max, eps=EPS_TARGET):
    """E4-B accounting: burn ceil(10 / gap) steps, then ceil(max s2 / (4 eps^2)) recorded steps; one call each."""
    return float(np.ceil(10.0 / gap) + np.ceil(s2max / (4 * eps ** 2)))


def metrics(P, logpi, p, Fobs, n, eps=EPS_TARGET, shots=SHOTS_PER_RUN, window=NEAR_WINDOW):
    pi = np.exp(logpi)
    pD = coassign(P, Fobs)
    d = pD - p
    v = pD * (1.0 - pD)
    k = int(np.argmax(np.abs(d)))
    near = logpi >= logpi.max() - window
    w = P[near] / pi[near]
    with np.errstate(divide="ignore"):
        logq = np.log(np.maximum(P, 0.0))
    u_true = node_uncertainty(p, n)
    u_dir = node_uncertainty(np.clip(pD, 0.0, 1.0), n)
    return {
        "floor": float(np.abs(d[k])), "floor_pair": k, "floor_signed": float(d[k]),
        "rms": float(np.sqrt(np.mean(d ** 2))),
        "calls_to_eps": calls_to_eps(d, v, eps),
        "precise_but_wrong": float(abs(d[k]) / np.sqrt(v[k] / shots)) if v[k] > 0 else float("inf"),
        "kl_P_pi_bits": kl(P, logpi) / np.log(2),
        "kl_pi_P_bits": (kl(pi, logq) / np.log(2)) if np.all(P[pi > 0] > 0) else float("inf"),
        "tv": float(0.5 * np.abs(P - pi).sum()),
        "near_size": int(near.sum()), "near_mass_pi": float(pi[near].sum()), "near_mass_P": float(P[near].sum()),
        "suppression_ratio": float(w.max() / w.min()) if w.min() > 0 else float("inf"),
        "p_ground": float(P[int(np.argmax(logpi))]),
        "u_mean_true": float(u_true.mean()), "u_mean": float(u_dir.mean()),
        "du_mean": float((u_dir - u_true).mean()), "du_abs_mean": float(np.abs(u_dir - u_true).mean()),
        "pD": pD.tolist(), "u": u_dir.tolist(),
    }


# ----------------------------------------------------------------------------------------------- QAOA angle search
def make_objective(kind, aE, n, logpi=None, p_target=None, Fobs=None, q=16):
    """Objective of theta = (g_1..g_p, b_1..b_p). kind: energy | kl | oracle_smooth | oracle_max."""
    M = 1 << (n - 1)
    aEr = aE[:M]

    def probs(theta):
        p = len(theta) // 2
        return qaoa_probs(aE, theta[:p], theta[p:])

    if kind == "energy":
        return lambda th: float(probs(th) @ aEr)
    if kind == "kl":
        return lambda th: kl(probs(th), logpi)

    def err(th):
        return coassign(probs(th), Fobs) - p_target

    if kind == "oracle_max":
        return lambda th: float(np.max(np.abs(err(th))))
    if kind == "oracle_smooth":
        def f(th):
            d = np.abs(err(th))
            m = d.max()
            return float(m * np.sum((d / m) ** q) ** (1.0 / q)) if m > 0 else 0.0
        return f
    raise ValueError(kind)


def _interp(theta):
    """INTERP initialisation (Zhou et al., PRX 10:021067, 2020): depth p-1 angles -> depth p."""
    p1 = len(theta) // 2
    out = []
    for x in (theta[:p1], theta[p1:]):
        x = np.concatenate([[0.0], x, [0.0]])
        p = p1 + 1
        out.append(np.array([((i - 1) / (p - 1)) * x[i - 1] + ((p - i) / (p - 1)) * x[i] for i in range(1, p + 1)]))
    return np.concatenate(out)


def _pad(theta):
    p1 = len(theta) // 2
    return np.concatenate([theta[:p1], [0.0], theta[p1:], [0.0]])


def _random_start(rng, p):
    return np.concatenate([rng.uniform(-np.pi / 2, np.pi / 2, p), rng.uniform(0.0, np.pi / 2, p)])


def _bfgs(f, x0, gtol=1e-8, maxiter=400):
    r = minimize(f, x0, method="BFGS", options={"gtol": gtol, "maxiter": maxiter})
    return float(r.fun), np.asarray(r.x, float), int(r.nfev)


def optimise_qaoa(f, p_max, rng, rng2, n_random=24, n_second=24, grid=(24, 12), polish=None, extra=None):
    """Depths 1..p_max. Starts: depth 1 = the 4 best points of a (gamma, b) grid covering one period; depth p >= 2 =
    the depth p-1 optimum padded with an idle layer (so the optimum cannot get worse with depth) and its INTERP
    extension; extra[p] (angles found by other objectives at depth p, cross-seeding); plus n_random random starts.
    A second, independent set of n_second random starts (rng2) measures whether more starts would have helped; the
    result is the best over both sets. polish (oracle only): exact-max objective, minimised by Nelder-Mead from the 3
    best smooth-surrogate solutions, from the padded depth p-1 result and from extra[p], each of which is also a
    candidate as is; so the oracle value is never worse than any of those points. The reported value is exact."""
    out, prev, prev_pol = {}, None, None
    for p in range(1, p_max + 1):
        starts = []
        if p == 1:
            G = np.linspace(0.0, np.pi, grid[0] + 1)[1:]
            B = np.linspace(0.0, np.pi / 2, grid[1], endpoint=False)
            vals = sorted((f(np.array([g, b])), g, b) for g in G for b in B)
            starts += [np.array([g, b]) for _, g, b in vals[:4]]
        else:
            starts += [_pad(prev), _interp(prev)]
        starts += [np.asarray(x, float) for x in (extra or {}).get(p, [])]
        starts += [_random_start(rng, p) for _ in range(n_random)]
        first = sorted((_bfgs(f, x) for x in starts), key=lambda t: t[0])
        second = sorted((_bfgs(f, _random_start(rng2, p)) for _ in range(n_second)), key=lambda t: t[0])
        pool = sorted(first + second, key=lambda t: t[0])
        rec = {"smooth_best": pool[0][0], "smooth_first": first[0][0], "smooth_second": second[0][0] if second else None,
               "n_within_1e-6": int(sum(t[0] <= pool[0][0] + 1e-6 for t in pool)), "n_starts": len(pool),
               "nfev": int(sum(t[2] for t in pool)), "theta_first": first[0][1].tolist()}
        theta = pool[0][1]
        prev = theta
        if polish is not None:
            cands = [(polish(theta), theta)]
            ext = [np.asarray(x, float) for x in (extra or {}).get(p, [])]
            cands += [(polish(x), x) for x in ext]
            seeds = [x for _, x, _ in pool[:3]] + ([_pad(prev_pol)] if prev_pol is not None else []) + ext
            for x in seeds:
                r = minimize(polish, x, method="Nelder-Mead",
                             options={"xatol": 1e-7, "fatol": 1e-9, "maxfev": 600 * len(x), "adaptive": True})
                cands.append((float(r.fun), np.asarray(r.x, float)))
            if prev_pol is not None:
                cands.append((polish(_pad(prev_pol)), _pad(prev_pol)))   # never worse than depth p - 1
            val, theta = min(cands, key=lambda t: t[0])
            rec["value"] = val
            prev_pol = theta
        else:
            rec["value"] = pool[0][0]
        rec["theta"] = theta.tolist()
        out[p] = rec
    return out
