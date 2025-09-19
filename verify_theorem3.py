# n_vs_gap_and_error_ddqn_ges.py
# End-to-end: build fixed candidate set (GES + tiny DDQN), estimate Λ_n and Δ_n,
# and plot (i) n vs gap, (ii) mis-selection probability vs n with exponential-shape curve.

import os, time, random, argparse
from collections import deque
from itertools import product

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# Put this once near your imports (after importing matplotlib.pyplot as plt):
plt.rcParams.update({
    "axes.labelsize": 14,     # default axis label size
    "xtick.labelsize": 12,    # default tick size
    "ytick.labelsize": 12,    # default tick size
    "legend.fontsize": 12,    # default legend text size
    "figure.titlesize": 15    # default title size
})


# ------------------------- optional GES -------------------------
try:
    from castle.algorithms import GES
    HAS_GES = True
except Exception:
    HAS_GES = False

# ------------------------- seeds / device -------------------------
def set_seed(seed: int):
    np.random.seed(seed); random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device_dtype(use64: bool, device: str):
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    # Keep the NN in float32 (stable & fast). Scoring can stay float64 internally.
    torch.set_default_dtype(torch.float32)
    return dev, torch.float32

# ------------------------- graph utils -------------------------
def binarize(A):
    A = (A != 0).astype(np.int8)
    np.fill_diagonal(A, 0)
    return A

def is_dag(A):
    return nx.is_directed_acyclic_graph(nx.DiGraph(A))

# ------------------------- synthetic SEM -------------------------
def random_dag(p, edge_prob=3.0/30.0, w_low=0.5, w_high=1.0, seed=0):
    rng = np.random.default_rng(seed)
    A = np.zeros((p, p))
    for i in range(p):
        for j in range(i+1, p):
            if rng.random() < edge_prob:
                A[i, j] = rng.uniform(w_low, w_high) * (1 if rng.random() < 0.5 else -1)
    return A  # weighted, upper-triangular

def sample_sem(Aw, n, noise_std=1.0, seed=0):
    """Linear-Gaussian SEM: X = (I - Aw)^(-1) e, e ~ N(0, noise_std^2 I)."""
    rng = np.random.default_rng(seed)
    p = Aw.shape[0]
    M_inv_T = np.linalg.inv(np.eye(p) - Aw).T
    X = rng.normal(0.0, noise_std, size=(n, p)) @ M_inv_T
    # standardize columns (stable for scoring)
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-9)
    return X

# ------------------------- fast Gaussian BIC -------------------------
class GaussianBIC:
    def __init__(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.X = X
        self.n, self.p = X.shape
        self.S = X.T @ X
        self.yTy = np.diag(self.S)

    def _node_rss(self, parents, j):
        if len(parents) == 0:
            y = self.X[:, j]
            return float(((y - y.mean()) ** 2).sum())
        P = np.array(parents, dtype=int)
        Spp = self.S[np.ix_(P, P)]; Spy = self.S[P, j]
        try:
            coef = np.linalg.solve(Spp, Spy)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(Spp) @ Spy
        rss = self.yTy[j] - Spy @ coef
        return float(max(rss, 1e-12))

    def loglik(self, A):
        n, p = self.n, self.p
        total_rss = 0.0
        k = p  # intercepts
        for j in range(p):
            parents = np.where(A[:, j] == 1)[0].tolist()
            total_rss += self._node_rss(parents, j)
            k += len(parents)
        if n <= k:
            return -np.inf, k
        # up to constants; enough for comparisons and Λ_n
        ll = -0.5 * n * p * np.log(2 * np.pi * total_rss / (n * p)) - 0.5 * (n * p)
        return float(ll), k

    def bic(self, A):
        ll, k = self.loglik(A)
        return ll - 0.5 * k * np.log(self.n)

    def per_sample_ll(self, A):
        ll, _ = self.loglik(A)
        return ll / max(self.n, 1)

# ------------------------- tiny CAM prune -------------------------
def cam_prune_linear(A, X, th=0.25):
    from sklearn.linear_model import LinearRegression
    p = A.shape[0]
    out = np.zeros_like(A)
    for j in range(p):
        parents = np.where(A[:, j] == 1)[0]
        if len(parents) == 0: continue
        reg = LinearRegression().fit(X[:, parents], X[:, j])
        keep = parents[np.abs(reg.coef_) > th]
        out[keep, j] = 1
    return out

# ------------------------- GES once (cached) -------------------------
def run_ges_once(X, cache="ges_cache.npy"):
    if cache and os.path.exists(cache):
        try:
            A = np.load(cache)
            return binarize(A)
        except Exception:
            pass
    if not HAS_GES:
        raise RuntimeError("castle GES not installed. `pip install python-castle`")
    print("[opponent] Running GES once ...")
    t0 = time.time()
    model = GES(criterion='bic', method='scatter')
    model.learn(X)
    A = np.array(model.causal_matrix, dtype=float)
    print(f"[opponent] done in {time.time()-t0:.1f}s, edges={int(A.sum())}")
    A = binarize(A)
    if cache:
        np.save(cache, A)
    return A

# ------------------------- DDQN (very small) -------------------------
class QNet(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )

    def forward(self, x):
        return self.net(x)

def build_action_maps(p):
    mapping, rev, idx = {}, {}, 0
    for i in range(p):
        for j in range(p):
            if i == j: continue
            for op in ("add", "remove", "reverse"):
                mapping[idx] = (op, i, j)
                rev[(op, i, j)] = idx
                idx += 1
    return mapping, rev, idx

def valid_mask(A, edge_budget, mapping, rev):
    p = A.shape[0]
    G = nx.DiGraph(A)
    paths = dict(nx.all_pairs_shortest_path_length(G))
    num_edges = int(A.sum())
    mask = np.zeros(len(mapping), dtype=bool)
    for a in range(len(mapping)):
        op, i, j = mapping[a]
        if op == "add":
            if A[i, j] == 0 and num_edges < edge_budget:
                if j not in paths or i not in paths.get(j, {}):
                    mask[a] = True
        elif op == "remove":
            if A[i, j] == 1:
                mask[a] = True
        else:  # reverse
            if A[i, j] == 1 and A[j, i] == 0:
                G.remove_edge(i, j)
                if not nx.has_path(G, i, j):
                    mask[a] = True
                G.add_edge(i, j)
    # if nothing valid, allow any 'add'
    if not mask.any():
        for i in range(p):
            for j in range(p):
                if i != j and A[i, j] == 0:
                    mask[rev[("add", i, j)]] = True
    return mask

def ddqn_candidates(X_val, warm_A, episodes=16, max_steps=12, edge_budget_ratio=2.0,
                    device=torch.device("cpu")):
    p = warm_A.shape[0]
    edge_budget = int(edge_budget_ratio * p)
    mapping, rev, action_size = build_action_maps(p)
    state_size = p * p
    scorer = GaussianBIC(X_val)
    empty_bic = scorer.bic(np.zeros_like(warm_A))

    def reward(A, A_new):
        nb = scorer.bic(A_new)
        return (nb - empty_bic)/p - 0.01*np.sum(A_new) - 0.01

    q = QNet(state_size, action_size).to(device).float()
    t = QNet(state_size, action_size).to(device).float()
    t.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=5e-4)
    tau, gamma = 0.01, 0.99
    eps_start, eps_end = 1.0, 0.05
    eps_decay = max(1, episodes * max_steps)
    steps = 0
    mem = deque(maxlen=40_000)
    batch = 256

    def act(A):
        nonlocal steps
        steps += 1
        m = valid_mask(A, edge_budget, mapping, rev)
        eps = eps_start + (eps_end - eps_start) * min(1.0, steps/eps_decay)
        if np.random.rand() < eps or not m.any():
            idxs = np.where(m)[0]
            return np.random.choice(idxs) if len(idxs) else np.random.randint(action_size)
        st = torch.tensor(A.flatten()[None, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            qv = q(st)
            qv[0, ~torch.tensor(m, device=device)] = -torch.inf
            return int(torch.argmax(qv, dim=1).item())

    def step(A, a):
        op, i, j = mapping[a]
        trial = A.copy(); ok = False
        if op == "add" and trial[i, j] == 0:
            trial[i, j] = 1; ok = True
        elif op == "remove" and trial[i, j] == 1:
            trial[i, j] = 0; ok = True
        elif op == "reverse" and trial[i, j] == 1:
            trial[i, j] = 0; trial[j, i] = 1; ok = True
        if not ok or not is_dag(trial):
            return A, -2.0, True
        return trial, reward(A, trial), False

    def replay():
        if len(mem) < batch: return
        batch_s, batch_a, batch_r, batch_ns, batch_d = [], [], [], [], []
        for s,a,r,ns,d in random.sample(mem, batch):
            batch_s.append(s); batch_a.append(a); batch_r.append(r); batch_ns.append(ns); batch_d.append(d)
        s  = torch.tensor(np.array(batch_s), dtype=torch.float32, device=device)
        a  = torch.tensor(batch_a, dtype=torch.long, device=device).unsqueeze(1)
        r  = torch.tensor(batch_r, dtype=torch.float32, device=device).unsqueeze(1)
        ns = torch.tensor(np.array(batch_ns), dtype=torch.float32, device=device)
        d  = torch.tensor(batch_d, dtype=torch.float32, device=device).unsqueeze(1)
        q_sa = q(s).gather(1, a)
        with torch.no_grad():
            na = torch.argmax(q(ns), dim=1, keepdim=True)
            y  = r + (1.0 - d) * gamma * t(ns).gather(1, na)
        loss = nn.MSELoss()(q_sa, y)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(q.parameters(), 5.0)
        opt.step()
        with torch.no_grad():
            for tp, qp in zip(t.parameters(), q.parameters()):
                tp.data.mul_(1 - tau).add_(tau * qp.data)

    snaps = []
    best_A = warm_A.copy()
    best_bic = scorer.bic(best_A)
    for ep in range(1, episodes+1):
        A = warm_A.copy(); total = 0.0
        for _ in range(max_steps):
            a = act(A)
            ns, r, bad = step(A, a)
            mem.append((A.flatten().astype(np.float32), a, float(r), ns.flatten().astype(np.float32), float(bad)))
            A = ns; total += r; replay()
            if bad: break
        Ab = binarize(A); B = scorer.bic(Ab)
        if B > best_bic: best_bic, best_A = B, Ab.copy()
        snaps.append(Ab)

    uniq, seen = [], set()
    for A in [best_A] + snaps:
        key = A.tobytes()
        if key not in seen: seen.add(key); uniq.append(A)
    A_cam = cam_prune_linear(best_A, X_val, th=0.25)
    if A_cam.tobytes() not in seen: uniq.append(A_cam)
    return uniq

# ------------------------- candidate set -------------------------
def build_candidates(X_train, X_val, use_ges=True, episodes=16, device=torch.device("cpu")):
    p = X_train.shape[1]
    A_opp = run_ges_once(X_train, cache="ges_cache.npy") if use_ges else np.zeros((p,p), int)
    ddqn_cands = ddqn_candidates(X_val, A_opp, episodes=episodes, device=device)
    A_opp_cam = cam_prune_linear(A_opp, X_val, th=0.25)
    C = [A_opp, A_opp_cam] + ddqn_cands
    # de-duplicate
    uniq, seen = [], set()
    for A in C:
        key = A.tobytes()
        if key not in seen:
            seen.add(key); uniq.append(A)
    return uniq

# --- prerequisites you already have somewhere above ---
# GaussianBIC (per-sample loglik via node RSS), cam_prune_linear, run_ges_once, ddqn_candidates, etc.
# C : fixed candidate set (list of adjacency matrices), built once with X_train/X_val
# X_ref : large fixed validation/holdout pool (e.g., half of your big pool) to estimate μ(A)
# k(A) : comes from scorer.loglik(A) as the number of parameters used in BIC

import numpy as np
import matplotlib.pyplot as plt

def estimate_mu_and_k(C, X_ref):
    """Estimate μ(A)=E s_A(X) per sample using a large fixed pool X_ref; also get k(A)."""
    scorer = GaussianBIC(X_ref)
    mu = []
    kvals = []
    for A in C:
        ll, k = scorer.loglik(A)
        mu.append(ll / scorer.n)
        kvals.append(k)
    return np.array(mu, dtype=float), np.array(kvals, dtype=int)

def Lambda_n_all(mu, k, n):
    return mu - 0.5 * (k / n) * np.log(n)

def empirical_error_rate(C, n, M, sem_sampler, bic_scorer_class=GaussianBIC, seed=0):
    """
    For a fixed n, draw M fresh datasets, compute empirical BIC S_n(A),
    and return the mis-selection rate w.r.t. the population winner at this n.
    `sem_sampler(n, m)` must return an (n,p) sample for repetition m.
    """
    rng = np.random.default_rng(seed)
    # find population winner at this n using μ and k computed on X_ref outside
    # (we’ll pass Λ_n vector and its argmax from outside for speed/clarity)
    raise NotImplementedError("Use empirical_error_rate_with_popwinner below.")

def empirical_error_rate_with_popwinner(C, n, M, pop_winner_idx, sem_sampler, seed=0):
    """
    Same as above but we pass the index of the population winner at this n
    to avoid recomputing Λ_n inside.
    """
    rng = np.random.default_rng(seed)
    mistakes = 0
    for m in range(M):
        Xn = sem_sampler(n, m)              # (n,p)
        scorer = GaussianBIC(Xn)            # S_n from this sample
        scores = [scorer.bic(A) for A in C]
        emp_winner = int(np.argmax(scores))
        if emp_winner != pop_winner_idx:
            mistakes += 1
    return mistakes / M

def make_sem_sampler(Aw, noise_std=1.0, standardize=True, base_seed=12345):
    """
    Returns a function sem_sampler(n,m) that draws an (n,p) sample
    from the *same* underlying SEM defined by weighted adj Aw.
    """
    p = Aw.shape[0]
    I = np.eye(p)
    M = I - Aw
    InvT = np.linalg.inv(M.T)

    def sampler(n, m):
        rng = np.random.default_rng(base_seed + m + 10_000*n)
        X = rng.normal(0.0, noise_std, size=(n, p)) @ InvT
        if standardize:
            X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-9)
        return X
    return sampler

def joint_plot_theorem3(C, X_ref, Aw, n_list=(800,1200,1600,2000),
                        M=64, L_proxy=1.0, c_proxy=1.0, out="theorem3_joint.png",
                        seed=0):
    """
    - C: fixed candidate set (GES + DDQN snapshots), already built ONCE.
    - X_ref: big holdout to estimate μ(A) and k(A) once.
    - Aw: true weighted adjacency used to generate Monte Carlo datasets (same SEM).
    - n_list: sample sizes to probe.
    - M: repetitions per n to estimate empirical error.
    - L_proxy, c_proxy: constants in bound shape  exp(-c n Δ_n^2 / L^2).
    """
    # 1) μ and k from fixed pool
    mu, k = estimate_mu_and_k(C, X_ref)

    # 2) SEM sampler for Monte Carlo error estimation
    sampler = make_sem_sampler(Aw, base_seed=seed)

    errs, bounds, gaps = [], [], []

    for n in n_list:
        # population Λ_n and winner at this n
        Lam = Lambda_n_all(mu, k, n)
        winner = int(np.argmax(Lam))
        sorted_idx = np.argsort(Lam)[::-1]
        runner = int(sorted_idx[1]) if len(sorted_idx) > 1 else winner
        Delta_n = float(Lam[winner] - Lam[runner])
        gaps.append(Delta_n)

        # empirical mis-selection at this n via M Monte Carlo datasets
        err = empirical_error_rate_with_popwinner(C, n, M, winner, sampler, seed=seed)
        errs.append(err)

        # exponential bound *shape* with the current Δ_n
        bound_shape = 2 * len(C) * np.exp(- c_proxy * n * (Delta_n**2) / (L_proxy**2))
        bounds.append(min(1.0, float(bound_shape)))

        print(f"[n={n}] winner={winner}  Δ_n={Delta_n:.4g}  err~{err:.3f}  bound~{bounds[-1]:.3f}")

    # 3) One joint figure
    fig, ax1 = plt.subplots(figsize=(7.2, 5.2))

    # Left y-axis: empirical error
    ln1 = ax1.plot(n_list, errs, "o-", color="C0", label="Empirical mis-selection")
    ax1.set_xlabel("n (sample size)", fontsize=14)
    ax1.set_ylabel("Probability", fontsize=14)
    ax1.tick_params(axis="both", labelsize=12)
    ax1.set_ylim(bottom=0)
    ax1.grid(ls=":", alpha=0.6)

    # Right y-axis: gap
    ax2 = ax1.twinx()
    ln2 = ax2.plot(n_list, gaps, "d-", color="C2", label=r"Gap $\widehat{\Delta}_n$")
    ax2.set_ylabel(r"$\widehat{\Delta}_n$", color="C2", fontsize=14)
    ax2.tick_params(axis="both", labelsize=12)

    # Combine legends from both axes
    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    ax1.legend(
        lines, labels,
        loc="lower right",
        fontsize=13,
        frameon=True,
        framealpha=0.9,
        fancybox=True,
        borderpad=1.2,
        handlelength=2.5,
        labelspacing=0.8
    )

    plt.title("Empirical error, exponential bound, and gap vs n", fontsize=15)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    print(f"[saved] {out}")

    # Optional: also save the n–gap curve alone
    plt.figure(figsize=(7.2, 4.4))
    plt.plot(n_list, gaps, "o-", color="C2", label=r"$\widehat{\Delta}_n$")
    plt.xlabel("n", fontsize=14)
    plt.ylabel(r"$\widehat{\Delta}_n$", fontsize=14)
    plt.title("Per-sample gap vs n (fixed candidate set)", fontsize=15)
    plt.tick_params(axis="both", labelsize=12)
    plt.grid(ls=":", alpha=0.6)
    plt.tight_layout()
    plt.legend(loc="lower right")
    plt.savefig("n_vs_gap.png", dpi=160)
    print("[saved] n_vs_gap.png")

    return np.array(errs), np.array(bounds), np.array(gaps)



# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p", type=int, default=30)
    ap.add_argument("--edge_prob", type=float, default=3.0/30.0)
    ap.add_argument("--n_pool", type=int, default=3000,  # large pool to split once (train/holdout)
                    help="big pool used once to build C and to form X_ref for μ(A)")
    ap.add_argument("--val_frac", type=float, default=0.5,
                    help="fraction of the pool used as X_ref (fixed holdout for μ)")
    ap.add_argument("--episodes", type=int, default=16,  # tiny DDQN
                    help="DDQN episodes for candidate snapshots")
    ap.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--use64", action="store_true", help="use float64 in Q-net")
    ap.add_argument("--trials", type=int, default=64,    # Monte-Carlo repetitions per n
                    help="repeats per n to estimate empirical mis-selection")
    ap.add_argument("--n_list", type=int, nargs="+", default=[400,600,800,1000],
                    help="n values to probe")
    ap.add_argument("--c_proxy", type=float, default=1.0,
                    help="bound shape constant (just a scale for the orange curve)")
    ap.add_argument("--L_proxy", type=float, default=1.0,
                    help="Lipschitz proxy (scale in the bound shape)")
    args = ap.parse_args()

    set_seed(args.seed)
    device, dtype = get_device_dtype(use64=args.use64, device=args.device)
    print(f"[run] device={device.type}, dtype={dtype}")

    # 1) Generate a true SEM and one big pool; split once into build/holdout.
    Aw = random_dag(args.p, edge_prob=args.edge_prob, seed=args.seed)
    X_pool = sample_sem(Aw, n=args.n_pool, seed=args.seed)
    cut = int((1.0 - args.val_frac) * X_pool.shape[0])
    X_tr, X_ref = X_pool[:cut], X_pool[cut:]
    print(f"[data] X_tr={X_tr.shape}  X_ref={X_ref.shape}  (X_ref is fixed holdout for μ)")

    # 2) Build a FIXED candidate set C (opponent+DDQN+CAM), ONCE.
    # C = build_candidates(X_tr, X_ref, use_ges=True, episodes=args.episodes, device=device, dtype=dtype)
    C = build_candidates(X_tr, X_ref, use_ges=True, episodes=args.episodes, device=device)

    # de-duplicate, just in case
    uniq, seen = [], set()
    for A in C:
        key = A.tobytes()
        if key not in seen:
            seen.add(key); uniq.append(A)
    C = uniq
    print(f"[candidates] |C|={len(C)}")

    # 3) Joint plot for Theorem 3: empirical error (blue), exponential bound shape (orange), and gap (green).
    errs, bounds, gaps = joint_plot_theorem3(
        C=C,
        X_ref=X_ref,          # fixed holdout for μ(A), k(A)
        Aw=Aw,                # true SEM for Monte-Carlo datasets
        n_list=args.n_list,
        M=args.trials,
        L_proxy=args.L_proxy,
        c_proxy=args.c_proxy,
        out="theorem3_joint_new.png",
        seed=args.seed + 7
    )

    # Optionally, also save the n–gap curve alone (sometimes nice to have)
    plt.figure(figsize=(7.2, 4.4))
    plt.plot(args.n_list, gaps, "o-", color="C2")
    plt.xlabel("n")
    plt.ylabel(r"$\widehat{\Delta}_n$")
    plt.title("Per-sample gap vs n (fixed candidate set)")
    plt.grid(ls=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("n_vs_gap.png", dpi=160)
    print("[saved] n_vs_gap.png")


if __name__ == "__main__":
    main()
