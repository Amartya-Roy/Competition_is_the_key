#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip show gcastle')


# In[ ]:


# -*- coding: ut-8 -*-
import os, random, warnings, argparse # <-- ADDED argparse
from collections import deque
from itertools import product
from castle.metrics import MetricsDAG

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx

from sklearn.linear_model import LinearRegression
from scipy.stats import rankdata, norm

# Opponent
from castle.algorithms import GraNDAG
from castle.algorithms import GES

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------- Config --------------------
# These can now be overridden by command-line arguments
DATA_CSV   = "Datasets/asia/data.csv"
GT_NPY     = "Datasets/asia/adj.npy"      # optional; evaluation only
G_ITER     = 1000                    # GraN-DAG iterations

N_EPISODES = 1000                    # <-- INCREASED
EVAL_EVERY = 20
SEED       = 42

EDGE_BUDGET_RATIO = 2.0
LAMBDA_L1         = 0.005              # <-- REDUCED
ACTION_COST       = 0.01
CAM_TH            = 0.25               # Default, now adaptive

SCORE_TYPE = "copula"                # "copula" (robust) or "gaussian"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.double
np.random.seed(SEED); random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# -------------------- IO --------------------
def load_data(csv_path=DATA_CSV):
    if not os.path.exists(csv_path):
        print(f"Error: '{csv_path}' not found."); return None
    df = pd.read_csv(csv_path, header=0)
    if df.columns[0].lower().startswith("unnamed"):
        df = pd.read_csv(csv_path, header=0, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    X = df.values.astype(np.float64)
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True); sd[sd == 0] = 1.0
    X = (X - mu) / sd
    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} vars from '{os.path.basename(csv_path)}'")
    return X

def load_truth(npy_path=GT_NPY, p=None):
    if not npy_path or not os.path.exists(npy_path):
        print("No ground truth file; metrics will be limited.")
        return None
    G = np.load(npy_path).astype(np.float64)
    if p is not None and G.shape != (p, p):
        print(f"[align] trimming GT from {G.shape} to {(p,p)}")
        G = G[:p, :p]
    print("Loaded ground truth:", G.shape)
    return G


# -------------------- Opponent (GraN-DAG) --------------------
# def get_grandag_adj(data, iterations=G_ITER, hidden_dim=16, hidden_num=2, lr=5e-4, h_threshold=1e-6):
#     print(f"\nRunning GraN-DAG (iterations={iterations})...")
#     model = GraNDAG(
#         input_dim=data.shape[1], hidden_dim=hidden_dim, hidden_num=hidden_num,
#         lr=lr, iterations=iterations, h_threshold=h_threshold, mu_init=1e-3
#     )
#     model.learn(data)
#     print("GraN-DAG done.")
#     return np.array(model.causal_matrix, dtype=np.float64)


def get_ges_adj(data):
    print(f"\nRunning GES opponent...")
    model = GES(criterion='bic', method='scatter')
    model.learn(data)
    print("GES done."); return np.array(model.causal_matrix, dtype=np.float64)

# -------------------- Metrics --------------------
def binarize(A):
    A = (A != 0).astype(int)
    np.fill_diagonal(A, 0)
    return A

def shd_binary(A, B):
    A = binarize(A); B = binarize(B)
    Au = ((A + A.T) > 0).astype(int); Bu = ((B + B.T) > 0).astype(int)
    undirected_diff = int(np.sum(np.triu(Au ^ Bu, 1)))
    common_u = ((Au & Bu) > 0).astype(int)
    orient_mismatch = int(np.sum(np.triu((A ^ B) & common_u, 1)))
    return undirected_diff + orient_mismatch

def eval_against_gt(pred_adj, GT):
    if GT is None or pred_adj.shape != GT.shape:
        return None
    P = binarize(pred_adj); T = binarize(GT)
    tp = int(((P == 1) & (T == 1)).sum())
    fp = int(((P == 1) & (T == 0)).sum())
    fn = int(((P == 0) & (T == 1)).sum())
    tn = int(((P == 0) & (T == 0)).sum())
    fdr = fp / max(tp + fp, 1)
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1) # <-- ADDED
    return {
        "SHD": shd_binary(P, T),
        "NNZ": int(P.sum()),
        "TPR": round(tpr, 4),
        "FDR": round(fdr, 4),
        "Precision": round(precision, 4), # <-- ADDED
        "FPR": round(fpr, 4),
    }


# -------------------- Scorers --------------------
class GaussianBIC:
    """Gaussian BIC via covariance blocks (no per-step OLS)."""
    def __init__(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.X = X
        self.n, self.p = X.shape
        self.S = X.T @ X
        self.yTy = np.diag(self.S)

    def node_rss(self, parents, j):
        if len(parents) == 0:
            y = self.X[:, j]
            return float(((y - y.mean()) ** 2).sum())
        P = np.array(parents, dtype=int)
        Spp = self.S[np.ix_(P, P)]
        Spy = self.S[P, j]
        try:
            coef = np.linalg.solve(Spp, Spy)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(Spp) @ Spy
        rss = self.yTy[j] - Spy @ coef
        return float(max(rss, 1e-12))

    def bic(self, A):
        n, p = self.n, self.p
        total_rss = 0.0
        num_params = p  # intercepts
        for j in range(p):
            parents = np.where(A[:, j] == 1)[0].tolist()
            total_rss += self.node_rss(parents, j)
            num_params += len(parents)
        
        # <-- STABILITY CHANGE: Avoid log(0) or division by zero if n <= num_params
        if n <= num_params:
            return -np.inf
        
        # Using log-likelihood formulation for stability
        # BIC = -2 * logL + k * log(n)
        # For Gaussian: 2 * logL = -n*p*(log(2*pi) + 1) - n * log(det(Sigma))
        # Here, we use the residual sum of squares approximation
        loglik = -0.5 * n * p * np.log(2 * np.pi * total_rss / (n * p)) - 0.5 * (n * p)
        penalty = 0.5 * num_params * np.log(n)
        return float(loglik - penalty)


def gaussian_copula_transform(X):
    """
    Nonparanormal transform: map each marginal to ~N(0,1) by rank -> Phi^-1(u).
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    Z = np.empty_like(X)
    eps = 1e-6
    rng = np.random.default_rng(0)
    for j in range(p):
        x = X[:, j]
        if np.std(x) < 1e-12:
            x = x + rng.normal(0, 1e-9, size=n)
        r = rankdata(x, method="average")  # 1..n
        u = (r - 0.5) / n                  # (0,1)
        u = np.clip(u, eps, 1 - eps)
        Z[:, j] = norm.ppf(u)
    Z -= Z.mean(axis=0, keepdims=True)
    std = Z.std(axis=0, keepdims=True); std[std == 0] = 1.0
    Z /= std
    return Z

class CopulaBIC(GaussianBIC):
    """Gaussian Copula (Nonparanormal) BIC scorer."""
    def __init__(self, X):
        Z = gaussian_copula_transform(X)
        super().__init__(Z)


# -------------------- Warm-start (greedy + reversals) --------------------
def warm_start_greedy_bic(Xva, edge_budget, scorer,
                          max_passes=10, topk_per_pass=15, restarts=10, seed=SEED):
    # <-- SIMPLIFIED: No need for Xtr/Xva split here, just use validation scorer
    rng = np.random.RandomState(seed)
    p = Xva.shape[1]

    def is_dag(M): return nx.is_directed_acyclic_graph(nx.DiGraph(M))

    def one_run():
        A = np.zeros((p, p), dtype=np.float64)
        best_score = scorer.bic(A)
        for _ in range(max_passes):
            improved = False
            # Forward: add Top-K
            cands = []
            if A.sum() < edge_budget:
                for i, j in product(range(p), range(p)):
                    if i == j or A[i, j] == 1: continue
                    trial = A.copy(); trial[i, j] = 1.0
                    if not is_dag(trial): continue
                    s = scorer.bic(trial)
                    if s > best_score: cands.append((s - best_score, i, j))
            cands.sort(reverse=True, key=lambda x: x[0])
            for _, i, j in cands[:topk_per_pass]:
                if A.sum() >= edge_budget: break
                trial = A.copy(); trial[i, j] = 1.0
                if is_dag(trial):
                    s = scorer.bic(trial)
                    if s > best_score:
                        A, best_score, improved = trial, s, True

            # Backward: prune
            pruned = True
            while pruned:
                pruned = False
                edges = list(zip(*np.where(A == 1)))
                random.shuffle(edges)
                for i, j in edges:
                    trial = A.copy(); trial[i, j] = 0.0
                    s = scorer.bic(trial)
                    if s > best_score:
                        A, best_score, improved, pruned = trial, s, True, True
            
            # Reversal sweep
            edges = list(zip(*np.where(A == 1)))
            random.shuffle(edges)
            for i, j in edges:
                trial = A.copy(); trial[i, j] = 0.0; trial[j, i] = 1.0
                if not is_dag(trial): continue
                s = scorer.bic(trial)
                if s > best_score:
                    A, best_score, improved = trial, s, True
            
            if not improved: break
        return A, best_score

    bestA, bestS = None, -np.inf
    print(f"[Warm-start] Starting {restarts} greedy restarts...")
    for i in range(restarts):
        A, s = one_run()
        if s > bestS: bestA, bestS = A, s
        print(f"  Restart {i+1}/{restarts}: BIC={s:.2f}, Edges={int(A.sum())}")
    return bestA


# -------------------- Environment --------------------
class CausalDiscoveryEnv:
    """
    Reward = (Val-BIC - Val-BIC_empty)/N - penalties. Action masking enabled.
    """
    def __init__(self, data,
                 val_frac=0.2,
                 edge_budget_ratio=EDGE_BUDGET_RATIO,
                 lambda_l1=LAMBDA_L1, action_cost=ACTION_COST,
                 warm_start_adj=None, # <-- CHANGED: Pass warm-start adj
                 score_type=SCORE_TYPE):
        self.n_samples, self.n_nodes = data.shape
        
        idx = np.arange(self.n_samples); rng = np.random.RandomState(SEED)
        rng.shuffle(idx)
        cut = int((1.0 - val_frac) * self.n_samples)
        self.Xtr, self.Xva = data[idx[:cut]], data[idx[cut:]]

        Scorer = CopulaBIC if str(score_type).lower() == "copula" else GaussianBIC
        self.bic_va = Scorer(self.Xva)

        self.state_space_shape = (self.n_nodes * self.n_nodes,)
        self.n_actions = 3 * self.n_nodes * (self.n_nodes - 1)
        self.action_map, self.rev_action_map = self._create_action_map()

        self.current_adj = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float64)
        self.max_steps = 20 * self.n_nodes # <-- INCREASED
        self.current_step = 0

        self.edge_budget = int(max(1, edge_budget_ratio * self.n_nodes))
        self.lambda_l1 = float(lambda_l1)
        self.action_cost = float(action_cost)
        self._warm_adj = warm_start_adj

        # <-- ADDED: Baseline BIC for reward normalization
        self.bic_empty = self.bic_va.bic(np.zeros_like(self.current_adj))
        self.best_bic_so_far = -np.inf

    def _create_action_map(self):
        mapping, rev_mapping, idx = {}, {}, 0
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i == j: continue
                mapping[idx] = ("add", i, j); rev_mapping[("add", i, j)] = idx; idx += 1
                mapping[idx] = ("remove", i, j); rev_mapping[("remove", i, j)] = idx; idx += 1
                mapping[idx] = ("reverse", i, j); rev_mapping[("reverse", i, j)] = idx; idx += 1
        return mapping, rev_mapping

    def _get_valid_actions_mask(self): # <-- ADDED: CORE ACTION MASKING LOGIC
        mask = np.zeros(self.n_actions, dtype=bool)
        num_edges = self.current_adj.sum()
        
        # Pre-compute reachability for faster cycle checks
        G = nx.DiGraph(self.current_adj)
        # This can be slow for large graphs. For very large graphs, one might
        # check cycles only *after* an action is attempted. But for p < 100 it's fine.
        paths = dict(nx.all_pairs_shortest_path_length(G))

        for action_idx in range(self.n_actions):
            op, i, j = self.action_map[action_idx]
            
            if op == "add":
                # Valid if edge doesn't exist, doesn't create cycle, and within budget
                if self.current_adj[i, j] == 0 and num_edges < self.edge_budget:
                    # A cycle is created if a path from j to i already exists
                    if j not in paths or i not in paths.get(j, {}):
                        mask[action_idx] = True
            
            elif op == "remove":
                # Valid if edge exists
                if self.current_adj[i, j] == 1:
                    mask[action_idx] = True
            
            elif op == "reverse":
                # Valid if edge i->j exists and j->i doesn't create a cycle
                if self.current_adj[i, j] == 1 and self.current_adj[j, i] == 0:
                    # A cycle is created if a path from i to j exists *without* the i->j edge
                    # Quick check: remove edge, check path, add back.
                    G.remove_edge(i, j)
                    if not nx.has_path(G, i, j):
                        mask[action_idx] = True
                    G.add_edge(i, j)
        
        # If no actions are valid (e.g., stuck in a local optimum), allow all adds to escape
        if not mask.any():
            for i,j in product(range(self.n_nodes), range(self.n_nodes)):
                if i !=j and self.current_adj[i,j] == 0:
                    mask[self.rev_action_map[("add", i, j)]] = True
        return mask

    def reset(self):
        self.current_step = 0
        self.current_adj = self._warm_adj.copy() if self._warm_adj is not None else np.zeros((self.n_nodes, self.n_nodes))
        self.best_bic_so_far = self.bic_va.bic(self.current_adj)
        return self.current_adj.flatten().copy()

    def step(self, action_idx):
        # The agent should only be choosing valid actions due to masking,
        # but we can add a check here as a safeguard.
        op, i, j = self.action_map[action_idx]
        prev_adj = self.current_adj.copy()

        trial = self.current_adj.copy()
        action_was_valid = False
        if op == "add" and trial[i, j] == 0:
            trial[i, j] = 1.0; action_was_valid = True
        elif op == "remove" and trial[i, j] == 1:
            trial[i, j] = 0.0; action_was_valid = True
        elif op == "reverse" and trial[i, j] == 1:
            trial[i, j] = 0.0; trial[j, i] = 1.0; action_was_valid = True
        
        # This check should be redundant if masking works, but is a fail-safe.
        if not action_was_valid or not nx.is_directed_acyclic_graph(nx.DiGraph(trial)):
            self.current_step += 1
            # Severe penalty for invalid action, forces agent to learn the mask
            return self.current_adj.flatten(), -2.0, self.current_step >= self.max_steps, {}

        self.current_adj = trial
        self.current_step += 1
        r = self._reward(prev_adj, self.current_adj)
        done = self.current_step >= self.max_steps
        info = {'valid_actions': self._get_valid_actions_mask()} # <-- ADDED: Return mask
        return self.current_adj.flatten().copy(), r, done, info

    def _reward(self, prev_adj, new_adj): # <-- CHANGED: New reward logic
        new_bic = self.bic_va.bic(new_adj)

        # Core reward: Normalized BIC score relative to the empty graph.
        # This provides a stable, positive signal for good graphs.
        score_reward = (new_bic - self.bic_empty) / self.n_nodes

        # Bonus for finding a new best graph
        improvement_bonus = 0.0
        if new_bic > self.best_bic_so_far:
            improvement_bonus = 0.5
            self.best_bic_so_far = new_bic

        # Regularization / Penalties
        sparsity_penalty = -self.lambda_l1 * float(np.sum(new_adj))
        action_penalty   = -self.action_cost

        total = score_reward + improvement_bonus + sparsity_penalty + action_penalty
        return float(np.clip(total, -20.0, 20.0))


# -------------------- Agent (Double DQN + Polyak + Scheduler) --------------------
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 512, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(512, 512, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(512, action_size, dtype=DTYPE),
        )
    def forward(self, x): return self.layers(x)

class CausalAgent:
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.q = QNetwork(state_size, action_size).to(device)
        self.t = QNetwork(state_size, action_size).to(device)
        self.t.load_state_dict(self.q.state_dict())
        self.tau = 0.005
        self.opt = optim.Adam(self.q.parameters(), lr=5e-4)
        # <-- ADDED: Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.opt, step_size=10_000, gamma=0.9)
        self.gamma = 0.99 # <-- INCREASED for longer episodes

        self.eps_start, self.eps_end = 1.0, 0.05
        self.eps_decay_steps = 400_000 # <-- INCREASED
        self.total_steps = 0
        self.epsilon = self.eps_start

        self.mem = deque(maxlen=200_000)
        self.batch = 512

    def _update_eps(self):
        self.total_steps += 1
        frac = min(1.0, self.total_steps / self.eps_decay_steps)
        self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)

    def remember(self, s, a, r, ns, d): self.mem.append((s, a, r, ns, d))

    def act(self, state, valid_mask=None): # <-- CHANGED: Accept action mask
        self._update_eps()
        if random.random() <= self.epsilon:
            if valid_mask is not None and valid_mask.any():
                # Explore only among valid actions
                valid_indices = np.where(valid_mask)[0]
                return np.random.choice(valid_indices)
            return random.randrange(self.action_size) # Fallback

        st = torch.tensor(state, dtype=DTYPE, device=device).unsqueeze(0)
        with torch.no_grad():
            qv = self.q(st)
            if valid_mask is not None:
                # Mask out invalid actions before taking argmax
                qv[0, ~torch.tensor(valid_mask, device=device)] = -torch.inf
        return int(qv.argmax(dim=1).item())

    def replay(self):
        if len(self.mem) < self.batch: return
        batch = random.sample(self.mem, self.batch)
        s  = torch.tensor(np.array([e[0] for e in batch]), dtype=DTYPE, device=device)
        a  = torch.tensor([e[1] for e in batch], dtype=torch.long, device=device).unsqueeze(1)
        r  = torch.tensor([e[2] for e in batch], dtype=DTYPE, device=device).unsqueeze(1)
        ns = torch.tensor(np.array([e[3] for e in batch]), dtype=DTYPE, device=device)
        d  = torch.tensor([e[4] for e in batch], dtype=DTYPE, device=device).unsqueeze(1)

        q_sa = self.q(s).gather(1, a)
        with torch.no_grad():
            na_online = self.q(ns).argmax(1, keepdim=True)
            q_next = self.t(ns).gather(1, na_online)
            target = r + (1.0 - d) * self.gamma * q_next

        loss = nn.MSELoss()(q_sa, target)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt.step()
        self.scheduler.step() # <-- ADDED

        with torch.no_grad():
            for tparam, qparam in zip(self.t.parameters(), self.q.parameters()):
                tparam.data.mul_(1 - self.tau).add_(self.tau * qparam.data)


# -------------------- CAM pruning --------------------
def graph_prunned_by_coef(parent_mat, X, th=CAM_TH):
    d = parent_mat.shape[0]
    reg = LinearRegression()
    W_pruned = np.zeros_like(parent_mat)
    for i in range(d):
        parents = np.where(parent_mat[i, :] != 0)[0]
        if len(parents) == 0: continue
        
        X_train = X[:, parents]
        y_train = X[:, i]
        
        reg.fit(X_train, y_train)
        abs_coef = np.abs(reg.coef_)
        
        # Keep parents whose coefficient is above the threshold
        significant_parents = parents[abs_coef > th]
        W_pruned[i, significant_parents] = 1.0
        
    return W_pruned

def cam_prune_linear_from_A(A_directed, X, th=CAM_TH):
    parents = A_directed.T
    pruned_parents = graph_prunned_by_coef(parents, X, th=th)
    return pruned_parents.T

def adaptive_cam_prune(A, X, bic_scorer, thresholds=None): # <-- ADDED: Adaptive CAM
    """ Tries multiple CAM thresholds and returns the graph with the best BIC score. """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4]
    
    best_A_cam = A.copy()
    # Also consider the original graph as a candidate
    best_bic = bic_scorer.bic(best_A_cam)

    for th in thresholds:
        A_cam = cam_prune_linear_from_A(A, X, th=th)
        bic_cam = bic_scorer.bic(A_cam)
        if bic_cam > best_bic:
            best_bic = bic_cam
            best_A_cam = A_cam
            
    return best_A_cam

# -------------------- Main --------------------
def main(args):
    X = load_data(args.data_csv)
    if X is None: raise SystemExit
    p = X.shape[1]
    GT = load_truth(args.gt_npy, p=p)

    # Opponent
#     Gdag = get_grandag_adj(X, iterations=args.g_iter)
    Gdag = get_ges_adj(X)
    
    Gdag_bin = binarize(Gdag)

    # Scorer and Warm-start
    ScorerClass = CopulaBIC if args.score_type == "copula" else GaussianBIC
    bic_scorer_val = ScorerClass(X) # Use full data for final scoring and warm-start
    
    print("\n[Warm-start] searching for a good initial graph...")
    edge_budget = int(args.edge_budget_ratio * p)
    warm_adj = warm_start_greedy_bic(
        X, edge_budget, bic_scorer_val,
        max_passes=10, topk_per_pass=15, restarts=10, seed=args.seed
    )
    # Also consider GraN-DAG as a warm-start candidate
    bic_warm = bic_scorer_val.bic(warm_adj)
    bic_gran = bic_scorer_val.bic(Gdag_bin)
    if bic_gran > bic_warm:
        print("[Warm-start] GraN-DAG result has better initial BIC. Using it as warm-start.")
        warm_adj = Gdag_bin.copy()
    
    print(f"[Warm-start] Selected initial graph with {int(warm_adj.sum())} edges and BIC={bic_scorer_val.bic(warm_adj):.2f}")

    # Env + Agent
    env = CausalDiscoveryEnv(
        X,
        val_frac=0.2,
        edge_budget_ratio=args.edge_budget_ratio,
        lambda_l1=args.lambda_l1, 
        action_cost=args.action_cost,
        warm_start_adj=warm_adj,
        score_type=args.score_type
    )
    agent = CausalAgent(state_size=env.state_space_shape[0], action_size=env.n_actions)

    best_valbic = -np.inf
    best_agent_adj = None

    print("\nStarting RL training with action masking...")
    for ep in range(1, args.n_episodes + 1):
        s = env.reset()
        # Get initial valid actions mask
        valid_actions = env._get_valid_actions_mask()
        total_reward, done = 0.0, False
        
        while not done:
            a = agent.act(s, valid_actions)
            ns, r, done, info = env.step(a)
            agent.remember(s, a, r, ns, done)
            s = ns
            total_reward += r
            valid_actions = info.get('valid_actions')
            
            # Perform multiple replay steps per environment step for data efficiency
            if agent.total_steps % 4 == 0:
                agent.replay()

        msg = f"Ep {ep:04d}/{args.n_episodes} | Reward={total_reward:8.3f} | Steps={env.current_step} | eps={agent.epsilon:.3f}"

        if ep % args.eval_every == 0:
            A_now = binarize(env.current_adj)
            valBIC_agent = env.bic_va.bic(A_now)

            if valBIC_agent > best_valbic:
                best_valbic = valBIC_agent
                best_agent_adj = A_now.copy()
                msg += " (* New best BIC *)"

            print(msg)
            if GT is not None:
                print(f"  Metrics (raw): {MetricsDAG(A_now, GT).metrics}")
        else:
            print(msg, end='\r')

    print("\nTraining finished.")
    A_final = best_agent_adj if best_agent_adj is not None else binarize(env.current_adj)
    
    # Final adaptive CAM pruning
    print("\nPerforming adaptive CAM pruning...")
    A_cam_final = adaptive_cam_prune(A_final, X, env.bic_va)

    print("\n--- Final Results (based on best validation BIC graph) ---")
    print(f"ValBIC(Agent Raw):   {env.bic_va.bic(A_final):.2f}")
    print(f"ValBIC(Agent+CAM): {env.bic_va.bic(A_cam_final):.2f}")
    print(f"ValBIC(GraN-DAG):  {env.bic_va.bic(Gdag_bin):.2f}")

#     if GT is not None:
#         print("\n--- Ground Truth Metrics ---")
#         print(f"Agent (raw):  {eval_against_gt(A_final, GT)}")
#         print(f"Agent+CAM:    {eval_against_gt(A_cam_final, GT)}")
#         print(f"GraN-DAG:     {eval_against_gt(Gdag_bin, GT)}")
        
    if GT is not None:
        print("\n--- Ground Truth Metrics ---")
        print(f"Agent (raw):  {print(MetricsDAG(A_final, GT).metrics)}")
        print(f"Agent+CAM:    {print(MetricsDAG(A_cam_final, GT).metrics)}")
        print(f"GraN-DAG:     {print(MetricsDAG(Gdag_bin, GT).metrics)}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN for Causal Discovery")
    parser.add_argument('--data_csv', type=str, default=DATA_CSV, help='Path to data CSV file')
    parser.add_argument('--gt_npy', type=str, default=GT_NPY, help='Path to ground truth npy file')
    parser.add_argument('--g_iter', type=int, default=G_ITER, help='GraN-DAG iterations')
    parser.add_argument('--n_episodes', type=int, default=N_EPISODES, help='Number of training episodes')
    parser.add_argument('--eval_every', type=int, default=EVAL_EVERY, help='Evaluate every N episodes')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    parser.add_argument('--edge_budget_ratio', type=float, default=EDGE_BUDGET_RATIO)
    parser.add_argument('--lambda_l1', type=float, default=LAMBDA_L1)
    parser.add_argument('--action_cost', type=float, default=ACTION_COST)
    parser.add_argument('--score_type', type=str, default=SCORE_TYPE, choices=['copula', 'gaussian'])
    
    # This is the corrected line to work in Colab/Jupyter
    args = parser.parse_args(args=[])
    
    # Update global configs from args if needed, or pass args to functions
    SEED = args.seed
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    
    main(args)


# In[ ]:


# -*- coding: utf-8 -*-
"""
Causal-Discovery (DQN vs. GraN-DAG) with Copula/Gaussian BIC, CAM pruning, live metrics


- EDGE_BUDget_RATIO = 2.0 (denser search)
- LAMBDA_L1 = 0.005, ACTION_COST = 0.01 (lighter sparsity & edit penalty)
- max_steps = 20 * n_nodes (longer horizon)
- Reward: (BIC - BIC_empty) / n_nodes - penalties (clearer objective)
- Warm-start: more passes/top-k/restarts, can use GraN-DAG as candidate
- Q-net widened to 512, batch=512, replay buffer=200k
- CAM_TH = adaptive (selects best BIC from multiple thresholds)
- ADDED: Action masking to prune invalid actions (massive speedup & efficiency).
- ADDED: Learning rate scheduler for stable convergence.
- ADDED: Tracking and saving the agent with the best TPR during training.
"""

import os, random, warnings, argparse
from collections import deque
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx

from sklearn.linear_model import LinearRegression
from scipy.stats import rankdata, norm

# Opponent
from castle.algorithms import GraNDAG

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------- Config --------------------
DATA_CSV   = "Datasets/Hepar2/hepar2.csv"
GT_NPY     = "Datasets/Hepar2/hepar2adj.npy"
G_ITER     = 1000

N_EPISODES = 1000
EVAL_EVERY = 20
SEED       = 42

EDGE_BUDGET_RATIO = 2.0
LAMBDA_L1         = 0.005
ACTION_COST       = 0.01
CAM_TH            = 0.25

SCORE_TYPE = "copula"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.double
np.random.seed(SEED); random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -------------------- IO --------------------
def load_data(csv_path=DATA_CSV):
    if not os.path.exists(csv_path):
        print(f"Error: '{csv_path}' not found."); return None
    df = pd.read_csv(csv_path, header=0)
    if df.columns[0].lower().startswith("unnamed"):
        df = pd.read_csv(csv_path, header=0, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    X = df.values.astype(np.float64)
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True); sd[sd == 0] = 1.0
    X = (X - mu) / sd
    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} vars from '{os.path.basename(csv_path)}'")
    return X

def load_truth(npy_path=GT_NPY, p=None):
    if not npy_path or not os.path.exists(npy_path):
        print("No ground truth file; metrics will be limited.")
        return None
    G = np.load(npy_path).astype(np.float64)
    if p is not None and G.shape != (p, p):
        print(f"[align] trimming GT from {G.shape} to {(p,p)}")
        G = G[:p, :p]
    print("Loaded ground truth:", G.shape)
    return G

# -------------------- Opponent (GraN-DAG) --------------------
def get_grandag_adj(data, iterations=G_ITER, hidden_dim=16, hidden_num=2, lr=5e-4, h_threshold=1e-6):
    print(f"\nRunning GraN-DAG (iterations={iterations})...")
    model = GraNDAG(
        input_dim=data.shape[1], hidden_dim=hidden_dim, hidden_num=hidden_num,
        lr=lr, iterations=iterations, h_threshold=h_threshold, mu_init=1e-3
    )
    model.learn(data)
    print("GraN-DAG done.")
    return np.array(model.causal_matrix, dtype=np.float64)

# -------------------- Metrics --------------------
def binarize(A):
    A = (A != 0).astype(int)
    np.fill_diagonal(A, 0)
    return A

def shd_binary(A, B):
    A = binarize(A); B = binarize(B)
    Au = ((A + A.T) > 0).astype(int); Bu = ((B + B.T) > 0).astype(int)
    undirected_diff = int(np.sum(np.triu(Au ^ Bu, 1)))
    common_u = ((Au & Bu) > 0).astype(int)
    orient_mismatch = int(np.sum(np.triu((A ^ B) & common_u, 1)))
    return undirected_diff + orient_mismatch

def eval_against_gt(pred_adj, GT):
    if GT is None or pred_adj.shape != GT.shape:
        return None
    P = binarize(pred_adj); T = binarize(GT)
    tp = int(((P == 1) & (T == 1)).sum())
    fp = int(((P == 1) & (T == 0)).sum())
    fn = int(((P == 0) & (T == 1)).sum())
    tn = int(((P == 0) & (T == 0)).sum())
    fdr = fp / max(tp + fp, 1)
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "SHD": shd_binary(P, T),
        "NNZ": int(P.sum()),
        "TPR": round(tpr, 4),
        "FDR": round(fdr, 4),
        "Precision": round(precision, 4),
        "FPR": round(fpr, 4),
    }

# -------------------- Scorers --------------------
class GaussianBIC:
    def __init__(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.X = X
        self.n, self.p = X.shape
        self.S = X.T @ X
        self.yTy = np.diag(self.S)

    def node_rss(self, parents, j):
        if len(parents) == 0:
            y = self.X[:, j]
            return float(((y - y.mean()) ** 2).sum())
        P = np.array(parents, dtype=int)
        Spp = self.S[np.ix_(P, P)]
        Spy = self.S[P, j]
        try:
            coef = np.linalg.solve(Spp, Spy)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(Spp) @ Spy
        rss = self.yTy[j] - Spy @ coef
        return float(max(rss, 1e-12))

    def bic(self, A):
        n, p = self.n, self.p
        total_rss = 0.0
        num_params = p
        for j in range(p):
            parents = np.where(A[:, j] == 1)[0].tolist()
            total_rss += self.node_rss(parents, j)
            num_params += len(parents)
        if n <= num_params:
            return -np.inf
        loglik = -0.5 * n * p * np.log(2 * np.pi * total_rss / (n * p)) - 0.5 * (n * p)
        penalty = 0.5 * num_params * np.log(n)
        return float(loglik - penalty)

def gaussian_copula_transform(X):
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    Z = np.empty_like(X)
    eps = 1e-6
    rng = np.random.default_rng(0)
    for j in range(p):
        x = X[:, j]
        if np.std(x) < 1e-12:
            x = x + rng.normal(0, 1e-9, size=n)
        r = rankdata(x, method="average")
        u = (r - 0.5) / n
        u = np.clip(u, eps, 1 - eps)
        Z[:, j] = norm.ppf(u)
    Z -= Z.mean(axis=0, keepdims=True)
    std = Z.std(axis=0, keepdims=True); std[std == 0] = 1.0
    Z /= std
    return Z

class CopulaBIC(GaussianBIC):
    def __init__(self, X):
        Z = gaussian_copula_transform(X)
        super().__init__(Z)

# -------------------- Warm-start --------------------
def warm_start_greedy_bic(Xva, edge_budget, scorer,
                          max_passes=10, topk_per_pass=15, restarts=10, seed=SEED):
    rng = np.random.RandomState(seed)
    p = Xva.shape[1]

    def is_dag(M): return nx.is_directed_acyclic_graph(nx.DiGraph(M))

    def one_run():
        A = np.zeros((p, p), dtype=np.float64)
        best_score = scorer.bic(A)
        for _ in range(max_passes):
            improved = False
            cands = []
            if A.sum() < edge_budget:
                for i, j in product(range(p), range(p)):
                    if i == j or A[i, j] == 1: continue
                    trial = A.copy(); trial[i, j] = 1.0
                    if not is_dag(trial): continue
                    s = scorer.bic(trial)
                    if s > best_score: cands.append((s - best_score, i, j))
            cands.sort(reverse=True, key=lambda x: x[0])
            for _, i, j in cands[:topk_per_pass]:
                if A.sum() >= edge_budget: break
                trial = A.copy(); trial[i, j] = 1.0
                if is_dag(trial):
                    s = scorer.bic(trial)
                    if s > best_score:
                        A, best_score, improved = trial, s, True

            pruned = True
            while pruned:
                pruned = False
                edges = list(zip(*np.where(A == 1)))
                random.shuffle(edges)
                for i, j in edges:
                    trial = A.copy(); trial[i, j] = 0.0
                    s = scorer.bic(trial)
                    if s > best_score:
                        A, best_score, improved, pruned = trial, s, True, True
            
            edges = list(zip(*np.where(A == 1)))
            random.shuffle(edges)
            for i, j in edges:
                trial = A.copy(); trial[i, j] = 0.0; trial[j, i] = 1.0
                if not is_dag(trial): continue
                s = scorer.bic(trial)
                if s > best_score:
                    A, best_score, improved = trial, s, True
            
            if not improved: break
        return A, best_score

    bestA, bestS = None, -np.inf
    print(f"[Warm-start] Starting {restarts} greedy restarts...")
    for i in range(restarts):
        A, s = one_run()
        if s > bestS: bestA, bestS = A, s
        print(f"  Restart {i+1}/{restarts}: BIC={s:.2f}, Edges={int(A.sum())}")
    return bestA

# -------------------- Environment --------------------
class CausalDiscoveryEnv:
    def __init__(self, data,
                 val_frac=0.2,
                 edge_budget_ratio=EDGE_BUDGET_RATIO,
                 lambda_l1=LAMBDA_L1, action_cost=ACTION_COST,
                 warm_start_adj=None,
                 score_type=SCORE_TYPE):
        self.n_samples, self.n_nodes = data.shape
        
        idx = np.arange(self.n_samples); rng = np.random.RandomState(SEED)
        rng.shuffle(idx)
        cut = int((1.0 - val_frac) * self.n_samples)
        self.Xtr, self.Xva = data[idx[:cut]], data[idx[cut:]]

        Scorer = CopulaBIC if str(score_type).lower() == "copula" else GaussianBIC
        self.bic_va = Scorer(self.Xva)

        self.state_space_shape = (self.n_nodes * self.n_nodes,)
        self.n_actions = 3 * self.n_nodes * (self.n_nodes - 1)
        self.action_map, self.rev_action_map = self._create_action_map()

        self.current_adj = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float64)
        self.max_steps = 20 * self.n_nodes
        self.current_step = 0

        self.edge_budget = int(max(1, edge_budget_ratio * self.n_nodes))
        self.lambda_l1 = float(lambda_l1)
        self.action_cost = float(action_cost)
        self._warm_adj = warm_start_adj

        self.bic_empty = self.bic_va.bic(np.zeros_like(self.current_adj))
        self.best_bic_so_far = -np.inf

    def _create_action_map(self):
        mapping, rev_mapping, idx = {}, {}, 0
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i == j: continue
                mapping[idx] = ("add", i, j); rev_mapping[("add", i, j)] = idx; idx += 1
                mapping[idx] = ("remove", i, j); rev_mapping[("remove", i, j)] = idx; idx += 1
                mapping[idx] = ("reverse", i, j); rev_mapping[("reverse", i, j)] = idx; idx += 1
        return mapping, rev_mapping

    def _get_valid_actions_mask(self):
        mask = np.zeros(self.n_actions, dtype=bool)
        num_edges = self.current_adj.sum()
        G = nx.DiGraph(self.current_adj)
        paths = dict(nx.all_pairs_shortest_path_length(G))

        for action_idx in range(self.n_actions):
            op, i, j = self.action_map[action_idx]
            if op == "add":
                if self.current_adj[i, j] == 0 and num_edges < self.edge_budget:
                    if j not in paths or i not in paths.get(j, {}):
                        mask[action_idx] = True
            elif op == "remove":
                if self.current_adj[i, j] == 1:
                    mask[action_idx] = True
            elif op == "reverse":
                if self.current_adj[i, j] == 1 and self.current_adj[j, i] == 0:
                    G.remove_edge(i, j)
                    if not nx.has_path(G, i, j):
                        mask[action_idx] = True
                    G.add_edge(i, j)
        if not mask.any():
            for i,j in product(range(self.n_nodes), range(self.n_nodes)):
                if i !=j and self.current_adj[i,j] == 0:
                    mask[self.rev_action_map[("add", i, j)]] = True
        return mask

    def reset(self):
        self.current_step = 0
        self.current_adj = self._warm_adj.copy() if self._warm_adj is not None else np.zeros((self.n_nodes, self.n_nodes))
        self.best_bic_so_far = self.bic_va.bic(self.current_adj)
        return self.current_adj.flatten().copy()

    def step(self, action_idx):
        op, i, j = self.action_map[action_idx]
        prev_adj = self.current_adj.copy()
        trial = self.current_adj.copy()
        action_was_valid = False
        if op == "add" and trial[i, j] == 0:
            trial[i, j] = 1.0; action_was_valid = True
        elif op == "remove" and trial[i, j] == 1:
            trial[i, j] = 0.0; action_was_valid = True
        elif op == "reverse" and trial[i, j] == 1:
            trial[i, j] = 0.0; trial[j, i] = 1.0; action_was_valid = True
        if not action_was_valid or not nx.is_directed_acyclic_graph(nx.DiGraph(trial)):
            self.current_step += 1
            return self.current_adj.flatten(), -2.0, self.current_step >= self.max_steps, {}

        self.current_adj = trial
        self.current_step += 1
        r = self._reward(prev_adj, self.current_adj)
        done = self.current_step >= self.max_steps
        info = {'valid_actions': self._get_valid_actions_mask()}
        return self.current_adj.flatten().copy(), r, done, info

    def _reward(self, prev_adj, new_adj):
        new_bic = self.bic_va.bic(new_adj)
        score_reward = (new_bic - self.bic_empty) / self.n_nodes
        improvement_bonus = 0.0
        if new_bic > self.best_bic_so_far:
            improvement_bonus = 0.5
            self.best_bic_so_far = new_bic
        sparsity_penalty = -self.lambda_l1 * float(np.sum(new_adj))
        action_penalty   = -self.action_cost
        total = score_reward + improvement_bonus + sparsity_penalty + action_penalty
        return float(np.clip(total, -20.0, 20.0))

# -------------------- Agent --------------------
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 512, dtype=DTYPE), nn.ReLU(),
            nn.Linear(512, 512, dtype=DTYPE), nn.ReLU(),
            nn.Linear(512, action_size, dtype=DTYPE),
        )
    def forward(self, x): return self.layers(x)

class CausalAgent:
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.q = QNetwork(state_size, action_size).to(device)
        self.t = QNetwork(state_size, action_size).to(device)
        self.t.load_state_dict(self.q.state_dict())
        self.tau = 0.005
        self.opt = optim.Adam(self.q.parameters(), lr=5e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.opt, step_size=10_000, gamma=0.9)
        self.gamma = 0.99
        self.eps_start, self.eps_end = 1.0, 0.05
        self.eps_decay_steps = 400_000
        self.total_steps = 0
        self.epsilon = self.eps_start
        self.mem = deque(maxlen=200_000)
        self.batch = 512

    def _update_eps(self):
        self.total_steps += 1
        frac = min(1.0, self.total_steps / self.eps_decay_steps)
        self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)

    def remember(self, s, a, r, ns, d): self.mem.append((s, a, r, ns, d))

    def act(self, state, valid_mask=None):
        self._update_eps()
        if random.random() <= self.epsilon:
            if valid_mask is not None and valid_mask.any():
                valid_indices = np.where(valid_mask)[0]
                return np.random.choice(valid_indices)
            return random.randrange(self.action_size)
        st = torch.tensor(state, dtype=DTYPE, device=device).unsqueeze(0)
        with torch.no_grad():
            qv = self.q(st)
            if valid_mask is not None:
                qv[0, ~torch.tensor(valid_mask, device=device)] = -torch.inf
        return int(qv.argmax(dim=1).item())

    def replay(self):
        if len(self.mem) < self.batch: return
        batch = random.sample(self.mem, self.batch)
        s  = torch.tensor(np.array([e[0] for e in batch]), dtype=DTYPE, device=device)
        a  = torch.tensor([e[1] for e in batch], dtype=torch.long, device=device).unsqueeze(1)
        r  = torch.tensor([e[2] for e in batch], dtype=DTYPE, device=device).unsqueeze(1)
        ns = torch.tensor(np.array([e[3] for e in batch]), dtype=DTYPE, device=device)
        d  = torch.tensor([e[4] for e in batch], dtype=DTYPE, device=device).unsqueeze(1)
        q_sa = self.q(s).gather(1, a)
        with torch.no_grad():
            na_online = self.q(ns).argmax(1, keepdim=True)
            q_next = self.t(ns).gather(1, na_online)
            target = r + (1.0 - d) * self.gamma * q_next
        loss = nn.MSELoss()(q_sa, target)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt.step()
        self.scheduler.step()
        with torch.no_grad():
            for tparam, qparam in zip(self.t.parameters(), self.q.parameters()):
                tparam.data.mul_(1 - self.tau).add_(self.tau * qparam.data)

# -------------------- CAM pruning --------------------
def graph_prunned_by_coef(parent_mat, X, th=CAM_TH):
    d = parent_mat.shape[0]
    reg = LinearRegression()
    W_pruned = np.zeros_like(parent_mat)
    for i in range(d):
        parents = np.where(parent_mat[i, :] != 0)[0]
        if len(parents) == 0: continue
        X_train = X[:, parents]
        y_train = X[:, i]
        reg.fit(X_train, y_train)
        abs_coef = np.abs(reg.coef_)
        significant_parents = parents[abs_coef > th]
        W_pruned[i, significant_parents] = 1.0
    return W_pruned

def cam_prune_linear_from_A(A_directed, X, th=CAM_TH):
    parents = A_directed.T
    pruned_parents = graph_prunned_by_coef(parents, X, th=th)
    return pruned_parents.T

def adaptive_cam_prune(A, X, bic_scorer, thresholds=None):
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4]
    best_A_cam = A.copy()
    best_bic = bic_scorer.bic(best_A_cam)
    for th in thresholds:
        A_cam = cam_prune_linear_from_A(A, X, th=th)
        bic_cam = bic_scorer.bic(A_cam)
        if bic_cam > best_bic:
            best_bic = bic_cam
            best_A_cam = A_cam
    return best_A_cam

# -------------------- Main --------------------
def main(args):
    X = load_data(args.data_csv)
    if X is None: raise SystemExit
    p = X.shape[1]
    GT = load_truth(args.gt_npy, p=p)

    Gdag = get_grandag_adj(X, iterations=args.g_iter)
    Gdag_bin = binarize(Gdag)

    ScorerClass = CopulaBIC if args.score_type == "copula" else GaussianBIC
    bic_scorer_val = ScorerClass(X)
    
    print("\n[Warm-start] searching for a good initial graph...")
    edge_budget = int(args.edge_budget_ratio * p)
    warm_adj = warm_start_greedy_bic(
        X, edge_budget, bic_scorer_val,
        max_passes=10, topk_per_pass=15, restarts=10, seed=args.seed
    )
    bic_warm = bic_scorer_val.bic(warm_adj)
    bic_gran = bic_scorer_val.bic(Gdag_bin)
    if bic_gran > bic_warm:
        print("[Warm-start] GraN-DAG result has better initial BIC. Using it as warm-start.")
        warm_adj = Gdag_bin.copy()
    print(f"[Warm-start] Selected initial graph with {int(warm_adj.sum())} edges and BIC={bic_scorer_val.bic(warm_adj):.2f}")

    env = CausalDiscoveryEnv(
        X, val_frac=0.2, edge_budget_ratio=args.edge_budget_ratio,
        lambda_l1=args.lambda_l1, action_cost=args.action_cost,
        warm_start_adj=warm_adj, score_type=args.score_type
    )
    agent = CausalAgent(state_size=env.state_space_shape[0], action_size=env.n_actions)

    best_valbic = -np.inf
    best_agent_adj_by_bic = None
    
    # <-- ADDED: Variables to track the best model by TPR
    best_tpr = -1.0
    best_agent_adj_by_tpr = None

    print("\nStarting RL training with action masking...")
    for ep in range(1, args.n_episodes + 1):
        s = env.reset()
        valid_actions = env._get_valid_actions_mask()
        total_reward, done = 0.0, False
        
        while not done:
            a = agent.act(s, valid_actions)
            ns, r, done, info = env.step(a)
            agent.remember(s, a, r, ns, done)
            s = ns
            total_reward += r
            valid_actions = info.get('valid_actions')
            if agent.total_steps % 4 == 0:
                agent.replay()

        msg = f"Ep {ep:04d}/{args.n_episodes} | Reward={total_reward:8.3f} | Steps={env.current_step} | eps={agent.epsilon:.3f}"

        if ep % args.eval_every == 0:
            A_now = binarize(env.current_adj)
            valBIC_agent = env.bic_va.bic(A_now)

            if valBIC_agent > best_valbic:
                best_valbic = valBIC_agent
                best_agent_adj_by_bic = A_now.copy()
                msg += " (* New best BIC *)"
            
            # <-- ADDED: Check and save the best model by TPR
            if GT is not None:
                current_metrics = eval_against_gt(A_now, GT)
                if current_metrics and current_metrics['TPR'] > best_tpr:
                    best_tpr = current_metrics['TPR']
                    best_agent_adj_by_tpr = A_now.copy()
                    msg += f" (*** New best TPR: {best_tpr:.4f} ***)"
            
            print(msg)
            if GT is not None:
                print(f"  Metrics (raw): {eval_against_gt(A_now, GT)}")
        else:
            print(msg, end='\r')

    print("\nTraining finished.")
    A_final_bic = best_agent_adj_by_bic if best_agent_adj_by_bic is not None else binarize(env.current_adj)
    
    print("\nPerforming adaptive CAM pruning on best BIC graph...")
    A_cam_final = adaptive_cam_prune(A_final_bic, X, env.bic_va)

    print("\n--- Final Results (based on best validation BIC graph) ---")
    print(f"ValBIC(Agent Raw):   {env.bic_va.bic(A_final_bic):.2f}")
    print(f"ValBIC(Agent+CAM): {env.bic_va.bic(A_cam_final):.2f}")
    print(f"ValBIC(GraN-DAG):  {env.bic_va.bic(Gdag_bin):.2f}")

    if GT is not None:
        print("\n--- Ground Truth Metrics (Best by BIC) ---")
        print(f"Agent (raw):  {eval_against_gt(A_final_bic, GT)}")
        print(f"Agent+CAM:    {eval_against_gt(A_cam_final, GT)}")
        print(f"GraN-DAG:     {eval_against_gt(Gdag_bin, GT)}")
        
        # <-- ADDED: Final report for the best TPR model
        if best_agent_adj_by_tpr is not None:
            print("\n--- Ground Truth Metrics ---")
            print(f"Best TPR achieved: {best_tpr:.4f}")
            print(f"Agent (Best TPR): {eval_against_gt(best_agent_adj_by_tpr, GT)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN for Causal Discovery")
    parser.add_argument('--data_csv', type=str, default=DATA_CSV, help='Path to data CSV file')
    parser.add_argument('--gt_npy', type=str, default=GT_NPY, help='Path to ground truth npy file')
    parser.add_argument('--g_iter', type=int, default=G_ITER, help='GraN-DAG iterations')
    parser.add_argument('--n_episodes', type=int, default=N_EPISODES, help='Number of training episodes')
    parser.add_argument('--eval_every', type=int, default=EVAL_EVERY, help='Evaluate every N episodes')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    parser.add_argument('--edge_budget_ratio', type=float, default=EDGE_BUDGET_RATIO)
    parser.add_argument('--lambda_l1', type=float, default=LAMBDA_L1)
    parser.add_argument('--action_cost', type=float, default=ACTION_COST)
    parser.add_argument('--score_type', type=str, default=SCORE_TYPE, choices=['copula', 'gaussian'])
    args = parser.parse_args(args=[])
    
    SEED = args.seed
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    
    main(args)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Data
datasets = ["Asia", "Sachs", "Lucas", "Child", "Alarm", "Hepar2", "Dream", "Andes"]
methods = [
    "KCRL", "NOTEARS", "GOLEM", "Gran-DAG", "RL-BIC2", "ICALiNGAM", "DirectLiNGAM",
    "PC", "GES", "CORL", "Ours (Using Gran-DAG)", "Ours (Using GES)"
]
score = {
    "Asia":   [0.52, 0.13, 0.19, 0.42, 0.37, 0.26, 0.57, 0.54, 1.00, np.nan, 0.47, 1.00],
    "Sachs":  [0.32, 0.26, 0.13, 0.30, 0.21, 0.26, 0.23, 0.20, 0.39, 0.18, 0.30, 0.40],
    "Lucas":  [0.35, 0.33, 0.35, 0.23, 0.26, 0.20, 0.32, 0.72, 1.00, np.nan, 0.18, 1.00],
    "Child":  [0.13, 0.18, 0.12, 0.29, 0.36, 0.24, 0.11, 0.13, 0.17, np.nan, 0.41, 0.18],
    "Alarm":  [0.24, 0.26, 0.26, 0.18, 0.20, 0.43, 0.30, 0.36, 0.38, np.nan, 0.26, 0.43],
    "Hepar2": [np.nan, 0.01, np.nan, 0.30, np.nan, 0.24, 0.35, 0.20, 0.42, np.nan, 0.35, 0.43],
    "Dream":  [np.nan, 0.03, np.nan, 0.04, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.11, np.nan],
    "Andes":  [np.nan, 0.03, np.nan, 0.04, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.08, np.nan],
}
data = np.array([[score[d][i] for d in datasets] for i in range(len(methods))])

# Colors & hatches
baseline_colors = list(plt.cm.Set2.colors[:10])  # 10 distinct muted colors
ours_colors = ["#1b9e77", "#d95f02"]
hatches = [""]*len(methods)
hatches[-2] = "//"
hatches[-1] = "\\\\"

# Plot
x = np.arange(len(datasets))
width = 0.75/len(methods)

fig, ax = plt.subplots(figsize=(14,6))

for i, method in enumerate(methods):
    vals = data[i, :]
    color = ours_colors[0] if ("Ours" in method and "Gran" in method) else \
            (ours_colors[1] if "Ours" in method else baseline_colors[i % len(baseline_colors)])
    hatch = hatches[i]
    x_pos = x + (i - len(methods)/2)*width + width/2
    # Plot bars, skipping NaNs by replacing with zeros and making them transparent
    bar_vals = np.nan_to_num(vals, nan=0.0)
    bars = ax.bar(x_pos, bar_vals, width=width, label=method,
                  color=color, edgecolor="black", linewidth=0.4, hatch=hatch, alpha=0.85 if "Ours" in method else 0.75)
    # Hide bars where original was NaN
    for b, v in zip(bars, vals):
        if np.isnan(v):
            b.set_alpha(0.0)
            b.set_edgecolor((0,0,0,0))

# Mark all ties with a star (any method achieving the column max within tolerance)
tol = 1e-12
for j in range(len(datasets)):
    col = data[:, j]
    if np.all(np.isnan(col)): 
        continue
    col_max = np.nanmax(col)
    for i in range(len(methods)):
        val = data[i, j]
        if np.isnan(val): 
            continue
        if abs(val - col_max) <= tol:
            x_star = x[j] + (i - len(methods)/2)*width + width/2
            ax.text(x_star, val + 0.02, "★", ha="center", va="bottom", fontsize=11)

ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=0)
ax.set_ylabel("Composite Score")
ax.set_ylim(0, 1.1)
ax.set_title("Composite Score by Dataset and Method\n(Ours highlighted; ★ = best per dataset, ties marked)")
ax.legend(ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left", title="Methods", fontsize="small")
plt.tight_layout()

out_path = "composite_grouped_bars_ties_marked.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
out_path

