#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, random, warnings, argparse, yaml
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

warnings.filterwarnings("ignore", category=UserWarning)

# Optional opponents (installed via 'castle')
try:
    from castle.algorithms import GES
except Exception:
    GES = None

try:
    from castle.algorithms import GraNDAG
except Exception:
    GraNDAG = None

# ------------- Utils & IO -------------
def set_seed(seed):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, header=0)
    if df.columns[0].lower().startswith("unnamed"):
        df = pd.read_csv(csv_path, header=0, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    X = df.values.astype(np.float64)
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True); sd[sd == 0] = 1.0
    X = (X - mu) / sd
    return X

def load_truth(npy_path, p=None):
    if npy_path is None or (not os.path.exists(npy_path)):
        return None
    G = np.load(npy_path).astype(np.float64)
    if p is not None and G.shape != (p, p):
        G = G[:p, :p]
    return G

def binarize(A):
    A = (A != 0).astype(int); np.fill_diagonal(A, 0); return A

# ------------- Metrics -------------
def shd_binary(A, B):
    A = binarize(A); B = binarize(B)
    Au = ((A + A.T) > 0).astype(int); Bu = ((B + B.T) > 0).astype(int)
    undirected_diff = int(np.sum(np.triu(Au ^ Bu, 1)))
    common_u = ((Au & Bu) > 0).astype(int)
    orient_mismatch = int(np.sum(np.triu((A ^ B) & common_u, 1)))
    return undirected_diff + orient_mismatch

def eval_against_gt(pred_adj, GT):
    if GT is None or pred_adj.shape != GT.shape: return None
    P = binarize(pred_adj); T = binarize(GT)
    tp = int(((P == 1) & (T == 1)).sum())
    fp = int(((P == 1) & (T == 0)).sum())
    fn = int(((P == 0) & (T == 1)).sum())
    tn = int(((P == 0) & (T == 0)).sum())
    fdr = fp / max(tp + fp, 1); tpr = tp / max(tp + fn, 1); fpr = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)
    return {"SHD": shd_binary(P, T), "NNZ": int(P.sum()),
            "TPR": round(tpr, 4), "FDR": round(fdr, 4),
            "Precision": round(precision, 4), "FPR": round(fpr, 4)}

# ------------- Scorers -------------
class GaussianBIC:
    def __init__(self, X):
        X = np.asarray(X, dtype=np.float64); self.X = X
        self.n, self.p = X.shape
        self.S = X.T @ X; self.yTy = np.diag(self.S)

    def node_rss(self, parents, j):
        if len(parents) == 0:
            y = self.X[:, j]; return float(((y - y.mean())**2).sum())
        P = np.array(parents, dtype=int); Spp = self.S[np.ix_(P, P)]; Spy = self.S[P, j]
        try: coef = np.linalg.solve(Spp, Spy)
        except np.linalg.LinAlgError: coef = np.linalg.pinv(Spp) @ Spy
        rss = self.yTy[j] - Spy @ coef
        return float(max(rss, 1e-12))

    def bic(self, A):
        n, p = self.n, self.p
        total_rss, num_params = 0.0, p
        for j in range(p):
            parents = np.where(A[:, j] == 1)[0].tolist()
            total_rss += self.node_rss(parents, j)
            num_params += len(parents)
        if n <= num_params: return -np.inf
        loglik = -0.5 * n * p * np.log(2 * np.pi * total_rss / (n * p)) - 0.5 * (n * p)
        penalty = 0.5 * num_params * np.log(n)
        return float(loglik - penalty)

def gaussian_copula_transform(X):
    X = np.asarray(X, dtype=np.float64); n, p = X.shape
    Z = np.empty_like(X); eps = 1e-6; rng = np.random.default_rng(0)
    for j in range(p):
        x = X[:, j]
        if np.std(x) < 1e-12: x = x + rng.normal(0, 1e-9, size=n)
        r = rankdata(x, method="average"); u = (r - 0.5) / n
        u = np.clip(u, eps, 1 - eps); Z[:, j] = norm.ppf(u)
    Z -= Z.mean(axis=0, keepdims=True)
    std = Z.std(axis=0, keepdims=True); std[std == 0] = 1.0
    Z /= std; return Z

class CopulaBIC(GaussianBIC):
    def __init__(self, X): super().__init__(gaussian_copula_transform(X))

# ------------- Warm-start greedy (BIC) -------------
def warm_start_greedy_bic(X, edge_budget, scorer, max_passes=10, topk_per_pass=15, restarts=10, seed=42):
    rng = np.random.RandomState(seed); p = X.shape[1]
    def is_dag(M): return nx.is_directed_acyclic_graph(nx.DiGraph(M))
    def one_run():
        A = np.zeros((p, p), dtype=np.float64); best_score = scorer.bic(A)
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
                    if s > best_score: A, best_score, improved = trial, s, True
            pruned = True
            while pruned:
                pruned = False
                edges = list(zip(*np.where(A == 1))); random.shuffle(edges)
                for i, j in edges:
                    trial = A.copy(); trial[i, j] = 0.0
                    s = scorer.bic(trial)
                    if s > best_score: A, best_score, improved, pruned = trial, s, True, True
            edges = list(zip(*np.where(A == 1))); random.shuffle(edges)
            for i, j in edges:
                trial = A.copy(); trial[i, j] = 0.0; trial[j, i] = 1.0
                if not is_dag(trial): continue
                s = scorer.bic(trial)
                if s > best_score: A, best_score, improved = trial, s, True
            if not improved: break
        return A, best_score
    bestA, bestS = None, -np.inf
    for _ in range(restarts):
        A, s = one_run()
        if s > bestS: bestA, bestS = A, s
    return bestA

# ------------- Environment -------------
class CausalDiscoveryEnv:
    def __init__(self, X, val_frac, edge_budget_ratio, lambda_l1, action_cost, warm_start_adj, score_type, max_steps_mult, seed):
        self.n_samples, self.n_nodes = X.shape
        idx = np.arange(self.n_samples); rng = np.random.RandomState(seed); rng.shuffle(idx)
        cut = int((1.0 - val_frac) * self.n_samples); self.Xtr, self.Xva = X[idx[:cut]], X[idx[cut:]]
        Scorer = CopulaBIC if score_type.lower() == "copula" else GaussianBIC
        self.bic_va = Scorer(self.Xva)
        self.state_space_shape = (self.n_nodes * self.n_nodes,)
        self.n_actions = 3 * self.n_nodes * (self.n_nodes - 1)
        self.action_map, self.rev_action_map = self._create_action_map()
        self.current_adj = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float64)
        self.max_steps = max_steps_mult * self.n_nodes
        self.current_step = 0
        self.edge_budget = int(max(1, edge_budget_ratio * self.n_nodes))
        self.lambda_l1 = float(lambda_l1); self.action_cost = float(action_cost)
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
        mask = np.zeros(self.n_actions, dtype=bool); num_edges = self.current_adj.sum()
        G = nx.DiGraph(self.current_adj); paths = dict(nx.all_pairs_shortest_path_length(G))
        for action_idx in range(self.n_actions):
            op, i, j = self.action_map[action_idx]
            if op == "add":
                if self.current_adj[i, j] == 0 and num_edges < self.edge_budget:
                    if j not in paths or i not in paths.get(j, {}): mask[action_idx] = True
            elif op == "remove":
                if self.current_adj[i, j] == 1: mask[action_idx] = True
            elif op == "reverse":
                if self.current_adj[i, j] == 1 and self.current_adj[j, i] == 0:
                    G.remove_edge(i, j)
                    if not nx.has_path(G, i, j): mask[action_idx] = True
                    G.add_edge(i, j)
        if not mask.any():
            for i,j in product(range(self.n_nodes), range(self.n_nodes)):
                if i != j and self.current_adj[i,j] == 0:
                    mask[self.rev_action_map[("add", i, j)]] = True
        return mask

    def reset(self):
        self.current_step = 0
        self.current_adj = self._warm_adj.copy() if self._warm_adj is not None else np.zeros((self.n_nodes, self.n_nodes))
        self.best_bic_so_far = self.bic_va.bic(self.current_adj)
        return self.current_adj.flatten().copy()

    def step(self, action_idx):
        op, i, j = self.action_map[action_idx]
        prev_adj = self.current_adj.copy(); trial = self.current_adj.copy(); valid = False
        if op == "add" and trial[i, j] == 0: trial[i, j] = 1.0; valid = True
        elif op == "remove" and trial[i, j] == 1: trial[i, j] = 0.0; valid = True
        elif op == "reverse" and trial[i, j] == 1: trial[i, j] = 0.0; trial[j, i] = 1.0; valid = True
        if not valid or not nx.is_directed_acyclic_graph(nx.DiGraph(trial)):
            self.current_step += 1
            return self.current_adj.flatten(), -2.0, self.current_step >= self.max_steps, {}
        self.current_adj = trial; self.current_step += 1
        r = self._reward(prev_adj, self.current_adj)
        done = self.current_step >= self.max_steps
        return self.current_adj.flatten().copy(), r, done, {'valid_actions': self._get_valid_actions_mask()}

    def _reward(self, prev_adj, new_adj):
        new_bic = self.bic_va.bic(new_adj)
        score_reward = (new_bic - self.bic_empty) / self.n_nodes
        improvement_bonus = 0.5 if new_bic > self.best_bic_so_far else 0.0
        if improvement_bonus > 0: self.best_bic_so_far = new_bic
        sparsity_penalty = -self.lambda_l1 * float(np.sum(new_adj))
        action_penalty   = -self.action_cost
        total = score_reward + improvement_bonus + sparsity_penalty + action_penalty
        return float(np.clip(total, -20.0, 20.0))

# ------------- Agent (DDQN) -------------
DTYPE = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, hidden, dtype=DTYPE), nn.ReLU(),
            nn.Linear(hidden, hidden, dtype=DTYPE), nn.ReLU(),
            nn.Linear(hidden, action_size, dtype=DTYPE),
        )
    def forward(self, x): return self.layers(x)

class CausalAgent:
    def __init__(self, state_size, action_size, cfg):
        self.action_size = action_size
        h, bs = cfg["hidden"], cfg["batch_size"]
        self.q = QNetwork(state_size, action_size, h).to(device)
        self.t = QNetwork(state_size, action_size, h).to(device)
        self.t.load_state_dict(self.q.state_dict())
        self.tau = cfg["tau"]
        self.opt = optim.Adam(self.q.parameters(), lr=cfg["lr"])
        self.scheduler = optim.lr_scheduler.StepLR(self.opt, step_size=cfg["lr_step_size"], gamma=cfg["lr_gamma"])
        self.gamma = cfg["gamma"]
        self.eps_start, self.eps_end, self.eps_decay_steps = cfg["eps_start"], cfg["eps_end"], cfg["eps_decay_steps"]
        self.total_steps = 0; self.epsilon = self.eps_start
        self.mem = deque(maxlen=cfg["replay_size"]); self.batch = bs

    def _update_eps(self):
        self.total_steps += 1
        frac = min(1.0, self.total_steps / self.eps_decay_steps)
        self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)

    def remember(self, s, a, r, ns, d): self.mem.append((s, a, r, ns, d))

    def act(self, state, valid_mask=None):
        self._update_eps()
        if random.random() <= self.epsilon:
            if valid_mask is not None and valid_mask.any():
                return np.random.choice(np.where(valid_mask)[0])
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
        self.opt.step(); self.scheduler.step()
        with torch.no_grad():
            for tparam, qparam in zip(self.t.parameters(), self.q.parameters()):
                tparam.data.mul_(1 - self.tau).add_(self.tau * qparam.data)

# ------------- CAM pruning -------------
def graph_prunned_by_coef(parent_mat, X, th):
    d = parent_mat.shape[0]; reg = LinearRegression(); Wp = np.zeros_like(parent_mat)
    for i in range(d):
        parents = np.where(parent_mat[i, :] != 0)[0]
        if len(parents) == 0: continue
        reg.fit(X[:, parents], X[:, i])
        abs_coef = np.abs(reg.coef_); keep = parents[abs_coef > th]
        Wp[i, keep] = 1.0
    return Wp

def cam_prune_linear_from_A(A_directed, X, th): return graph_prunned_by_coef(A_directed.T, X, th).T

def adaptive_cam_prune(A, X, bic_scorer, thresholds):
    best_A, best_bic = A.copy(), bic_scorer.bic(A)
    for th in thresholds:
        Ac = cam_prune_linear_from_A(A, X, th=th)
        b = bic_scorer.bic(Ac)
        if b > best_bic: best_A, best_bic = Ac, b
    return best_A

# ------------- Opponents -------------
def get_ges_adj(data):
    if GES is None: raise RuntimeError("castle.algorithms.GES is not available.")
    model = GES(criterion='bic', method='scatter'); model.learn(data)
    return np.array(model.causal_matrix, dtype=np.float64)

def get_grandag_adj(data, iterations):
    if GraNDAG is None: raise RuntimeError("castle.algorithms.GraNDAG is not available.")
    model = GraNDAG(input_dim=data.shape[1], hidden_dim=16, hidden_num=2, lr=5e-4,
                    iterations=iterations, h_threshold=1e-6, mu_init=1e-3)
    model.learn(data); return np.array(model.causal_matrix, dtype=np.float64)

# ------------- Main -------------
def run(mode, cfg):
    set_seed(cfg["seed"])
    outdir = cfg["outputs"]["save_dir"]; os.makedirs(outdir, exist_ok=True)

    if mode == "simple":
        csv = cfg["data"]["simple_csv"]; gt  = cfg["data"]["simple_gt"]
    elif mode == "advanced":
        csv = cfg["data"]["advanced_csv"]; gt = cfg["data"]["advanced_gt"]
    else:
        raise ValueError("mode must be 'simple' or 'advanced'.")

    X = load_data(csv); p = X.shape[1]
    GT = load_truth(gt, p=p)

    ScorerClass = CopulaBIC if cfg["score_type"].lower() == "copula" else GaussianBIC
    bic_val = ScorerClass(X)

    # Opponent & warm-start selection
    if mode == "simple":
        if GES is None:
            raise RuntimeError("GES not available. Install 'castle' for simple mode.")
        G_init = get_ges_adj(X)
    else:
        if GraNDAG is None:
            raise RuntimeError("GraNDAG not available. Install 'castle' for advanced mode.")
        G_init = get_grandag_adj(X, iterations=cfg["grndag_iterations"])

    G_bin = binarize(G_init)

    edge_budget = int(cfg["edge_budget_ratio"] * p)
    W = warm_start_greedy_bic(
        X, edge_budget, bic_val,
        max_passes=cfg["warm_start"]["max_passes"],
        topk_per_pass=cfg["warm_start"]["topk_per_pass"],
        restarts=cfg["warm_start"]["restarts"],
        seed=cfg["seed"]
    )
    if bic_val.bic(G_bin) > bic_val.bic(W):
        W = G_bin.copy()

    env = CausalDiscoveryEnv(
        X=X, val_frac=0.2,
        edge_budget_ratio=cfg["edge_budget_ratio"],
        lambda_l1=cfg["lambda_l1"], action_cost=cfg["action_cost"],
        warm_start_adj=W, score_type=cfg["score_type"],
        max_steps_mult=cfg["max_steps_multiplier"], seed=cfg["seed"]
    )
    agent = CausalAgent(state_size=env.state_space_shape[0], action_size=env.n_actions, cfg=cfg["qnet"])

    best_valbic, best_adj = -np.inf, None
    best_tpr, best_adj_tpr = -1.0, None

    print(f"==> Training [{mode}] with {cfg['episodes']} episodes")
    for ep in range(1, cfg["episodes"] + 1):
        s = env.reset(); mask = env._get_valid_actions_mask()
        total_reward, done = 0.0, False
        while not done:
            a = agent.act(s, mask); ns, r, done, info = env.step(a)
            agent.remember(s, a, r, ns, done); s = ns; total_reward += r
            mask = info.get('valid_actions')
            if agent.total_steps % 4 == 0: agent.replay()

        msg = f"Ep {ep:04d}/{cfg['episodes']} | Reward={total_reward:8.3f} | Steps={env.current_step} | eps={agent.epsilon:.3f}"
        if ep % cfg["eval_every"] == 0:
            A_now = binarize(env.current_adj)
            valBIC = env.bic_va.bic(A_now)
            if valBIC > best_valbic: best_valbic, best_adj = valBIC, A_now.copy(); msg += " (* New best BIC *)"
            print(msg)
            if GT is not None:
                met = eval_against_gt(A_now, GT)
                if met and met['TPR'] > best_tpr: best_tpr, best_adj_tpr = met['TPR'], A_now.copy()
                print(f"  Metrics (raw): {met}")
        else:
            print(msg, end='\r')

    A_final = best_adj if best_adj is not None else binarize(env.current_adj)
    print("\n==> Adaptive CAM pruning on best-BIC graph...")
    A_cam = adaptive_cam_prune(A_final, X, env.bic_va, thresholds=cfg["cam_thresholds"])

    print("\n--- Final (Validation BIC) ---")
    print(f"ValBIC(Agent Raw): {env.bic_va.bic(A_final):.2f}")
    print(f"ValBIC(Agent+CAM): {env.bic_va.bic(A_cam):.2f}")
    print(f"ValBIC(Init Opp.): {env.bic_va.bic(G_bin):.2f}")

    if GT is not None:
        print("\n--- Ground Truth Metrics (Best BIC) ---")
        print(f"Agent (raw):  {eval_against_gt(A_final, GT)}")
        print(f"Agent+CAM:    {eval_against_gt(A_cam, GT)}")
        print(f"Opponent:     {eval_against_gt(G_bin, GT)}")
        if best_adj_tpr is not None:
            print("\n--- Best TPR snapshot ---")
            print(f"Best TPR: {best_tpr:.4f} | Metrics: {eval_against_gt(best_adj_tpr, GT)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDQN-CD trainer (simple|advanced)")
    parser.add_argument("--config", type=str, default="config.yml")
    parser.add_argument("--mode",   type=str, choices=["simple", "advanced"], default="simple")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    run(args.mode, cfg)
