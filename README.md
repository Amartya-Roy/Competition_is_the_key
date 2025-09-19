Causal discovery remains a central challenge in machine learning, yet existing methods face a
fundamental gap: algorithms like GES and GraN-DAG achieve strong empirical performance but lack
finite-sample guarantees, while theoretically principled approaches fail to scale. We close this gap by
introducing a game-theoretic reinforcement learning framework for causal discovery, where a
DDQN agent directly competes against a strong baseline (GES or GraN–DAG), always warm-starting
from the opponent’s solution. This design yields three provable guarantees: the learned graph is
never worse than the opponent, warm-starting strictly accelerates convergence, and most importantly
with high probability the algorithm selects the true best candidate graph

![Comparison](https://github.com/Amartya-Roy/Competition_is_the_key/blob/main/competition_is_the_key.png)



# 0) Install deps (example)
pip install numpy pandas torch networkx scikit-learn scipy pyyaml castle-algorithms matplotlib

# 1) Simple mode (GES path)
python main.py --config config.yml --mode simple

# 2) Advanced mode (GraN-DAG-based)
python main.py --config config.yml --mode advanced

# 3) Plot the grouped-bar figure with ties marked
python plot_scores.py
