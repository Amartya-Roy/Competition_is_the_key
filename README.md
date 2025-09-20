Causal discovery remains a central challenge in machine learning, yet existing methods face a
fundamental gap: algorithms like GES and GraN-DAG achieve strong empirical performance but lack
finite-sample guarantees, while theoretically principled approaches fail to scale. We close this gap by
introducing a game-theoretic reinforcement learning framework for causal discovery, where a
DDQN agent directly competes against a strong baseline (GES or GraN–DAG), always warm-starting
from the opponent’s solution. This design yields three provable guarantees: the learned graph is
never worse than the opponent, warm-starting strictly accelerates convergence, and most importantly
with high probability the algorithm selects the true best candidate graph

![Comparison](https://github.com/Amartya-Roy/Competition_is_the_key/blob/main/competition_is_the_key.png)



## Table of Contents
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Command Line Arguments](#command-line-arguments)
- [Configuration](#configuration)
- [Examples](#examples)

---

## Installation

### Prerequisites

Make sure you have Python installed (preferably Python 3.7+).  
Install the required dependencies by running:

```bash
pip install numpy pandas torch networkx scikit-learn scipy pyyaml castle-algorithms matplotlib
````

---

## Project Structure

```
project_root/
├── README.md
├── main.py
├── plot_scores.py
├── verify_theorem3.py
└── Data/
    ├── alarm/
    │   ├── adj.npy
    │   ├── dag.gexf
    │   └── nodes.npy
    ├── andes/
    │   ├── dag.gexf
    │   └── nodes.npy
    ├── asia/
    │   ├── adj.npy
    │   ├── dag.gexf
    │   └── nodes.npy
    ├── child/
    │   └── Child_s1000_v1_adj.npy
    ├── dream41/
    │   ├── adj.npy
    │   ├── dag.gexf
    │   └── nodes.npy
    ├── Hepar2/
    │   └── hepar2adj.npy
    ├── lucas/
    │   ├── adj.npy
    │   └── lucas.csv
    └── sachs/
        ├── adj.npy
        ├── dag.gexf
        └── nodes.npy
```

---

## Usage

### Quick Start

Run a quick test with synthetic data:

```bash
python main.py
```

### Command Line Arguments

| Argument   | Type   | Default      | Description                                                |
| ---------- | ------ | ------------ | ---------------------------------------------------------- |
| `--config` | string | `config.yml` | Path to the configuration file                             |
| `--mode`   | string | `simple`     | Mode to run: `simple` (GES) or `advanced` (GraN-DAG-based) |

---

## Configuration

All configurable parameters can be set in the `config.yml` file.
Modify this file to customize dataset paths, model parameters, and training options.

---

## Examples

1. **Simple mode (GES path)**
   Run the model in simple mode using the GES path:

```bash
python main.py --config config.yml --mode simple
```

2. **Advanced mode (GraN-DAG-based)**
   Run the model in advanced mode using GraN-DAG:

```bash
python main.py --config config.yml --mode advanced
```

3. **Plot the grouped-bar figure with ties marked**
   Use the plotting script (if applicable) to visualize results:

```bash
python plot_scores.py
```


