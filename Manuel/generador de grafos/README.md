# IoT Synthetic Graph Generator

Reproducible synthetic graph generator for the project **"Graph-based distributed analytics for anomaly detection and data quality in IoT networks"** (Team T800).

The script produces IoT-like graphs with a known community structure (ground truth), optional spatial coordinates, and optional heavy-tailed degree distributions. Its purpose is to provide a controlled benchmark for comparing graph partitioning strategies (community-based, spatial/geohash, hash, degree-aware) *on the same base graph* before measuring downstream anomaly-detection F1.

## Features

- **Real ground-truth communities** via a Stochastic Block Model (SBM), so you can verify that a partitioner (Louvain, METIS, ...) actually recovers the planted structure.
- **Spatially coherent coordinates** (`hybrid` mode): each community occupies a zone of the plane, enabling a spatial/geohash partitioning baseline.
- **Heterogeneous degree distribution** (`powerlaw` mode): real hubs for degree-aware strategies such as DBH/HDRF.
- **Full reproducibility**: a mandatory seed, and a `manifest.json` that stores every parameter used (supports the "version freezing and random seeds" requirement of Phase 4 of the experimental protocol).
- **Sanity-check report**: connectivity, degree statistics, and modularity of the planted communities.
- **Scales to 100k nodes**: no O(n²) pair enumeration. The SBM uses `nx.stochastic_block_model(sparse=True)` and the spatial mode uses a `cKDTree` (O(n log n)).

## Requirements

- Python 3.8+
- `numpy`
- `networkx`
- `scipy`

```bash
pip install numpy networkx scipy
```

## Generation modes

| Mode        | Description                                                                 | Typical use                                    |
|-------------|-----------------------------------------------------------------------------|------------------------------------------------|
| `community` | Pure Stochastic Block Model                                                 | Community-based partitioning baseline          |
| `spatial`   | Uniform random coordinates in `[0, area]²`, edges via k-NN or fixed radius  | Spatial / geohash partitioning baseline        |
| `powerlaw`  | Barabási–Albert preferential attachment                                     | Degree-aware strategies (DBH / HDRF)           |
| `hybrid`    | SBM + coordinates clustered by community (**recommended**)                  | Comparing community vs. spatial vs. hash on the same graph |

`hybrid` is the closest to a real IoT deployment, where sensors in the same community/neighborhood tend to be physically close.

### How the SBM is parameterized

Instead of exposing `p_intra` / `p_inter` directly (which yields very different densities as `n` or the number of communities changes), the generator takes a **target average degree** and a **community ratio** and derives the probabilities:

```
p_inter = avg_degree / (n * (community_ratio / k + 1))
p_intra = community_ratio * p_inter
```

where `k` is the number of communities. This keeps density comparable across the 1k / 10k / 100k node scenarios.

Community sizes are balanced, with the remainder (`n mod k`) randomly distributed so that sizes differ slightly, which is more realistic than identical communities.

## Usage

```bash
python iot_graph_generator.py --mode <mode> --n <nodes> --seed <seed> [options]
```

### Examples: the three scales from the experimental protocol

```bash
python iot_graph_generator.py --mode hybrid --n 1000    --seed 42 --out ./data
python iot_graph_generator.py --mode hybrid --n 10000   --seed 42 --out ./data
python iot_graph_generator.py --mode hybrid --n 100000  --seed 42 --out ./data
```

### Repetitions (Phase 4: 20–30 runs)

To compare partitioning strategies while keeping everything else fixed, change only the seed:

```bash
python iot_graph_generator.py --mode hybrid --n 10000 --seed 1 --out ./data
python iot_graph_generator.py --mode hybrid --n 10000 --seed 2 --out ./data
# ...
```

### Other modes

```bash
# Pure SBM
python iot_graph_generator.py --mode community --n 10000 --seed 42 --communities 20

# Spatial graph with k-NN edges
python iot_graph_generator.py --mode spatial --n 10000 --seed 42 --k-neighbors 8

# Spatial graph with fixed-radius edges
python iot_graph_generator.py --mode spatial --n 10000 --seed 42 --radius 15

# Power-law (Barabási–Albert)
python iot_graph_generator.py --mode powerlaw --n 10000 --seed 42 --avg-degree 10
```

## Command-line arguments

### Common

| Argument      | Type  | Default  | Description                                                      |
|---------------|-------|----------|------------------------------------------------------------------|
| `--mode`      | str   | required | `community`, `spatial`, `powerlaw`, or `hybrid`                  |
| `--n`         | int   | required | Number of nodes (protocol scenarios: 1000, 10000, 100000)        |
| `--seed`      | int   | required | Random seed (mandatory for reproducibility)                      |
| `--out`       | str   | `./data` | Output directory                                                 |

### Community structure (`community`, `hybrid`)

| Argument            | Type  | Default | Description                                                                 |
|---------------------|-------|---------|-----------------------------------------------------------------------------|
| `--communities`     | int   | `10`    | Number of communities                                                       |
| `--avg-degree`      | float | `10.0`  | Target average degree (realistic for IoT: 5–20). Also used by `powerlaw`    |
| `--community-ratio` | float | `25.0`  | How many times more likely an intra-community edge is than an inter-community one. Higher = easier-to-detect communities (typical: 15–50) |

### Spatial (`spatial`, `hybrid`)

| Argument           | Type  | Default  | Description                                                       |
|--------------------|-------|----------|-------------------------------------------------------------------|
| `--area`           | float | `1000.0` | Side length of the square plane                                   |
| `--k-neighbors`    | int   | `6`      | k for k-NN edges (`spatial` mode)                                 |
| `--radius`         | float | `None`   | Fixed radius instead of k-NN (`spatial` mode)                     |
| `--cluster-spread` | float | `40.0`   | Std. deviation of node positions around each community centroid (`hybrid` mode) |

## Output

Each run creates a directory named `{mode}_n{n}_seed{seed}` inside `--out`:

```
data/
└── hybrid_n10000_seed42/
    ├── nodes.csv
    ├── edges.csv
    └── manifest.json
```

### `nodes.csv`

One row per node. Columns (sorted alphabetically) depend on the mode:

| Column      | Present in                  | Description                    |
|-------------|-----------------------------|--------------------------------|
| `id`        | all                         | Node identifier                |
| `community` | `community`, `hybrid`       | Ground-truth community label   |
| `x`, `y`    | `spatial`, `hybrid`         | Node coordinates               |

### `edges.csv`

One row per undirected edge.

| Column   | Description                                                                                         |
|----------|-----------------------------------------------------------------------------------------------------|
| `source` | Source node id                                                                                      |
| `target` | Target node id                                                                                      |
| `weight` | `1.0` in `community` / `powerlaw`; Euclidean distance between endpoints in `spatial` / `hybrid`     |

### `manifest.json`

Contains the full set of CLI parameters (`params`) and the validation report (`validation`), so any run can be reproduced exactly.

### Validation report

| Field                        | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `n_nodes`, `n_edges`         | Graph size                                                                  |
| `avg_degree`, `max_degree`   | Degree statistics                                                           |
| `n_connected_components`     | Number of connected components                                              |
| `giant_component_fraction`   | Fraction of nodes in the largest component                                  |
| `isolated_nodes`             | Nodes with degree 0                                                         |
| `n_communities`              | Number of planted communities (`community`, `hybrid`)                       |
| `modularity_ground_truth`    | Modularity of the planted partition (`community`, `hybrid`)                 |
| `generation_time_seconds`    | Generation time                                                             |

> **Note:** the ground-truth modularity is computed with `weight=None` on purpose. It measures the *topological* quality of the planted communities, independent of what `weight` means in each mode (uniform in `community`, spatial distance in `hybrid`). This makes `community` and `hybrid` comparable for the same topology.

## Loading the data

### NetworkX / pandas

```python
import pandas as pd
import networkx as nx

nodes = pd.read_csv("data/hybrid_n10000_seed42/nodes.csv")
edges = pd.read_csv("data/hybrid_n10000_seed42/edges.csv")

G = nx.from_pandas_edgelist(edges, "source", "target", edge_attr="weight")
nx.set_node_attributes(G, nodes.set_index("id").to_dict("index"))
```

### PyTorch Geometric

```python
import pandas as pd
import torch

edges = pd.read_csv("data/hybrid_n10000_seed42/edges.csv")
edge_index = torch.tensor(edges[["source", "target"]].values.T, dtype=torch.long)
# Edges are stored once (undirected); symmetrize if your model expects both directions:
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
```

## Design notes

- **Reproducibility:** all stochastic components (SBM, community sizes, coordinates, Barabási–Albert) are driven by the single `--seed`.
- **CSV instead of GraphML:** lighter and faster at 100k nodes, and trivial to load into common graph libraries.
- **Spatial k-NN edges** are symmetrized (an edge `(i, j)` exists if `j` is among the k nearest neighbors of `i` or vice versa), so the resulting average degree can be slightly above `k`.
- **`powerlaw` mode** uses `m = round(avg_degree / 2)` edges per new node, giving an average degree of roughly `2m`.
- **`hybrid` coordinates** are drawn from a Gaussian around a random centroid per community and clipped to `[0, area]`.

## Project structure

```
iot_graph_generator.py   # generator, validation, export and CLI (single file)
```