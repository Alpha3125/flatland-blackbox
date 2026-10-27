# MAPF NCO

A replication of **"Learned Representations Enhance Multi Agent Path Planning"**
(Marius Captari and Herke van Hoof, ICML 2025 Workshop on Programmatic Representations for Agent Learning).

A neural network predicts a cost for every edge of the MAPF grid. Prioritized Planning (PP) plans on those
costs, and the network is trained so that PP's plans match expert plans from EECBS. PP is not differentiable,
so gradients come from black-box differentiation (Pogančić et al., 2019): one extra PP call on perturbed costs
per scenario.

This repository started as a fork of [AI4REALNET/flatland-blackbox](https://github.com/AI4REALNET/flatland-blackbox),
which applies the same idea to Flatland railway graphs. That Flatland-specific code (rail graphs, proxy nodes,
city generators, flowtime) has been replaced by the grid MAPF setting the paper uses.

## How the code maps to the paper

| Paper | Code |
|---|---|
| §3.1 grid graph, wait self-loops, directed edges | `mapf_nco/grid.py` |
| §3.1 edge-usage vector *y* (Eq. 1) | `mapf_nco/metrics.py` → `edge_usage` |
| §3.1/3.3 PP: space-time A\*, reservation table, learned costs for *g*, unit costs for timing, true-distance heuristic recomputed under the learned weights | `mapf_nco/solvers/pp.py` |
| §3.2 MSE loss on directed usage (Eq. 4), perturbation (Eq. 5), surrogate gradient (Eq. 6) | `mapf_nco/blackbox.py` |
| §3.3 / §4 Edge model and two-layer GraphSAGE model with an edge MLP | `mapf_nco/models.py` |
| Algorithm 1, Adam (lr 5e-4), λ = 1.0, validation-delay checkpointing | `mapf_nco/train.py` |
| §4 EECBS experts (suboptimality 1.0 for 50 agents, 1.05 for 75/100) | `mapf_nco/solvers/eecbs.py`, `mapf_nco/experts.py` |
| §4 priority order: ascending true distance, otherwise one fixed random permutation | `mapf_nco/scenarios.py` → `assign_priority_order` |
| §4 random-32-32-20 map, 500 train scenarios (20 % for validation), 25 test scenarios | `configs/paper.json` |
| Table 1: average delay and optimality gap relative to EECBS | `mapf_nco/evaluate.py` |

## Installation

```bash
conda create --name mapf-nco python=3.12
conda activate mapf-nco
pip install -r requirements.txt
```

### EECBS (expert solver)

The paper's expert plans come from the official C++ [EECBS](https://github.com/Jiaoyang-Li/EECBS). It needs
CMake and Boost, so on Windows build and run it inside **WSL**.

```bash
# Ubuntu / WSL: sudo apt install build-essential cmake libboost-all-dev
# macOS:        brew install cmake boost
bash scripts/build_eecbs.sh
```

This builds `third_party/EECBS/build/eecbs`, which is found automatically. You can also pass
`--eecbs_bin /path/to/eecbs` or set `MAPF_NCO_EECBS`.

#### EECBS on Windows (via WSL)

Build EECBS inside WSL. Keep running Python normally on Windows: the Linux binary is detected and called
through `wsl` automatically.

1. Install WSL once, from an **administrator** PowerShell, then restart:
   ```powershell
   wsl --install
   ```
2. Open the Ubuntu app (or type `wsl` in PowerShell) and run:
   ```bash
   sudo apt update && sudo apt install -y build-essential cmake git libboost-all-dev
   cd /mnt/c/Users/<you>/Desktop/flatland-blackbox
   bash scripts/build_eecbs.sh
   ```
3. Back in your normal PowerShell or conda prompt:
   ```powershell
   python main.py generate
   ```
   It first runs EECBS on a 10-agent test instance and prints `EECBS OK (...) (via WSL)`. If something is wrong,
   it stops and shows EECBS's error output.

How long generation takes: at 50 agents with suboptimality 1.0, most scenarios solve in under a second. Some take
a minute or more, and about 1 in 3 hits the 120 s limit and gets replaced. The paper reports roughly 1 hour for 500
scenarios on 36 cores, so a laptop needs several hours. Start with a small set, e.g. `--set num_train=40 num_test=5`.

To build natively instead, install Visual Studio (C++), CMake and Boost through
[vcpkg](https://vcpkg.io) (`vcpkg install boost-program-options boost-tokenizer boost-heap boost-unordered`).
Then build with `-DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake`. An `eecbs.exe` under
`third_party/EECBS/build/Release/` is found automatically.

If you don't have EECBS, `--expert cbs` uses a pure-Python optimal CBS. It is only practical for about
20 agents or fewer, so use it for testing, not for the paper's 50–100-agent experiments.

## Using it: the interactive menu

```powershell
python main.py                                # paper settings (EECBS experts)
python main.py --config configs/smoke.json    # quick test: 12 agents, Python CBS, 3 epochs
```

```
Number of agents [50]: 50
--- random-32-32-20, 50 agents ---            (what already exists for this agent count)
  1) Generate scenarios + expert (EECBS) plans
  2) Train a model from scratch (Edge / GNN), then optionally test
  3) Test: PP vs EECBS vs trained PP on N test instances (comparison plots)
  4) Compare on one new random scenario
  5) Change number of agents
  0) Quit
```

1. **Generate** asks how many training and test scenarios to make, then solves them with EECBS.
2. **Train** asks which model and how many epochs. Every epoch prints the training loss and validation delay. At
   the end it saves and opens the loss graph, then asks whether to test.
3. **Test** asks how many test instances to use. It runs original PP and every trained model, reads the EECBS plan
   stored with each instance, and writes comparison maps plus a summary chart and table. If no model is trained
   yet, it compares PP with EECBS only.

**The smoke config** checks that everything runs in a couple of minutes, without EECBS. It uses 12 agents, Python
CBS as the expert, 32 training scenarios (+8 for validation), 3 epochs and 6 test scenarios. At 12 agents PP is
already almost optimal, so don't expect learned PP to improve on it there. The paper's effect needs 50–100 agents.

## Commands (same steps, for scripts)

```bash
python main.py generate --agents 50                       # train + test scenarios with EECBS plans
python main.py train    --agents 50 --model gnn --epochs 30
python main.py test     --agents 50 --num 10 --animate    # first 10 test instances, with GIFs
python main.py solve    --agents 50 --seed 7              # one brand-new random scenario, compared
python main.py evaluate --agents 50 75 100                # paper's Table 1 (all test instances, no plots)
python main.py all      --agents 50                       # generate -> train -> test
```

`--set key=value` overrides any entry of `configs/paper.json`, e.g. `--set batch_size=16 num_test=10`.

## Where things are saved

```
data/scenarios/random-32-32-20/n50/
    train.json.gz, test.json.gz     generated INPUT data: scenarios, PP priority orders, EECBS paths
outputs/random-32-32-20/n50/
    checkpoints/edge.pt, gnn.pt     trained models (the epoch with the best validation delay)
    training/<model>/curves.png     loss graph (loss function in the title) + validation delay
    training/<model>/history.csv    the numbers behind the graph (also history.json)
    tensorboard/<model>/            TensorBoard logs of training
    tensorboard/test/               comparison maps and summary of the last test
    test/                           emptied and rewritten on every test:
        summary.png                 bar chart: delay of every method on every instance + mean
        summary.md / summary.csv    Table-1 style numbers (delay, gap vs EECBS, time, PP failures)
        per_instance.csv            delay per instance and method
        instance_000/comparison.png PP (orig) | PP (Edge) | PP (GNN) | EECBS side by side
        instance_000/*.png          each method on its own (+ animation.gif with --animate)
    plots/                          single-scenario comparisons from option 4 / `main.py solve`
outputs/results/table1.md           from `main.py evaluate`
```

The datasets are `.json.gz`, i.e. gzip-compressed JSON. 500 scenarios × 100 agents of expert paths would be tens
of MB uncompressed. They are inputs, not results, which is why they sit in `data/` rather than `outputs/`.
Python reads them with `gzip.open`, and 7-Zip can unpack them if you want to look inside.

Training a model again replaces that model's checkpoint, `training/<model>/` and `tensorboard/<model>/`. Testing
again replaces `test/` and `tensorboard/test/`.

### TensorBoard

```bash
pip install tensorboard
tensorboard --logdir outputs        # then open http://localhost:6006
```

- **SCALARS:** `train/loss` per epoch, `train/batch_loss` per mini-batch, `val/delay` with the original-PP and
  expert reference lines, and PP failures, zero-gradient scenarios and epoch time.
- **HISTOGRAMS:** the Edge model's edge costs, one per epoch.
- **IMAGES:** the loss graph, plus each test's comparison maps and summary chart.

### Comparing single scenarios

Option 4 in the menu, or `main.py solve`. It prints SOC, delay and makespan per method, which agents changed route
compared with the original PP, and whose arrival time changed. It writes `comparison.png` to
`outputs/<map>/n<N>/plots/<scenario>/`: agents whose path differs from the original PP are coloured and numbered,
unchanged ones are grey.

```bash
python main.py solve --agents 50                 # delay table for all test scenarios, plots the most interesting
python main.py solve --agents 50 --index 7       # a specific test scenario
python main.py solve --agents 50 --seed 123      # a brand-new random scenario, solved by EECBS on the spot
```

## Metrics

- **Delay** = PP sum-of-costs − Σ shortest-path lengths ignoring other agents (the collision-free lower bound).
- **Gap** = (mean delay − mean EECBS delay) / mean EECBS delay × 100. This reproduces the paper's percentages,
  e.g. 56.20 vs 23.96 → 134.6 %.
- Agents stay at their goal after arriving. Sum-of-costs counts each agent's final arrival time.

## Choices where the paper leaves details open

These are documented so they can be changed if the authors clarify them:

- **Directed costs.** The loss compares *directed* edge-usage vectors and Eq. 5 adds ∂L/∂ŷ to *w*, so *w* and *y*
  have to share one index space. Every directed move u→v and every wait self-loop v→v therefore has its own cost
  (M = 3,359 for random-32-32-20).
- **Epochs and batch size** are not reported. The defaults are 30 epochs and B = 8. Model selection on validation
  delay makes the epoch count less critical.
- **Scenarios** are uniformly random, distinct start and goal cells in the largest connected component. MovingAI
  provides only 25 `.scen` files per map, so the 525 scenarios are sampled. `read_scen` can load the official
  ones if you want them.
- **PP failures with learned weights.** In training, a scenario whose forward or perturbed PP call fails gets a
  zero gradient. In validation and testing it falls back to the original PP plan and is counted in the logs and
  results. The paper doesn't say how it handles this case.
- **GNN inputs.** Node features are start and goal counts (the demand signal), sinusoidal (row, col) embeddings,
  node degree (local obstacle context) and agent-ID embeddings. The "agent ID" is the agent's rank in the PP
  priority order, embedded at its start and its goal node. The edge MLP's last layer starts at zero, so training
  begins from exactly *w* = 1, i.e. the original PP, just like the Edge model.
- **w_min = 1e-3.** Predicted costs are clamped to this value before planning, because Dijkstra and A\* need
  strictly positive costs.
- **The scale of the perturbation.** With the mean-reduced loss (Eq. 4), λ·∂L/∂ŷ is about 2/M ≈ 6e-4 per edge.
  That is large enough to change PP's tie-breaks between equally long routes, and in practice nearly every
  scenario gets a non-zero gradient. `loss_reduction: "sum"` is available if you want larger perturbations.

PP here is pure Python (about 25–35 ms per instance for 50–100 agents), so absolute timings will differ from
the paper's.

## Repository layout

```
main.py                 interactive menu, or commands: generate | train | test | solve | evaluate | all
configs/                paper.json (full replication), smoke.json (quick check)
mapf_nco/
  grid.py               MovingAI map -> grid graph with directed edges and self-loops
  scenarios.py          scenario sampling, priority orders, dataset and .scen I/O
  solvers/pp.py         Prioritized Planning with learned edge costs
  solvers/eecbs.py      wrapper around the EECBS binary
  solvers/cbs.py        Python CBS fallback expert
  blackbox.py           black-box gradient through PP
  models.py             Edge and GraphSAGE models
  compare.py / viz.py    comparisons, pictures, animations, training curves
  paths.py / tb.py       output folder layout, TensorBoard logging
  train.py / evaluate.py / experts.py / metrics.py
scripts/build_eecbs.sh  builds EECBS into third_party/
tests/                  pytest suite
```

## Citation

```bibtex
@inproceedings{captari2025learned,
  title     = {Learned Representations Enhance Multi Agent Path Planning},
  author    = {Captari, Marius and van Hoof, Herke},
  booktitle = {ICML 2025 Workshop on Programmatic Representations for Agent Learning},
  year      = {2025}
}
```

Related work this code builds on: EECBS (Li, Ruml & Koenig, AAAI 2021), black-box differentiation of
combinatorial solvers (Pogančić et al., ICLR 2020), MAPF benchmarks (Stern et al., SoCS 2019).
