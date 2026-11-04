# MAPF NCO

A replication of **"Learned Representations Enhance Multi Agent Path Planning"**
(Marius Captari and Herke van Hoof, ICML 2025 Workshop on Programmatic Representations for Agent Learning).

A cost is learned for every edge of the MAPF grid (the paper's Edge model). Prioritized Planning (PP) plans on
those costs, and the costs are trained so that PP's plans match expert plans from EECBS. PP is not
differentiable, so gradients come from black-box differentiation (Pogančić et al., 2019): one extra PP call on
perturbed costs per scenario. The paper also has a GNN variant; it is not included here. As an extension, the
costs can also come from a **potential model (PotP)**: a few learned strengths on map features, such as moving
towards obstructions or into single-lane passages.

This repository started as a fork of [AI4REALNET/flatland-blackbox](https://github.com/AI4REALNET/flatland-blackbox),
which applies the same idea to Flatland railway graphs. That Flatland-specific code (rail graphs, proxy nodes,
city generators, flowtime) has been replaced by the grid MAPF setting the paper uses.

## How the code maps to the paper

| Paper | Code |
|---|---|
| §3.1 grid graph, wait self-loops, directed edges | `mapf_nco/grid.py` |
| §3.1 edge-usage vector *y* (Eq. 1) | `mapf_nco/metrics.py` → `edge_usage` |
| §3.1/3.3 PP: space-time A\*, reservation table, learned costs for *g*, unit costs for timing, true-distance heuristic recomputed under the learned weights | `mapf_nco/solver_pp.py` |
| §3.2 MSE loss on directed usage (Eq. 4), perturbation (Eq. 5), surrogate gradient (Eq. 6) | `mapf_nco/blackbox.py` |
| §3.3 / §4 Edge model: one learned cost per directed edge, shared by all scenarios (the paper's GNN variant is not included) | `mapf_nco/models.py` |
| Algorithm 1, Adam (lr 5e-4), λ = 1.0, validation-delay checkpointing | `mapf_nco/train.py` |
| §4 EECBS experts (suboptimality 1.0 for 50 agents, 1.05 for 75/100) | `mapf_nco/solver_eecbs.py`, `mapf_nco/experts.py` |
| §4 priority order: ascending true distance, otherwise one fixed random permutation | `mapf_nco/scenarios.py` → `assign_priority_order` |
| §4 random-32-32-20 map, 500 train scenarios (20 % for validation), 25 test scenarios (default here: 100, for steadier averages; enter 25 to match the paper exactly) | `configs/paper.json` |
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

#### EECBS on a Linux server without sudo

Install the build tools into your conda environment, then build:

```bash
conda activate flatland-blackbox          # or whichever env you use
conda install -c conda-forge boost cmake  # plus gxx_linux-64 if the server has no g++
bash scripts/build_eecbs.sh               # picks up Boost from the active conda env
```

The script checks for git, cmake, a C++ compiler and Boost before building, and says exactly what's missing.
If `python main.py` still can't find EECBS, it lists every place it looked and why each one failed.

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
python main.py          # settings from configs/paper.json
```

```
  1) Generate scenarios + EECBS, CBS and PP plans
  2) Experiment: train a model (Edge / Potential / Potential+Edge; choose hyperparameters)
  3) Test: saved test scenarios, or new random scenarios (comparison plots)
  4) Change number of agents
  5) Sweep: generate + queue of experiments + tests
  6) Clear outputs (models, plots, results; generated data is kept)
  0) Quit
```

1. **Generate** samples the training and test scenarios and pre-computes three plans for each:
   - **EECBS:** the expert plan the models learn from.
   - **PP:** the original prioritized-planning plan.
   - **CBS:** the optimal plan, from EECBS run at suboptimality 1.0. It has a time limit, so a few large
     instances may have no CBS plan. At 50 agents the expert is already optimal, so it's reused.

   For data generated earlier, choose "add missing PP/CBS plans" to keep the EECBS plans and only add the rest.
2. **Experiment** asks for every hyperparameter (model, planner, epochs, batch size, learning rate, lambda,
   objective, loss, w_min, validation fraction and seed), trains the model(s) and opens the loss graph. It
   doesn't test.
   - **Model** sets how the edge costs are learned: `edge` (the paper, the default), `potential`, or
     `potential_edge`. See [Potential-based edge costs](#potential-based-edge-costs-potp).
   - **Planner** picks the downstream algorithm the costs are learned for: `pp` (the paper), `cbs`, or `both`
     (the default), a separate model for each. See [CBS as the downstream planner](#cbs-as-the-downstream-planner).
3. **Test** asks whether to use the saved test scenarios or new random ones. For new ones it also asks the number
   of agents, which can differ from the training count, then solves them with EECBS and CBS on the spot. Every
   test replaces `test/`. Each `comparison.png` has three parts:
   - **Header:** the training hyperparameters and the best epoch.
   - **Top row:** the plans of PP (orig), PP (Edge), CBS (Edge) if trained, EECBS and CBS.
   - **Bottom row:** each model's learned edge costs, as a blue (cheaper) to red (dearer) gradient.
6. **Clear outputs** deletes results, after a y/N confirmation (default no):
   - `1` the models, plots and tests for the current agent count (`outputs/<map>/n<N>/`);
   - `2` all sweeps (`outputs/sweeps/`);
   - `3` everything in `outputs/`.

   The generated scenarios in `data/` are never deleted, so you don't have to regenerate.

### Option 5: Sweep (everything in one go)

A sweep generates the data if needed, then runs a queue of experiments. For each hyperparameter you can give
several values separated by commas, e.g. planner `pp, cbs` and learning rate `1e-4, 5e-4`, and it runs every
combination (4 experiments here). Every experiment uses the agent count you selected. Each experiment:
- trains the Edge model from scratch, for each planner it uses;
- tests them on the same saved test scenarios (you choose how many);
- optionally (answer **y**; the default is no) also tests them on a second set of new random scenarios with a
  different agent count. That set is generated once and shared, so experiments are compared fairly.

Results go to `outputs/sweeps/exp_1/`, `exp_2/`, …: each sweep takes the smallest free number, so after
deleting `exp_2` the next sweep is `exp_2` again. The agent count and settings are in `sweep_summary.md`.
- **`sweep_summary.md` / `.csv`:** one row per experiment with its settings, best epochs, validation delay and both
  tests' delays. It's updated after every experiment, so a long sweep can be checked while it runs.
- **`sweep_summary.png`:** a bar chart of every experiment on both test sets, against original PP and EECBS.
- **`exp_01_planner-pp_lr-0.0001/`, `exp_02_…/`, …:** one folder per experiment, named after the settings
  that vary in the sweep. Each holds that experiment's models, loss graphs, TensorBoard logs and comparison plots
  (`test/` and `test_n<M>/`).

The same from the command line:

```bash
python main.py sweep --agents 50 --grid lr=1e-4,5e-4 lam=0.5,1,2 --num 25    # add --second_agents 25 for a second test set
```

Sweepable: model, planner, objective, makespan_tau, makespan_weight, epochs, batch_size, lr, lam, loss_reduction, w_min,
val_fraction, seed.

## Commands (same steps, for scripts)

```bash
python main.py generate   --agents 50                     # add --augment to keep existing data
python main.py experiment --agents 50 --epochs 30 --lr 5e-4 --lam 1.0 --batch_size 8
python main.py experiment --agents 20 --planner cbs      # only the CBS models (pp / cbs / both)
python main.py experiment --agents 50 --model potential_edge --planner pp
python main.py sweep      --agents 50 --grid model=edge,potential,potential_edge --num 25   # compare them
python main.py test       --agents 50 --num 10            # saved test scenarios
python main.py test       --agents 50 --random 5 --test_agents 75   # 50-agent models, new 75-agent scenarios
python main.py evaluate   --agents 50 75 100              # paper's Table 1, no plots
python main.py clean      --agents 50 --what agents       # or --what sweeps / all; add --yes to skip the prompt
```

`--set key=value` overrides any entry of `configs/paper.json`.

## Where things are saved

```
data/scenarios/random-32-32-20/n50/
    train.json.gz, test.json.gz     generated INPUT data: scenarios, PP priority orders,
                                    and the EECBS, PP and CBS plans
outputs/random-32-32-20/n50/
    checkpoints/<model>.pt          trained PP models: edge.pt, potential.pt, potential_edge.pt (the
                                    epoch with the best validation delay, or makespan for a makespan
                                    objective)
    checkpoints/<model>_cbs.pt      the same for CBS (planner cbs or both)
    training/curves.png             all models together: loss (formula in the title) and
                                    validation metric, with the selected epochs starred
    training/history.csv            the numbers behind the graph, one row per model and epoch
    tensorboard/<model>/            TensorBoard logs of training
    tensorboard/test/               comparison maps and summary of the last test
    test/                           emptied and rewritten on every test:
        summary.png                 bar chart: delay of every method on every instance + mean
        summary.md / summary.csv    Table-1 style numbers (delay, gap vs EECBS, makespan, time,
                                    PP failures)
        per_instance.csv            delay and makespan per instance and method
        instance_000/comparison.png header: hyperparameters; top: PP (orig) | PP (Edge) |
                                    [CBS (Edge)] | EECBS | CBS; bottom: learned edge costs
        instance_000/*.png          each method on its own (+ animation.gif with --animate)
outputs/sweeps/exp_<k>/            option 5 (sweep): sweep_summary.*, exp_01_<settings>/ ...
outputs/results/table1.md           from `main.py evaluate`
```

The datasets are `.json.gz`, i.e. gzip-compressed JSON. 500 scenarios × 100 agents of expert paths would be tens
of MB uncompressed. They are inputs, not results, which is why they sit in `data/` rather than `outputs/`.
Python reads them with `gzip.open`, and 7-Zip can unpack them if you want to look inside.

Training a model again replaces that model's checkpoint, its curves and rows in `training/`, and
`tensorboard/<model>/`. Testing
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

## Metrics

- **Delay** = PP sum-of-costs − Σ shortest-path lengths ignoring other agents (the collision-free lower bound).
- **Gap** = (mean delay − mean EECBS delay) / mean EECBS delay × 100. This reproduces the paper's percentages,
  e.g. 56.20 vs 23.96 → 134.6 %.
- Agents stay at their goal after arriving. Sum-of-costs counts each agent's final arrival time.

## Potential-based edge costs (PotP)

Instead of one free cost per edge, the potential model builds the costs from a few properties of the map. The
assumption is that edges *towards obstructions* and into *single-lane passages* should cost more, and that
one-way lanes may help. For a move u → v (waits keep cost 1):

```
w = 1 + θ_near·near(v) + θ_toward·max(0, near(v) − near(u)) + θ_corridor·corridor(u→v) + θ_dead_end·dead_end(v)
      + lane_row[r]·(±1 east/west)  or  lane_col[c]·(±1 south/north)
```

- **`near(v) = 1/d(v)`**, with d(v) the number of steps from v to the nearest blocked cell or the map border. This
  is a repulsive potential, as in potential-field planning. `toward` is how much a move climbs it.
- **`corridor`** is 1 when both cells beside v, across the direction of the move, are blocked: a single-lane
  passage where agents can't pass each other. **`dead_end`** is 1 when v has only one free neighbour.
- **Lanes** are one-way preferences: one learned value per row and per column makes one direction along it
  cheaper and the opposite one dearer.
- **Parameters:** that's 4 strengths + 32 + 32 lane values = 68, against 3,359 for the Edge model, so it can't
  memorise a small training set. It is trained with the same black-box gradient
  (`mapf_nco/potential.py`, `PotentialModel` in `mapf_nco/models.py`). The training log prints the strengths
  after every epoch, and test plots list them in the header.

Three choices for `model`:

| model | what is trained |
|---|---|
| `edge` | the paper's Edge model, starting from cost 1 everywhere |
| `potential` | the 68 potential parameters, starting from `potential_init` |
| `potential_edge` | first `potential`, then the Edge model starting from the trained potential costs and adjusted per edge (both checkpoints are kept) |

**Starting strengths** (`potential_init` in `configs/paper.json`; the experiment menu asks for them): toward 0.2,
corridor 0.2, the rest 0. They come from a check with no training, PP on random scenarios with and without
these costs (no expert involved):

| added cost | 50 agents (120 scenarios) | 75 agents (60 scenarios) |
|---|---|---|
| toward 0.2 | delay −5.7 % (better in 70, worse in 38) | |
| toward 0.2 + corridor 0.3 | −6.8 % (70 / 40) | −7.7 % (43 / 12) |
| toward 0.2 + corridor 0.1 | −7.2 % (78 / 33) | −6.0 % (42 / 15) |
| corridor 1.0 | +12.5 % (too strong: detours) | |
| alternating one-way lanes 0.3 | +80 % (forced detours) | |

So steering agents away from obstructions and out of single-lane passages helps PP even before training. Fixed
one-way lanes hurt, which is why the lanes start at 0 and only change through training. With non-unit costs PP
fails on about 1 in 100 scenarios; those count as failures and fall back to the original PP plan.

Two things to keep in mind when reading the results:

- **The training loss is a poor guide for this model.** The paper's loss measures which of many equally short
  routes each agent takes, compared with EECBS. The potential model changes costs across the whole map at once,
  so it can't copy EECBS route by route. Its starting strengths roughly double that loss even where they lower
  the delay. On single scenarios, the black-box gradient still points the right way: it ends far below the
  same run with the gradient's sign flipped. But the loss often doesn't fall below its starting value. The best
  epoch is chosen by validation delay, which is what matters.
- **Epoch 0 is the starting strengths, not the original PP.** For the Edge model, epoch 0 is the original PP, so
  the saved model can never be worse than PP on validation. For `potential` and `potential_edge` it can; the
  training log then says so ("the selected model is worse than the original PP on validation"). With small
  datasets (a dozen validation scenarios) this is mostly noise; compare on 50+ test scenarios.

## Makespan objective (extension, not in the paper)

The paper's loss makes PP copy EECBS, which minimises the **sum** of arrival times. Setting `objective` to
`makespan` instead trains the edge weights to minimise the **latest** arrival time in a scenario. It needs no
expert plans.

- **Loss:** a smooth maximum of the agents' arrival times, L = τ·log Σᵢ exp(Tᵢ/τ), with Tᵢ = Σₑ ŷᵢₑ. Each agent
  uses one edge per timestep, so Tᵢ is the sum of its own edge usage. As τ → 0 this becomes the exact makespan.
- **Lateness weights:** the gradient with respect to agent i is αᵢ = softmaxᵢ(T/τ), so it's concentrated on the
  agents that finish last.
- **Perturbation:** in PP every agent already takes its fastest route given the agents planned before it. A late
  agent is late because others are in its way, so the perturbation raises the costs of the late agents' routes for
  all *other* agents: agent j gets λ·Σᵢ≠ⱼ αᵢ·rᵢ, where rᵢ marks the cells on agent i's ideal route. Each late agent
  also gets λ·αᵢ added to every one of its own steps. PP is run again with these per-agent costs.
- **Gradient:** the shared edge weights get −(ŷ − y_λ)/λ on total usage, the same rule as the paper, by the chain
  rule.
- **`usage+makespan`:** combines the paper's loss with `makespan_weight` × the makespan loss.
- **Best epoch:** with a makespan objective it's chosen by validation makespan instead of delay.

Every test also reports makespan and the **makespan lower bound** (the longest single shortest path; no plan can
beat it). On random-32-32-20 with 50 agents, PP is on average only about 2 steps above that bound and already
optimal in about half the scenarios, so gains are small by nature. In a quick check, the Edge model closed roughly
15% of that gap. Set it in the experiment menu, or use
`python main.py experiment --objective makespan --lam 1 --lr 0.002`.

## CBS as the downstream planner

The paper only uses the learned costs inside PP. With `planner` set to `cbs` (or `both`, the default), the Edge
model is also trained for **CBS**, as a separate model (`edge_cbs.pt`).

- **How CBS uses the costs:** the low-level search adds up the learned cost of every move and wait, and the
  high-level search expands the node with the smallest total learned cost. So CBS returns the plan that
  minimises Σₑ wₑ·yₑ. Every action still takes one timestep. This is `WeightedCBS` in `mapf_nco/solver_cbs.py`.
- **Gradient:** the same black-box rule as for PP (two CBS calls per scenario). Only the planner differs.
- **Time limit:** each CBS call during training stops after `cbs_train_time_limit` seconds (default 10). A run that
  times out gives no gradient, and in validation and testing it falls back to the original CBS plan and counts as
  a failure. Python CBS is practical up to roughly 20-30 agents on random-32-32-20, not at 50-100.
- **What to expect:** with unit costs, CBS is already optimal for the sum of costs. With the paper's `usage`
  objective and an optimal expert, it usually reproduces the expert already, so the loss and gradient are close to
  zero and the delay can't improve. The learned costs can still change *which* optimal plan CBS finds and how
  hard it searches, so the best epoch is chosen by validation delay (or makespan), and ties go to the epoch with
  fewer CBS high-level nodes. The `makespan` objective gives CBS something to improve, because CBS minimises the
  sum of costs, not the makespan.
- **Testing:** when CBS models are loaded, the original CBS is also run live with the same time limit, so the
  times in the table compare fairly. The table gets a CBS (Edge) column.

## Choices where the paper leaves details open

These are documented so they can be changed if the authors clarify them:

- **Directed costs.** The loss compares *directed* edge-usage vectors and Eq. 5 adds ∂L/∂ŷ to *w*, so *w* and *y*
  have to share one index space. Every directed move u→v and every wait self-loop v→v therefore has its own cost
  (M = 3,359 for random-32-32-20).
- **Epochs and batch size** are not reported. The defaults are 20 epochs (validation delay levelled off by
  about epoch 15 in our runs) and B = 8. Model selection on validation
  delay makes the epoch count less critical.
- **Scenarios** are uniformly random, distinct start and goal cells in the largest connected component. MovingAI
  provides only 25 `.scen` files per map, so the 525 scenarios are sampled. `read_scen` can load the official
  ones if you want them.
- **PP failures with learned weights.** In training, a scenario whose forward or perturbed PP call fails gets a
  zero gradient. In validation and testing it falls back to the original PP plan and is counted in the logs and
  results. The paper doesn't say how it handles this case.
- **w_min = 1e-3.** Predicted costs are clamped to this value before planning, because Dijkstra and A\* need
  strictly positive costs.
- **The scale of the perturbation.** With the mean-reduced loss (Eq. 4), λ·∂L/∂ŷ is about 2/M ≈ 6e-4 per edge.
  That is large enough to change PP's tie-breaks between equally long routes, and in practice nearly every
  scenario gets a non-zero gradient. `loss_reduction: "sum"` is available if you want larger perturbations.

PP here is pure Python (about 25–35 ms per instance for 50–100 agents), so absolute timings will differ from
the paper's.

## Repository layout

```
main.py                 interactive menu, or commands: generate | experiment | test | sweep | evaluate
configs/paper.json      all settings (paper defaults)
mapf_nco/
  grid.py               MovingAI map -> grid graph with directed edges and self-loops
  scenarios.py          scenario sampling, priority orders, dataset and .scen I/O
  solver_pp.py          Prioritized Planning with learned edge costs
  solver_eecbs.py       wrapper around the EECBS binary
  solver_cbs.py         Python CBS: fallback expert, and CBS with learned costs (planner cbs)
  planners.py           model (edge / potential / potential_edge) and planner (pp / cbs / both) settings, model ids
  potential.py          map features of the potential model (obstructions, single-lane passages, lanes)
  blackbox.py           black-box gradient through PP
  models.py             the Edge model and the potential model
  compare.py / viz.py    test helpers, pictures, animations, training curves
  sweep.py              option 5: queue of experiments, shared second test set, summary
  paths.py / tb.py       output folder layout, TensorBoard logging
  train.py / evaluate.py / experts.py / metrics.py
scripts/build_eecbs.sh  builds EECBS into third_party/
tests/                  pytest suite (test_correctness.py: checks against reference implementations)
```

## Correctness checks

`python -m pytest` runs the whole suite. `tests/test_correctness.py` checks each algorithm against a separately written,
deliberately simple reference:

| Check | Reference |
|---|---|
| Grid graph | 4-neighbour pairs read straight from the `.map` text |
| PP: each agent's path is the cheapest one its higher-priority agents allow | plain Dijkstra over explicit (cell, time) states, no heuristic, constraints re-implemented |
| PP: a single agent's cost equals the weighted shortest path | SciPy Dijkstra |
| PP: lower-priority agents never change higher-priority paths | re-planning a prefix of the priority order |
| CBS returns the optimal sum-of-costs | brute-force joint-state search over all agents |
| Σ y_e = SOC and delay ≥ 0 | definition |
| Black-box gradient | Eqs. 4–6 recomputed by hand |

Each check was also confirmed to catch deliberately introduced bugs: PP ignoring swap conflicts, PP parking on a
goal that a higher-priority agent visits later, waits not using the learned self-loop cost, CBS ignoring swaps, a
flipped gradient sign, and a wrong loss scaling.

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
