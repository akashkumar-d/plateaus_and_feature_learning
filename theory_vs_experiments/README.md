# theory_vs_experiments

Engine, runners, and the consolidated theory-vs-experiment overlay plot.

## Engine

`m3_runner.py`, `teacher_activations.py` — imported by every runner.

## Runners

```
python relu_plateau_grid.py       # writes experiments_out/relu_plateau_grid/*.npz
python teacher_full_runner.py     # writes experiments_out/teacher_full/*.npz (smooth GELU/SiLU/tanh)
```

Each is restartable (re-run the same command until each config reports `[skip]`).

## Plotter

```
python plot_theory_vs_experiment_consolidated.py
   # → experiments_out/theory_vs_experiment_consolidated.png
```

The plotter accepts any `n=*` checkpoint, so it works on either the
short bundled runs or fresh long sweeps. The bundled
`experiments_out/relu_plateau_grid/` reproduces the ReLU points of
`theory_vs_experiment_consolidated.png` directly; the smooth-teacher
points (GELU, SiLU at large `r_t`) require running
`teacher_full_runner.py` first.

Requirements: `numpy`, `matplotlib`. Tested on Python 3.10.
