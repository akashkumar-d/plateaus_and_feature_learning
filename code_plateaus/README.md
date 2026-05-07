# code_plateaus

Engine, runners, and plotters for the 1-HL teacher–student matrix-Muon experiments.

## Engine

`m3_runner.py`, `teacher_activations.py` — imported by every runner and plotter.
`run_one(...)` runs a single (p, r_t, r_s, eta) configuration and writes a
`.npz` checkpoint with the per-step loss, AGOP metrics, and final weights.

## Runners

```
python run_14configs.py        --budget-s 600 --chunk-steps 4000   # ReLU teacher
python run_14configs_silu.py   --budget-s 600 --chunk-steps 4000   # SiLU teacher
python run_14configs_gelu.py   --budget-s 600 --chunk-steps 4000   # GELU teacher
```

Each writes per-config `.npz` files to `unified_v3d/`. Re-run until each config reports `[skip]`.

## Plotters

```
python plot_plateau_search.py                       # full per-config grid
python plot6_top_configs.py --in-dir unified_v3d \
       --out top6_relu.png --score-mode flatness_growth_drop
python plot_top1_per_teacher.py --out three_best_paper.png
python make_appendix_tables.py                      # writes tables/*.tex
```

Bundled `unified_v3d/` checkpoints and PNGs reproduce the paper figures
(`top6_relu.png`, `top6_silu.png`, `top6_gelu.png`, `three_best_paper.png`)
and the appendix tables (`tables/tab_*.tex`) without re-running the experiments.

Requirements: `numpy`, `matplotlib`. Tested on Python 3.10.
