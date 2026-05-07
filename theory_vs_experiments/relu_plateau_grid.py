import argparse, math, time
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent))
from teacher_full_runner import run_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="relu_plateau_grid")
    ap.add_argument("--budget-s", type=float, default=38.0)
    ap.add_argument("--chunk-steps", type=int, default=4000)
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    seed = 5; n_steps = 8000

    # ReLU plateau-search grid: large r_t and EoS LR (η ≈ 2) give cleanest plateau-with-learning.
    # The "winner" set from our 14-config grid: rt∈{8, 12} at η∈{1.5, 2.0} with rs=50.
    configs = [
        # (teacher, p, r_t, r_s, n, eta)
        ("relu", 100, 4,  50, 15000, 2.0),
        ("relu", 100, 6,  50, 15000, 1.5),
        ("relu", 100, 8,  50, 15000, 1.5),
        ("relu", 100, 8,  50, 15000, 1.0),
        ("relu", 100, 8,  30, 15000, 1.5),
        ("relu", 100, 12, 50, 15000, 1.5),
        ("relu", 100, 16, 50, 15000, 1.5),
        ("relu", 100, 16, 50, 15000, 2.0),
        ("relu", 100, 20, 50, 15000, 2.0),
    ]

    t_start = time.time()
    for ta, p, r_t, r_s, n, eta in configs:
        if time.time() - t_start > args.budget_s:
            print("[budget global] stop"); break
        ckpt = out_dir / f"{ta}_p{p}_rt{r_t}_rs{r_s}_n{n}_eta{eta}_seed{seed}.npz"
        run_one(ta, p, r_t, r_s, n, eta, seed, n_steps, log_every=20,
                ckpt_path=ckpt, chunk_steps=args.chunk_steps,
                budget_s=max(5.0, args.budget_s - (time.time() - t_start)))


if __name__ == "__main__":
    main()
