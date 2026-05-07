import argparse
import sys
import time
from pathlib import Path

# --- locate shared/m3_runner.py (walks up parents)
_HERE = Path(__file__).resolve().parent
for _up in (_HERE, *_HERE.parents):
    if (_up / "shared" / "m3_runner.py").exists():
        sys.path.insert(0, str(_up / "shared")); break
    if (_up / "m3_runner.py").exists():
        sys.path.insert(0, str(_up)); break
else:
    raise RuntimeError("Cannot find m3_runner.py — place it next to this "
                       "script or in a `shared/` folder along the path.")
from m3_runner import run_one


# (panel_letter, p, r_t, r_s, eta)
CONFIGS = [
    ("a", 100, 10,  50, 1.5),
    ("b", 100, 12,  50, 1.5),
    ("c", 100,  4,  50, 2.0),
    ("d", 100,  6,  50, 1.5),
    ("e", 100,  8,  30, 1.5),
    ("f", 100,  8,  50, 1.0),
    ("g", 100,  8,  50, 1.25),
    ("h", 100,  8,  50, 1.5),
    ("i", 100,  8,  80, 1.5),
    ("j", 150,  8,  50, 1.5),
    ("k", 150,  8,  80, 1.5),
    ("l", 200, 12,  80, 2.0),
    ("m", 200,  8, 100, 2.5),
    ("n", 200,  8,  80, 2.0),
]
SEED = 5
N = 15000          # training data size
N_STEPS = 10000    # target step count (matches the original 10k-step grid)
LOG_EVERY = 20


def ckpt_path(out_dir: Path, panel: str, p: int, r_t: int, r_s: int, eta: float) -> Path:
    return out_dir / f"{panel}_p{p}_rt{r_t}_rs{r_s}_n{N}_eta{eta}_seed{SEED}.npz"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir",
                    default="../checkpoints/grid_14configs",
                    help="Where to write the .npz checkpoints.")
    ap.add_argument("--budget-s", type=float, default=600.0,
                    help="Wall-clock budget per overall script call (sec).")
    ap.add_argument("--chunk-steps", type=int, default=4000,
                    help="How many steps to advance per config per call.")
    ap.add_argument("--n-steps", type=int, default=N_STEPS,
                    help=f"Target step count (default {N_STEPS}).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    t_start = time.time()
    for panel, p, r_t, r_s, eta in CONFIGS:
        elapsed = time.time() - t_start
        if elapsed > args.budget_s:
            print(f"[budget global] stop after {elapsed:.0f}s; rerun to continue.")
            break
        ckpt = ckpt_path(out_dir, panel, p, r_t, r_s, eta)
        per_config_budget = max(5.0, (args.budget_s - elapsed) / max(
            1, len(CONFIGS) - CONFIGS.index((panel, p, r_t, r_s, eta))))
        run_one("relu", p, r_t, r_s, N, eta, SEED, args.n_steps,
                log_every=LOG_EVERY,
                ckpt_path=ckpt,
                chunk_steps=args.chunk_steps,
                budget_s=per_config_budget)


if __name__ == "__main__":
    main()
