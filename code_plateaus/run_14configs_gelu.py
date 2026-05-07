import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _up in (_HERE, *_HERE.parents):
    if (_up / "shared" / "m3_runner.py").exists():
        sys.path.insert(0, str(_up / "shared")); break
    if (_up / "m3_runner.py").exists():
        sys.path.insert(0, str(_up)); break
else:
    raise RuntimeError("Cannot find m3_runner.py")
from m3_runner import run_one


CONFIGS = [
    ("a", 100, 10,  50, 0.7),
    ("b", 100, 12,  50, 0.7),
    ("c", 100,  4,  50, 0.85),
    ("d", 100,  6,  50, 0.7),
    ("e", 100,  8,  30, 0.7),
    ("f", 100,  8,  50, 0.5),
    ("g", 100,  8,  50, 0.6),
    ("h", 100,  8,  50, 0.7),
    ("i", 100, 12,  50, 0.6),
    ("j", 100, 12,  50, 0.85),
    ("k", 100, 16,  50, 0.7),
    ("l", 100, 16,  50, 0.85),
    ("m", 100, 20,  50, 0.7),
    ("n", 100, 12,  80, 0.7),
]
TEACHER = "gelu"
SEED = 5
N = 15000
N_STEPS = 10000
LOG_EVERY = 1


def ckpt_path(out_dir, panel, p, rt, rs, eta):
    return out_dir / f"{panel}_p{p}_rt{rt}_rs{rs}_n{N}_eta{eta}_seed{SEED}.npz"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="../checkpoints/grid_14configs_gelu")
    ap.add_argument("--budget-s", type=float, default=600.0)
    ap.add_argument("--chunk-steps", type=int, default=500)
    ap.add_argument("--n-steps", type=int, default=N_STEPS)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}\nTeacher: {TEACHER}")

    t_start = time.time()
    for panel, p, rt, rs, eta in CONFIGS:
        elapsed = time.time() - t_start
        if elapsed > args.budget_s:
            print(f"[budget global] stop after {elapsed:.0f}s; rerun to continue.")
            break
        ckpt = ckpt_path(out_dir, panel, p, rt, rs, eta)
        per_cfg_budget = max(5.0, (args.budget_s - elapsed) /
                             max(1, len(CONFIGS) - CONFIGS.index((panel, p, rt, rs, eta))))
        run_one(TEACHER, p, rt, rs, N, eta, SEED, args.n_steps,
                log_every=LOG_EVERY, ckpt_path=ckpt,
                chunk_steps=args.chunk_steps, budget_s=per_cfg_budget)


if __name__ == "__main__":
    main()
