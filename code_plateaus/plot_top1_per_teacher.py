from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from plot6_top_configs import (
    parse_filename,
    plateau_quality_score,
    eff_rank_match_score,
    combined_score,
    plot_one_column,
)


SCORE_FNS = {
    "flatness_growth_drop": lambda z, rt: plateau_quality_score(z),
    "eff_rank_match":       lambda z, rt: eff_rank_match_score(z, rt),
    "combined":             lambda z, rt: combined_score(z, rt),
}

# Per-activation default score mode. Combined works well for smooth
# teachers; ReLU's eff-rank closeness is uniformly tiny so the
# combined product zeros out and we fall back to the 3-factor score
# that doesn't require eff-rank-to-r_t convergence.
DEFAULT_PER_ACT_MODE = {
    "ReLU": "flatness_growth_drop",
    "SiLU": "combined",
    "GELU": "combined",
}


def best_in_dir(in_dir: Path, score_mode: str):
    """Return (best_ckpt_path, panel, p, rt, rs, eta, score) or None
    if no valid checkpoint is found."""
    if not in_dir.exists():
        print(f"  [warn] {in_dir} does not exist; skipping.")
        return None
    score_fn = SCORE_FNS[score_mode]
    best = None
    for ckpt in sorted(in_dir.glob("*.npz")):
        try:
            z = np.load(ckpt)
            panel, p, rt, rs, eta = parse_filename(ckpt.stem)
            if any(v is None for v in (p, rt, rs, eta)):
                continue
            score = score_fn(z, rt)[0]   # first element is the score
            if best is None or score > best[-1]:
                best = (ckpt, panel, p, rt, rs, eta, score)
        except Exception as e:
            print(f"  [skip] {ckpt.name}: {e}")
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--relu-dir", default="../checkpoints/grid_14configs")
    ap.add_argument("--silu-dir", default="../checkpoints/grid_14configs_silu")
    ap.add_argument("--gelu-dir", default="../checkpoints/grid_14configs_gelu")
    ap.add_argument("--out", default="three_best_paper.png")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--score-mode", default="auto",
                    choices=["auto", "combined", "flatness_growth_drop",
                             "eff_rank_match"],
                    help="Score for top-1 selection. 'auto' uses "
                         "flatness_growth_drop for ReLU and combined "
                         "for SiLU/GELU. Override to apply the same "
                         "score to all three.")
    args = ap.parse_args()

    teachers = [
        ("ReLU", Path(args.relu_dir).resolve()),
        ("SiLU", Path(args.silu_dir).resolve()),
        ("GELU", Path(args.gelu_dir).resolve()),
    ]

    picks = []
    for name, in_dir in teachers:
        mode = (DEFAULT_PER_ACT_MODE[name]
                if args.score_mode == "auto" else args.score_mode)
        print(f"  {name}: directory = {in_dir}, score-mode = {mode}")
        best = best_in_dir(in_dir, mode)
        if best is None:
            print(f"  [warn] no valid checkpoint found for {name}; "
                  f"column will be left blank.")
            picks.append((name, mode, None))
            continue
        ckpt, panel, p, rt, rs, eta, score = best
        print(f"    picked: {ckpt.name}  "
              f"(p={p}, r_t={rt}, r_s={rs}, eta={eta})  score={score:.4g}")
        picks.append((name, mode, best))

    valid = [(name, mode, best) for (name, mode, best) in picks if best is not None]
    if not valid:
        raise RuntimeError("No valid checkpoints found in any directory. "
                           "Check --relu-dir / --silu-dir / --gelu-dir paths.")

    n_cols = len(valid)
    fig, axes = plt.subplots(4, n_cols, figsize=(5.0 * n_cols, 4.0 * 4),
                             squeeze=False)
    for col, (name, mode, best) in enumerate(valid):
        ckpt, panel, p, rt, rs, eta, score = best
        plot_one_column(axes[:, col], ckpt, panel, p, rt, rs, eta, score)
        # Override the auto-generated title to be "Teacher: cfg"
        axes[0, col].set_title(
            f"{name}: $p$={p}, $r_t$={rt}, $r_s$={rs}, $\\eta$={eta}\n"
            f"(score-mode={mode}, score={score:.3g})",
            fontsize=10)

    fig.suptitle(
        "Top-1 cleanest plateau-with-feature-learning configuration "
        "per teacher activation.\n"
        "Row 1: loss decomposition. Row 2: AGOP direction-only "
        "alignment. Row 3: per-eigvec $V$-mass. Row 4: AGOP effective rank.",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = Path(args.out).resolve()
    fig.savefig(out_path, dpi=args.dpi); plt.close(fig)
    print(f"\nSaved {out_path} ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
