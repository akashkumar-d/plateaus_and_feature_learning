import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def running_min(y, w=4):
    out = np.empty_like(y, dtype=float)
    for i in range(len(y)):
        out[i] = float(np.min(y[max(0, i-w+1):i+1]))
    return out


def running_mean(y, w=20):
    y = np.asarray(y, dtype=float); h = w // 2
    out = np.full(len(y), np.nan)
    for i in range(len(y)):
        out[i] = float(np.mean(y[max(0, i-h):min(len(y), i+h+1)]))
    return out


def thr_rank_vs_l2(lams, thr_frac=0.5):
    sorted_lams = np.sort(lams, axis=1)[:, ::-1]
    l2 = np.maximum(sorted_lams[:, 1:2], 1e-30)
    return (lams >= thr_frac * l2).sum(axis=1)


def parse_filename(stem: str):
    p, rt, rs, eta, panel = None, None, None, None, ""
    parts = stem.split("_")
    if parts and len(parts[0]) == 1 and parts[0].isalpha():
        panel = parts[0]
    for tok in parts:
        if tok.startswith("p") and tok[1:].isdigit():
            p = int(tok[1:])
        elif tok.startswith("rt"):
            try: rt = int(tok[2:])
            except ValueError: pass
        elif tok.startswith("rs"):
            try: rs = int(tok[2:])
            except ValueError: pass
        elif tok.startswith("eta"):
            try: eta = float(tok[3:])
            except ValueError: pass
    return panel, p, rt, rs, eta


def combined_score(z, r_t):
    """All five criteria multiplied together. A configuration wins
    only if it satisfies them all simultaneously:

      flatness          : loss is flat in the late half
      delta_cos2_min    : alignment continues to grow in the plateau
      loss_drop_factor  : visible Phase 1 (loss drops by >= ~1 OOM)
      closeness         : AGOP eff rank converges to r_t in late half
      stability         : eff rank has low variance in late half

    score = flatness * min(growth, 1) * loss_drop_factor
            * closeness * stability

    This is the score to use when you want all of the headline visual
    properties at once. Small-r_t configurations (rt=4, 6) typically
    fail the loss_drop_factor (loss already small at init); rising-loss
    configurations (rt=20 with high eta) fail closeness AND
    loss_drop_factor; the winners are usually moderate r_t (8--12)
    with eta near the bottom of the EoS regime.
    """
    if "L_full_dense" in z.files and len(z["L_full_dense"]) >= 100:
        L = z["L_full_dense"]
    else:
        L = z["L_full"]
    L_smooth = running_mean(L, max(3, len(L) // 30))
    n = len(L_smooth); s = n // 2
    # L_init: use the raw first few steps before any smoothing window has
    # had time to mix in plateau values. (Previously took L_smooth[10],
    # which underestimated initial loss whenever the smoothing window
    # already reached into the cycle by step 10 -- common for ReLU.)
    L_init = float(np.mean(L[:max(3, min(5, n))]))
    L_tail = L_smooth[s:]; L_tail = L_tail[~np.isnan(L_tail)]
    if len(L_tail) < 5:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    L_plateau_mean = float(np.mean(L_tail))

    cv = float(np.std(L_tail) / max(L_plateau_mean, 1e-6))
    flatness = max(0.0, 1.0 - cv)
    cmin = z["cos2_min_AGOP"]
    growth = max(0.0, float(cmin[-1] - cmin[len(cmin) // 2]))
    drop_log10 = max(0.0, float(np.log10(max(L_init, 1e-12)
                                        / max(L_plateau_mean, 1e-12))))
    loss_drop_factor = min(drop_log10, 1.0)

    if "thr50_l2" in z.files:
        thr_late = z["thr50_l2"][len(z["thr50_l2"]) // 2:]
        mean_thr = float(np.mean(thr_late))
        std_thr = float(np.std(thr_late))
        closeness = float(np.exp(-abs(mean_thr - r_t)))
        stability = 1.0 / (1.0 + std_thr)
    else:
        closeness = stability = 0.0

    score = float(min(flatness, 1.0)
                  * min(growth, 1.0)
                  * loss_drop_factor
                  * closeness
                  * stability)
    return score, flatness, growth, loss_drop_factor, closeness, stability


def eff_rank_match_score(z, r_t):
    """Score: prefer configurations whose AGOP `thr@50% lambda_2` effective
    rank sits cleanly at r_t in the late half of training, AND whose loss
    is flat in the same window.

      flatness   := 1 - std(L_late)/mean(L_late)
      closeness  := exp(-|<thr50_l2>_late - r_t|)        in (0, 1]
      stability  := 1 / (1 + std(thr50_l2_late))         in (0, 1]
      score      := flatness * closeness * stability

    No alignment-growth or above-oracle term -- the user wants this score
    to specifically pick configs where the AGOP effective rank converges
    to r_t and loss is flat, regardless of how dramatic the cos2_min
    growth is. Tracks the metric in score_eff_rank.py.
    """
    if "L_full_dense" in z.files and len(z["L_full_dense"]) >= 100:
        L = z["L_full_dense"]
    else:
        L = z["L_full"]
    L_smooth = running_mean(L, max(3, len(L) // 30))
    n = len(L_smooth); s = n // 2
    L_tail = L_smooth[s:]; L_tail = L_tail[~np.isnan(L_tail)]
    if len(L_tail) < 5:
        return 0.0, 0.0, 0.0, 0.0
    cv = float(np.std(L_tail) / max(np.mean(L_tail), 1e-6))
    flatness = max(0.0, 1.0 - cv)

    if "thr50_l2" not in z.files:
        # Cannot score without the eff-rank trace; fall back to 0.
        return 0.0, flatness, 0.0, 0.0
    thr_late = z["thr50_l2"][len(z["thr50_l2"]) // 2:]
    mean_thr = float(np.mean(thr_late))
    std_thr = float(np.std(thr_late))
    closeness = float(np.exp(-abs(mean_thr - r_t)))
    stability = 1.0 / (1.0 + std_thr)
    score = float(flatness * closeness * stability)
    return score, flatness, closeness, stability


def plateau_quality_score(z):
    """Higher = cleaner plateau-with-feature-learning.

    Score = flatness  *  Delta cos2_min  *  loss_drop_factor
    where loss_drop_factor = min(log10(L_init / L_plateau), 1.0).

    The previous version multiplied by `above_oracle / L_V_opt` capped
    at 5x, which over-rewarded rising-loss configurations sitting far
    above the head-fit oracle. We drop that amplifier; the loss-drop
    factor is the correct way to require a visible Phase 1.
    """
    if "L_full_dense" in z.files and len(z["L_full_dense"]) >= 100:
        L = z["L_full_dense"]
    else:
        L = z["L_full"]
    L_smooth = running_mean(L, max(3, len(L) // 30))
    n = len(L_smooth); s = n // 2
    # L_init: use the raw first few steps before any smoothing window has
    # had time to mix in plateau values. (Previously took L_smooth[10],
    # which underestimated initial loss whenever the smoothing window
    # already reached into the cycle by step 10 -- common for ReLU.)
    L_init = float(np.mean(L[:max(3, min(5, n))]))
    L_tail = L_smooth[s:]; L_tail = L_tail[~np.isnan(L_tail)]
    if len(L_tail) < 5:
        return 0.0, 0.0, 0.0, 0.0
    L_plateau_mean = float(np.mean(L_tail))
    cv = float(np.std(L_tail) / max(L_plateau_mean, 1e-6))
    flatness = max(0.0, 1.0 - cv)

    cmin = z["cos2_min_AGOP"]
    growth = max(0.0, float(cmin[-1] - cmin[len(cmin) // 2]))

    drop_log10 = max(0.0, float(np.log10(max(L_init, 1e-12)
                                        / max(L_plateau_mean, 1e-12))))
    loss_drop_factor = min(drop_log10, 1.0)

    score = float(min(flatness, 1.0)
                  * min(growth, 1.0)
                  * loss_drop_factor)
    return score, flatness, growth, loss_drop_factor


def plot_one_column(axes_4, ckpt: Path, panel: str, p: int, r_t: int,
                    r_s: int, eta: float, score: float):
    z = np.load(ckpt)
    title = (f"{'(' + panel + ')' if panel else ''} p={p}, $r_t$={r_t}, "
             f"$r_s$={r_s}, η={eta}\nscore={score:.3f}")

    # Row 1 — five-line loss panel
    ax = axes_4[0]
    if "L_full_dense" in z.files and len(z["L_full_dense"]) >= 4:
        step_d = z["step_dense"]; L_d = z["L_full_dense"]
        ax.semilogy(step_d, L_d, color="tab:gray", lw=0.4, alpha=0.45,
                    label=r"$L_{\mathrm{train}}$ raw")
        ax.semilogy(step_d, running_min(L_d, 4), color="tab:blue", lw=0.9,
                    alpha=0.7, label=r"min")
        ax.semilogy(step_d, running_mean(L_d, 20), color="tab:orange", lw=1.7,
                    label=r"$L_{\mathrm{train}}$ mean")
    else:
        step_c = z["step"]; L_c = z["L_full"]
        ax.semilogy(step_c, L_c, color="tab:gray", lw=0.6, alpha=0.6, label="raw")
        ax.semilogy(step_c, running_mean(L_c, max(3, len(L_c) // 20)),
                    color="tab:orange", lw=1.7, label=r"$L_{\mathrm{train}}$ mean")
    if "L_test_dense" in z.files and len(z["L_test_dense"]) >= 4:
        ax.semilogy(z["step_dense"], z["L_test_dense"], color="tab:green",
                    lw=0.9, alpha=0.8, label="test")
    elif "L_test" in z.files:
        ax.semilogy(z["step"], z["L_test"], color="tab:green", lw=1.1,
                    alpha=0.85, label="test")
    if "L_V_opt_train" in z.files:
        ax.semilogy(z["step"], np.maximum(z["L_V_opt_train"], 1e-6),
                    color="tab:purple", lw=1.4, ls="--",
                    label=r"$L_{V,opt}$")
    if "L_AGOP_opt_train" in z.files:
        ax.semilogy(z["step"], np.maximum(z["L_AGOP_opt_train"], 1e-6),
                    color="tab:brown", lw=1.4, ls=":",
                    label=r"$L_{AGOP,opt}$")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("step", fontsize=9); ax.set_ylabel("loss (log)", fontsize=9)
    ax.tick_params(labelsize=8); ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="best", ncol=2)

    # Row 2 — AGOP direction-only Pearson
    ax = axes_4[1]
    step = z["step"]
    if "mean_cos2_AGOP" in z.files:
        ax.plot(step, z["mean_cos2_AGOP"], color="tab:blue", lw=1.8,
                label=r"$\frac{1}{r_t}\sum\cos^2\theta_i$")
    if "cos2_min_AGOP" in z.files:
        ax.plot(step, z["cos2_min_AGOP"], color="tab:red", lw=1.4, alpha=0.85,
                label=r"$\cos^2\theta_{\min}$")
    ax.set_ylim(-0.05, 1.05); ax.axhline(1.0, color="gray", ls=":", lw=0.5)
    ax.set_title("AGOP direction-only Pearson", fontsize=10)
    ax.set_xlabel("step", fontsize=9); ax.set_ylabel("alignment", fontsize=9)
    ax.tick_params(labelsize=8); ax.grid(alpha=0.3)

    # Row 3 — per-eigvec V-mass
    # In-spectrum (j ≤ r_t): viridis-colored solid lines, label first/last only.
    # Beyond r_t (j > r_t): red leak directions (dashed/dot/dot-dash).
    ax = axes_4[2]
    if "per_eigvec_in_V" in z.files:
        per_eigvec = z["per_eigvec_in_V"]; n_track = per_eigvec.shape[1]
        cmap = plt.colormaps["viridis"]
        for j in range(n_track):
            if j < r_t:
                color_j = cmap(j / max(r_t-1, 1)); ls = "-"; lw = 1.4
                if j == 0:
                    lab = r"$v_1$ (top eig)"
                elif j == r_t - 1:
                    lab = fr"$v_{{{r_t}}}$ (= $v_{{r_t}}$)"
                else:
                    lab = None
            else:
                color_j = "tab:red"; ls = ["--", "-.", ":"][j - r_t]; lw = 1.1
                lab = fr"$v_{{{j+1}}}$ (beyond $r_t$)"
            ax.plot(step, per_eigvec[:, j], color=color_j, lw=lw, ls=ls, label=lab)
    ax.set_ylim(-0.05, 1.05); ax.axhline(1.0, color="gray", ls=":", lw=0.5)
    ax.set_title(r"per-eigvec $V$-mass (top $r_t$+3)", fontsize=10)
    ax.set_xlabel("step", fontsize=9); ax.set_ylabel(r"$\|P_V v_j\|^2$", fontsize=9)
    ax.tick_params(labelsize=8); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best", ncol=1)

    # Row 4 — AGOP effective rank (two threshold-based diagnostics + r_t).
    ax = axes_4[3]
    if "top_lams" in z.files:
        lams = z["top_lams"]
        lam_max = np.maximum(lams.max(axis=1, keepdims=True), 1e-30)
        thr5 = (lams >= 0.05 * lam_max).sum(axis=1)
        ax.plot(step, thr5, color="tab:orange", lw=1.6,
                label=r"thr@5% $\lambda_{\max}$")
        thr50_l2 = thr_rank_vs_l2(lams, 0.5)
        ax.plot(step, thr50_l2, color="tab:purple", lw=1.6, ls="-.",
                label=r"thr@50% $\lambda_2$")
    ax.axhline(r_t, color="gray", ls=":", lw=0.9, label=f"$r_t$={r_t}")
    ax.set_ylim(bottom=0)
    ax.set_title("AGOP effective rank", fontsize=10)
    ax.set_xlabel("step", fontsize=9); ax.set_ylabel("eff rank", fontsize=9)
    ax.tick_params(labelsize=8); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default="../checkpoints/grid_14configs")
    ap.add_argument("--out", default="top6_paper.png")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--score-mode",
                    choices=["flatness_growth_drop",
                             "eff_rank_match",
                             "combined"],
                    default="combined",
                    help="Ranking criterion. flatness_growth_drop = "
                         "flatness * delta_cos2_min * loss_drop_factor "
                         "(rewards visible Phase 1 + alignment growth). "
                         "eff_rank_match = flatness * "
                         "exp(-|mean_thr_late - r_t|) * 1/(1+std) "
                         "(rewards configs whose AGOP effective rank "
                         "converges to r_t with low variance). "
                         "combined = product of all five criteria above "
                         "(default; the only configs that win are those "
                         "with both visible plateau dynamics AND clean "
                         "eff-rank-to-r_t convergence).")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).resolve()
    print(f"Reading from: {in_dir}")
    print(f"Score mode:   {args.score_mode}")
    if not in_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {in_dir}")

    # Score every checkpoint, sort, pick top-K
    rows = []
    for ckpt in sorted(in_dir.glob("*.npz")):
        try:
            z = np.load(ckpt)
            panel, p, rt, rs, eta = parse_filename(ckpt.stem)
            if any(v is None for v in (p, rt, rs, eta)): continue
            if args.score_mode == "flatness_growth_drop":
                score, a, b, c = plateau_quality_score(z)
                d = e = 0.0
            elif args.score_mode == "eff_rank_match":
                score, a, b, c = eff_rank_match_score(z, rt)
                d = e = 0.0
            else:  # combined
                score, a, b, c, d, e = combined_score(z, rt)
            rows.append((score, a, b, c, d, e, ckpt, panel, p, rt, rs, eta))
        except Exception as e:
            print(f"  [skip] {ckpt.name}: {e}")
    if not rows:
        raise RuntimeError(f"No valid checkpoints in {in_dir}")
    rows.sort(key=lambda r: -r[0])

    # Header columns depend on the score mode.
    if args.score_mode == "flatness_growth_drop":
        cols = ("flat", "Δcos²", "L_drop OOM", "", "")
    elif args.score_mode == "eff_rank_match":
        cols = ("flat", "closeness", "stability", "", "")
    else:  # combined
        cols = ("flat", "Δcos²", "L_drop", "close", "stab")
    print()
    header = f"{'rank':>5} {'panel':>6} {'config':<30} {'score':>6}"
    for c in cols:
        if c:
            header += f" {c:>10}"
    print(header)
    print("-" * len(header))
    def fmt_score(s):
        # Combined-mode scores are products of up to 5 factors in [0,1] and
        # routinely fall below 0.01; show in scientific notation when small.
        if s == 0:
            return f"{0.0:>9.3e}"
        return f"{s:>9.3e}" if abs(s) < 0.01 else f"{s:>9.4f}"

    for i, row in enumerate(rows):
        score, a, b, c, d, e = row[:6]
        ckpt, panel, p, rt, rs, eta = row[6:]
        marker = "*" if i < args.top else " "
        cfg_str = f"p={p}, r_t={rt}, r_s={rs}, η={eta}"
        line = f"{marker} {i+1:>3} ({panel:>3}) {cfg_str:<30} {fmt_score(score)}"
        vals = (a, b, c, d, e)
        for label, val in zip(cols, vals):
            if label:
                line += f" {val:>10.3f}"
        print(line)

    # Sanity check: warn if all scores are essentially zero in combined mode
    # (typically means the loss-drop factor or eff-rank closeness factor is
    # zero across the board, so nothing is differentiating the configurations).
    if args.score_mode == "combined" and rows[0][0] < 1e-6:
        print()
        print("  [warn] All combined scores are < 1e-6. The bottleneck is "
              "usually one factor across all configs. Inspect the columns "
              "above: if 'L_drop' is ~0 everywhere, the loss does not have "
              "a visible Phase 1 in this grid; if 'close' is ~0 everywhere, "
              "the AGOP eff rank does not converge to r_t. In either case, "
              "try --score-mode flatness_growth_drop or --score-mode "
              "eff_rank_match to see which criterion is the limiting one.")

    selected = rows[:args.top]
    n_cols = len(selected)
    fig, axes = plt.subplots(4, n_cols, figsize=(3.7 * n_cols, 2.8 * 4),
                             squeeze=False)
    for col, row in enumerate(selected):
        score = row[0]
        ckpt, panel, p, rt, rs, eta = row[6:]
        plot_one_column(axes[:, col], ckpt, panel, p, rt, rs, eta, score)

    fig.suptitle(
        f"Plateau-with-feature-learning configurations.\n"
        f"Each column: 1 config. Row 1: loss decomposition (raw zigzag · running min "
        f"· running mean · test · $L_{{V,opt}}$ · $L_{{AGOP,opt}}$). "
        f"Row 2: AGOP direction-only alignment. Row 3: per-eigvec $V$-mass. "
        f"Row 4: AGOP effective rank.",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=args.dpi); plt.close(fig)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
