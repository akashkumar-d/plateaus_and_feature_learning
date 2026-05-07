from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os


DIR = Path("unified_v3d")


def running_min(y, w=4):
    out = np.empty_like(y, dtype=float)
    for i in range(len(y)): out[i] = float(np.min(y[max(0,i-w+1):i+1]))
    return out


def running_mean(y, w=20):
    y = np.asarray(y, dtype=float)
    out = np.full(len(y), np.nan); h = w // 2
    for i in range(len(y)):
        lo = max(0, i - h); hi = min(len(y), i + h + 1)
        out[i] = float(np.mean(y[lo:hi]))
    return out


def parse_config_name(filename):
    """Parse 'Muon_p100_rt8_rs50_eta1.5_p100_rt8_rs50_n15000_seed5.npz'."""
    parts = filename.replace(".npz", "").split("_")
    p, rt, rs, eta = None, None, None, None
    for tok in parts:
        if tok.startswith("p") and tok[1:].isdigit():
            p = int(tok[1:])
        elif tok.startswith("rt"):
            try: rt = int(tok[2:])
            except: pass
        elif tok.startswith("rs"):
            try: rs = int(tok[2:])
            except: pass
        elif tok.startswith("eta"):
            try: eta = float(tok[3:])
            except: pass
    return p, rt, rs, eta


def plateau_quality(L_train_mean, cos2_min, frac=0.5):
    """Score for plateau cleanliness.
    plateau window: last `frac` of trajectory.
    Returns: (loss_flatness ∈ [0,1], cos2_min_growth, plateau_score)."""
    n = len(L_train_mean)
    s = int(n * (1 - frac))
    L = np.asarray(L_train_mean[s:], dtype=float)
    # Loss flatness: 1 - normalized std (higher = flatter)
    L_mean = float(np.mean(L)); L_std = float(np.std(L))
    if L_mean > 1e-9:
        flatness = max(0.0, 1.0 - L_std / L_mean)
    else:
        flatness = 0.0
    cos_growth = float(cos2_min[-1] - cos2_min[s])
    score = flatness * max(0.0, cos_growth)
    return flatness, cos_growth, score


def main():
    cfgs = []
    for f in sorted(DIR.glob("Muon_*.npz")):
        z = np.load(f)
        if int(z["next_t"]) < 1000:
            continue
        p, rt, rs, eta = parse_config_name(f.name)
        if "L_AGOP_opt_train" not in z.files:
            continue
        cfgs.append({"name": f.name, "z": z, "p": p, "rt": rt, "rs": rs, "eta": eta})

    n_cfg = len(cfgs)
    print(f"Loaded {n_cfg} configs.")

    # ============ Compute plateau quality ============
    rows = []
    for c in cfgs:
        z = c["z"]
        L_mean = running_mean(z["L_full"], w=20)
        flatness, cos_growth, score = plateau_quality(L_mean, z["cos2_min_AGOP"])
        rows.append({
            **c, "flatness": flatness, "cos_growth": cos_growth, "plateau_score": score,
            "L_train_end": float(z["L_full"][-1]),
            "L_test_end": float(z["L_test"][-1]),
            "L_V_opt_end": float(z["L_V_opt_train"][-1]),
            "L_AGOP_opt_end": float(z["L_AGOP_opt_train"][-1]),
            "in_V_end": float(z["in_V_frac"][-1]),
            "cos2_min_end": float(z["cos2_min_AGOP"][-1]),
            "mean_cos2_end": float(z["mean_cos2_AGOP"][-1]),
        })

    rows.sort(key=lambda r: -r["plateau_score"])
    print()
    print("Configs ranked by plateau quality (flatness × cos²_min growth in last 50%):")
    print(f"{'config':<40} {'(p, rt, rs, η)':<22} {'flatness':>9} {'cos_grow':>9} "
          f"{'score':>7} {'L_tr':>6} {'L_te':>6} {'L_Vopt':>7} {'L_AGopt':>7} {'cos²_min':>9}")
    print("-" * 145)
    for r in rows:
        print(f"{r['name'][:38]:<40} (p={r['p']}, rt={r['rt']}, rs={r['rs']}, η={r['eta']}) "
              f"{r['flatness']:>9.3f} {r['cos_growth']:>+9.3f} {r['plateau_score']:>7.4f} "
              f"{r['L_train_end']:>6.3f} {r['L_test_end']:>6.3f} "
              f"{r['L_V_opt_end']:>7.4f} {r['L_AGOP_opt_end']:>7.4f} "
              f"{r['cos2_min_end']:>9.3f}")

    # ============ FIGURE: per-config grid (5 rows × n_cfg cols) ============
    n_cols = min(7, n_cfg); n_rows_grid = (n_cfg + n_cols - 1) // n_cols
    rows_per_block = 4  # loss / cos2 / per-eigvec / eff rank

    fig, axes = plt.subplots(rows_per_block * n_rows_grid, n_cols,
                             figsize=(3.5*n_cols, 3.0*rows_per_block*n_rows_grid),
                             squeeze=False)
    for idx, c in enumerate(cfgs):
        z = c["z"]; step = z["step"]
        col = idx % n_cols; block = idx // n_cols
        cfg = f"(p={c['p']}, r_t={c['rt']}, r_s={c['rs']}, η={c['eta']})"

        # Loss row
        ax = axes[block * rows_per_block, col]
        L = z["L_full"]
        ax.semilogy(step, L, color="tab:gray", lw=0.4, alpha=0.4)
        ax.semilogy(step, running_min(L, w=4),  color="tab:blue", lw=1.3, label="train (min)")
        ax.semilogy(step, z["L_test"], color="tab:green", lw=1.2, alpha=0.85, label="test")
        ax.semilogy(step, running_mean(L, w=20), color="tab:orange", lw=1.7, label="train (mean)")
        ax.semilogy(step, np.maximum(z["L_V_opt_train"], 1e-6), color="tab:purple", lw=1.4, ls="--",
                    label=r"$L_{V,opt}$ (oracle)")
        ax.semilogy(step, np.maximum(z["L_AGOP_opt_train"], 1e-6), color="tab:brown", lw=1.4, ls=":",
                    label=r"$L_{AGOP,opt}$ (practical)")
        ax.set_xlabel("step"); ax.set_ylabel("loss (log)")
        ax.set_title(f"({chr(ord('a')+idx)}) {c['name'][:30]}\n{cfg}", fontsize=8)
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=5)

        # cos² row
        ax = axes[block * rows_per_block + 1, col]
        ax.plot(step, z["cos2_min_AGOP"], color="tab:red", lw=1.4, alpha=0.7,
                label=r"$\cos^2\theta_{\min}$")
        ax.plot(step, z["mean_cos2_AGOP"], color="tab:blue", lw=1.6,
                label=r"$\frac{1}{r_t}\Sigma\cos^2$")
        ax.set_ylim(-0.05, 1.05); ax.axhline(1.0, color="gray", ls=":", lw=0.5)
        ax.set_xlabel("step"); ax.set_ylabel("alignment")
        ax.set_title(f"AGOP direction-only Pearson", fontsize=8)
        ax.legend(fontsize=6); ax.grid(alpha=0.3)

        # per-eigvec row
        ax = axes[block * rows_per_block + 2, col]
        per_eigvec = z["per_eigvec_in_V"]; r_t = c["rt"]
        n_track = per_eigvec.shape[1]
        cmap = plt.colormaps["viridis"]
        for j in range(n_track):
            if j < r_t:
                color_j = cmap(j / max(r_t-1, 1)); ls="-"; lw=1.3
            else:
                color_j = "tab:red"; ls = ["--", "-.", ":"][j - r_t]; lw = 1.0
            ax.plot(step, per_eigvec[:, j], color=color_j, lw=lw, ls=ls)
        ax.set_ylim(-0.05, 1.05); ax.axhline(1.0, color="gray", ls=":", lw=0.5)
        ax.set_xlabel("step"); ax.set_ylabel(r"$\|P_V v_j\|^2$")
        ax.set_title(f"per-eigvec V-mass (top r_t+3)", fontsize=8)
        ax.grid(alpha=0.3)

        # r̃ + thr ranks row
        ax = axes[block * rows_per_block + 3, col]
        ax.plot(step, z["r_tilde_AGOP"], color="tab:purple", lw=1.5, label="r̃ (used by L_AGOP)")
        ax.plot(step, z["in_V_frac"], color="tab:olive", lw=1.4, ls="--", label="in_V (Frob)")
        ax.axhline(r_t, color="gray", ls=":", lw=0.7, label=f"r_t={r_t}")
        ax.set_xlabel("step"); ax.set_ylabel("rank / fraction")
        ax.set_title(f"AGOP-rank r̃ + in_V", fontsize=8)
        ax.legend(fontsize=6); ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "Plateau-search grid: 14 Muon configs at 10k steps. Each col = 1 config.\n"
        "Row 1: loss decomposition (orange/green = train/test, purple/brown = oracle/practical optimal). Row 2: AGOP direction-Pearson. Row 3: per-eigvec V-mass. Row 4: r̃ + in_V.",
        fontsize=12,
    )
    fig.tight_layout(); fig.savefig("plateau_search_grid.png", dpi=110); plt.close(fig)
    print("Saved plateau_search_grid.png")

    # ============ FIGURE: top-3 plateau cleanest, focused (4 rows) ============
    top3 = rows[:3]
    fig, axes = plt.subplots(4, 3, figsize=(18, 18))
    for idx, r in enumerate(top3):
        z = r["z"]; step = z["step"]; r_t = r["rt"]
        cfg = f"(p={r['p']}, r_t={r_t}, r_s={r['rs']}, η={r['eta']})"
        plateau_start = int(len(step) * 0.5)

        # Row 1: loss decomposition.
        ax = axes[0, idx]
        L = z["L_full"]
        ax.semilogy(step, L, color="tab:gray", lw=0.4, alpha=0.4)
        ax.semilogy(step, running_min(L, w=4),  color="tab:blue", lw=1.4, label="train (min)")
        ax.semilogy(step, z["L_test"], color="tab:green", lw=1.4, alpha=0.85, label="test")
        ax.semilogy(step, running_mean(L, w=20), color="tab:orange", lw=2.0,
                    label="train (mean)")
        ax.semilogy(step, np.maximum(z["L_V_opt_train"], 1e-6),
                    color="tab:purple", lw=1.6, ls="--", label=r"$L_{V,opt}$ (oracle)")
        ax.semilogy(step, np.maximum(z["L_AGOP_opt_train"], 1e-6),
                    color="tab:brown", lw=1.6, ls=":", label=r"$L_{AGOP,opt}$")
        ax.axvspan(step[plateau_start], step[-1], color="yellow", alpha=0.15,
                   label="plateau window")
        ax.set_xlabel("step"); ax.set_ylabel("loss (log)")
        ax.set_title(f"#{idx+1}: {cfg}\n"
                     fr"flatness={r['flatness']:.3f}, "
                     fr"$\Delta\cos^2_{{\min}}$=+{r['cos_growth']:.3f}",
                     fontsize=10)
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

        # Row 2: AGOP direction-only Pearson alignment.
        ax = axes[1, idx]
        ax.plot(step, z["cos2_min_AGOP"], color="tab:red", lw=1.6,
                label=r"$\cos^2_{\min}$")
        ax.plot(step, z["mean_cos2_AGOP"], color="tab:blue", lw=2.0,
                label=r"$\overline{\cos^2}$")
        ax.plot(step, z["in_V_frac"], color="tab:olive", lw=1.4, ls="--",
                label=r"$A_{\mathrm{sub}}$")
        ax.axvspan(step[plateau_start], step[-1], color="yellow", alpha=0.15)
        ax.set_ylim(-0.05, 1.05); ax.axhline(1.0, color="gray", ls=":", lw=0.5)
        ax.set_xlabel("step"); ax.set_ylabel("alignment")
        ax.set_title("AGOP direction-only Pearson", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # Row 3: per-eigvec V-mass (top r_t in viridis, leak r_t+1..r_t+3 in red).
        ax = axes[2, idx]
        per_eigvec = z["per_eigvec_in_V"]
        n_track = per_eigvec.shape[1]
        cmap = plt.colormaps["viridis"]
        for j in range(n_track):
            if j < r_t:
                color_j = cmap(j / max(r_t - 1, 1)); ls = "-"; lw = 1.2
            else:
                color_j = "tab:red"
                ls = ["--", "-.", ":"][min(j - r_t, 2)]; lw = 1.0
            ax.plot(step, per_eigvec[:, j], color=color_j, lw=lw, ls=ls)
        ax.plot([], [], color=cmap(0.0), lw=1.2,
                label=fr"top $r_t={r_t}$ AGOP eigvecs")
        ax.plot([], [], color="tab:red", ls="--", lw=1.0,
                label=fr"$j > r_t$ (leak)")
        ax.axvspan(step[plateau_start], step[-1], color="yellow", alpha=0.15)
        ax.set_ylim(-0.05, 1.05); ax.axhline(1.0, color="gray", ls=":", lw=0.5)
        ax.set_xlabel("step"); ax.set_ylabel(r"$\|P_V v_j\|^2$")
        ax.set_title(r"per-eigvec $V$-mass (top $r_t+3$)", fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=7)

        # Row 4: AGOP effective rank.
        ax = axes[3, idx]
        lams = z["top_lams"] if "top_lams" in z.files else None
        if lams is not None:
            lam_max = np.maximum(lams.max(axis=1, keepdims=True), 1e-30)
            thr5 = (lams >= 0.05 * lam_max).sum(axis=1)
            sorted_lams = np.sort(lams, axis=1)[:, ::-1]
            l2 = np.maximum(sorted_lams[:, 1:2], 1e-30)
            thr50 = (lams >= 0.5 * l2).sum(axis=1)
            ax.plot(step, thr5, color="tab:orange", lw=1.6,
                    label=r"thr@5\% $\lambda_{\max}$")
            ax.plot(step, thr50, color="tab:purple", lw=1.6, ls="-.",
                    label=r"thr@50\% $\lambda_2$")
        elif "r_tilde_AGOP" in z.files:
            ax.plot(step, z["r_tilde_AGOP"], color="tab:purple", lw=1.6,
                    label=r"$\tilde{r}$ (AGOP eff rank)")
        ax.axhline(r_t, color="gray", ls=":", lw=0.9, label=fr"$r_t={r_t}$")
        ax.axvspan(step[plateau_start], step[-1], color="yellow", alpha=0.15)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("step"); ax.set_ylabel("effective rank")
        ax.set_title("AGOP effective rank", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(
        "Top-3 cleanest plateau-with-feature-learning configurations "
        "(ranked by flatness $\\times$ $\\cos^2_{\\min}$-growth).",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("plateau_top3.png", dpi=140); plt.close(fig)
    print("Saved plateau_top3.png")


if __name__ == "__main__":
    main()
