from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


LAMBDAS = np.array([1.0, 0.30, 0.10, 0.05, 0.02, 0.01, 0.005, 0.001])
N_STEPS = 4000

# (eta, k, c_off) -- one per panel
CONFIGS = [
    (0.10, 50, 0.70),
    (0.20,  8, 0.94),
    (0.50,  2, 0.94),
]


def muon_step(w, A, eta):
    g = A @ w
    return w - eta * g / max(np.linalg.norm(g), 1e-30)


def run(A, w0, eta, n_steps):
    w = w0.copy(); p = len(w)
    log = {"step": np.arange(n_steps + 1),
           "y":    np.zeros((n_steps + 1, p)),
           "loss": np.zeros(n_steps + 1)}
    for t in range(n_steps + 1):
        log["y"][t] = w
        log["loss"][t] = 0.5 * w @ (A @ w)
        if t == n_steps:
            break
        w = muon_step(w, A, eta)
    return log


def main():
    out_dir = Path(__file__).resolve().parent
    out_png = out_dir / "single_neuron_headline_v12.png"
    out_csv = out_dir / "single_neuron_headline_v12_table.csv"

    A = np.diag(LAMBDAS)
    p = len(LAMBDAS)
    sumlam2 = float(np.sum(LAMBDAS[1:] ** 2))

    n_cols = len(CONFIGS)
    fig, axes = plt.subplots(3, n_cols, figsize=(15, 11))
    rows = []

    for col, (eta, k, c_off) in enumerate(CONFIGS):
        y01 = (k + 0.5) * eta
        y0 = np.full(p, c_off * eta); y0[0] = y01

        log = run(A, y0, eta, N_STEPS)
        step = log["step"]; y = log["y"]; loss = log["loss"]
        floor = LAMBDAS[0] * eta * eta / 8.0
        T1 = k
        decay_pred = 1.0 - 2.0 * LAMBDAS / LAMBDAS[0]
        rho_0_sq = c_off * c_off * sumlam2 / (LAMBDAS[0] ** 2)
        rate_p = decay_pred[-1]                       # slowest off-mode rate

        # ---- NEW: leading-order loss prediction (Theorem M2(ii)) ----
        soft_tail = 0.5 * (LAMBDAS[1:] * y[:, 1:] ** 2).sum(axis=1)
        loss_pred = floor + soft_tail

        # ---- NEW: squared-rate alignment prediction (Theorem M2(iv)) ----
        # In the cycle alpha_t = +-eta/2, so 1 - align_t = ||y_>1||^2 / ((eta/2)^2 + ||y_>1||^2)
        # The slowest off-mode dominates ||y_>1||^2 asymptotically:
        # ||y_>1||^2 ~ ||y_>1(T_1)||^2 * (1 - 2 lambda_p/lambda_1)^{2(t-T_1)}
        # Anchor the prediction at T_1.
        anchor = T1
        norm_off2_anchor = (y[anchor, 1:] ** 2).sum()
        t_arr = step.astype(float)
        norm_off2_pred = norm_off2_anchor * rate_p ** (
            2.0 * np.maximum(t_arr - anchor, 0))
        align_err_pred = norm_off2_pred / ((eta / 2.0) ** 2 + norm_off2_pred)
        align_pred = np.where(t_arr >= anchor, 1.0 - align_err_pred, np.nan)

        win = max(20, N_STEPS // 100)
        smooth = np.convolve(loss, np.ones(win) / win, mode="same")

        align = y[:, 0] ** 2 / np.maximum((y ** 2).sum(axis=1), 1e-30)
        align_0 = float(align[0])

        late_start = max(T1 + 50, N_STEPS // 4)
        ys_p = np.abs(y[late_start:, -1])
        ys_p = ys_p[ys_p > 1e-25]
        if len(ys_p) > 5:
            slope = np.polyfit(np.arange(len(ys_p)), np.log(ys_p), 1)[0]
            rate_meas = float(np.exp(slope))
        else:
            rate_meas = float("nan")

        late = N_STEPS // 2
        rows.append({
            "eta": eta, "k": k, "c_off": c_off,
            "y01": round(y01, 2),
            "rho_0_sq": round(float(rho_0_sq), 4),
            "T1": T1,
            "floor_pred": round(floor, 6),
            "L_late_measured": round(float(loss[late:].mean()), 6),
            "ratio": round(float(loss[late:].mean()) / floor, 4),
            "slowest_rate_pred": round(rate_p, 6),
            "slowest_rate_measured": round(rate_meas, 6) if not np.isnan(rate_meas) else float("nan"),
            "align_at_0": round(align_0, 3),
            "align_at_T1": round(float(align[T1]), 3),
            "align_at_end": round(float(align[-1]), 3),
        })

        # ============ Row 1: log-loss + leading-order prediction + floor ============
        ax = axes[0, col]
        ax.semilogy(step, loss, color="tab:blue", lw=0.6, alpha=0.4,
                    label="raw $L_t$")
        ax.semilogy(step, smooth, color="tab:blue", lw=2.0,
                    label=f"MA(w={win})")
        # NEW: leading-order prediction = floor + soft tail
        ax.semilogy(step, loss_pred, color="tab:red", lw=1.8, alpha=0.9,
                    label="theory")
        ax.axhline(floor, color="tab:red", ls="--", lw=1.0,
                   label=fr"floor $= {floor:.4g}$")
        ax.set_xlabel("step"); ax.set_ylabel("loss (log)")
        ax.set_title(f"({chr(ord('a') + col)}) $\\eta = {eta:.2f}$,  "
                     f"$k = {k}$,  $\\rho_0^2 = {rho_0_sq:.3f}$",
                     fontsize=12)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.3, which="both")

        # ============ Row 2: alignment + squared-rate prediction ============
        ax = axes[1, col]
        ax.plot(step, align, color="tab:purple", lw=2,
                label=r"$y_1^2 / \|y\|^2$ (measured)")
        # NEW: squared-rate prediction
        ax.plot(step, align_pred, color="tab:red", ls=":", lw=1.8,
                label="theory")
        ax.axhline(1.0, color="gray", ls=":", lw=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("step"); ax.set_ylabel("alignment")
        ax.set_title(f"$\\mathrm{{align}}(0) = {align_0:.2f}$  $\\to 1$",
                     fontsize=11)
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(alpha=0.3)

        # ============ Row 3: per-mode |y_k| (unchanged from v11) ============
        ax = axes[2, col]
        cmap = plt.get_cmap("viridis")
        for kk in range(p):
            color = cmap(kk / max(p - 1, 1))
            ax.semilogy(step, np.abs(y[:, kk]) + 1e-30, color=color, lw=1.5,
                        label=fr"$|y_{kk+1}|, \lambda={LAMBDAS[kk]:.3g}$")
            if kk >= 1 and decay_pred[kk] > 0 and T1 < len(step):
                steps_pred = step[T1:]
                y_pred = abs(y[T1, kk]) * decay_pred[kk] ** (steps_pred - T1)
                ax.semilogy(steps_pred, y_pred + 1e-30, color=color,
                            ls=":", lw=0.9, alpha=0.7)
        ax.set_xlabel("step"); ax.set_ylabel(r"$|y_k|$ (log)")
        ax.set_title(r"solid empirical, dotted = $|y_{T_1, k}|(1-2\lambda_k/\lambda_1)^{t-T_1}$",
                     fontsize=10)
        ax.legend(fontsize=8, loc="upper right", ncol=2)
        ax.grid(alpha=0.3, which="both")
        ax.set_ylim(1e-30, 10)

    fig.suptitle(
        "M2 single-neuron Muon (literal theorem verification): "
        "scan of starting alignment.\n"
        r"All panels: $y_{0,1} = (k+\frac{1}{2})\eta$ (so $T_1 = k$ exactly), "
        r"$y_{0,k\geq 2} = c\,\eta$.  $\rho_0^2 \leq 0.09$ everywhere.  "
        r"Panel (c) demonstrates $\mathrm{align}(0) \approx 0.5$.",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=140); plt.close(fig)
    print(f"Saved {out_png}")

    keys = list(rows[0].keys())
    with open(out_csv, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]}" for k in keys) + "\n")
    print(f"Saved {out_csv}")
    print()
    print(f"{'eta':>5} {'k':>3} {'c':>5} {'y01':>5} | {'rho_0^2':>8} | "
          f"{'T1':>3} | {'floor':>9} {'L_late':>9} {'ratio':>6} | "
          f"{'rate_pred':>9} {'rate_meas':>9} | "
          f"{'align(0)':>9} {'align(T1)':>9} {'align_end':>9}")
    print("-" * 145)
    for r in rows:
        print(f"{r['eta']:>5.2f} {r['k']:>3d} {r['c_off']:>5.2f} {r['y01']:>5.2f} | "
              f"{r['rho_0_sq']:>8.4f} | {r['T1']:>3d} | "
              f"{r['floor_pred']:>9.6f} {r['L_late_measured']:>9.6f} {r['ratio']:>6.3f} | "
              f"{r['slowest_rate_pred']:>9.6f} {r['slowest_rate_measured']:>9.6f} | "
              f"{r['align_at_0']:>9.3f} {r['align_at_T1']:>9.3f} {r['align_at_end']:>9.3f}")


if __name__ == "__main__":
    main()
