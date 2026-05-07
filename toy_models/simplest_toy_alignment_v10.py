from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def run(lambda_a: float, lambda_b: float, eta: float,
        alpha0: float, beta0: float, n_steps: int):
    a, b = float(alpha0), float(beta0)
    out = {"step": [], "alpha": [], "beta": [], "loss": []}
    for t in range(n_steps + 1):
        out["step"].append(t)
        out["alpha"].append(a); out["beta"].append(b)
        out["loss"].append(0.5 * lambda_a * a**2 + 0.5 * lambda_b * b**2)
        gA, gB = lambda_a * a, lambda_b * b
        gnorm = (gA**2 + gB**2) ** 0.5
        if gnorm > 1e-15:
            a -= eta * gA / gnorm
            b -= eta * gB / gnorm
    return {k: np.asarray(v) for k, v in out.items()}


def detect_plateau_onset(loss: np.ndarray, window: int = 4,
                         rel_eps: float = 0.01) -> int:
    """
    Plateau onset = smallest t such that the period-2-averaged loss is
    no longer decreasing by more than `rel_eps` per `window` steps.

    We use a window-averaged loss to remove the period-2 oscillation
    (loss_avg_t = mean(loss[t-window+1 .. t])), then find the first t
    where  (loss_avg[t] - loss_avg[t+window]) / loss_avg[t] < rel_eps.

    This formalises "loss has stopped going down (apart from the
    oscillation)".
    """
    L = np.asarray(loss, dtype=float)
    n = len(L)
    if n < 3 * window:
        return 0

    # Centred window-average (cumulative-sum-of-pairs trick).
    cs = np.concatenate(([0.0], np.cumsum(L)))
    avg = (cs[window:] - cs[:-window]) / window  # avg[i] = mean(L[i..i+window-1])

    for t in range(len(avg) - window):
        denom = max(abs(avg[t]), 1e-30)
        if (avg[t] - avg[t + window]) / denom < rel_eps:
            return t  # avg index = step index of left edge of window
    return n - 1


def measured_beta_rate(beta_abs: np.ndarray, plateau_idx: int) -> float:
    seg = beta_abs[plateau_idx:]
    seg = seg[seg > 1e-30]
    if len(seg) < 5:
        return float("nan")
    t = np.arange(len(seg))
    slope, _ = np.polyfit(t, np.log(seg), 1)
    return float(np.exp(slope))


def main():
    out_dir = Path(__file__).resolve().parent
    out_png = out_dir / "simplest_toy_alignment_v10.png"
    out_csv = out_dir / "simplest_toy_alignment_v10_table.csv"

    eta = 0.25
    k = 6                              # Phase 1 length (theorem T_1 = k)
    alpha0 = (k + 0.5) * eta            # = 5.25
    beta0 = 1                      # RELAXED beyond eta/2 = 0.25;
                                        # rho_0 << 1 still holds -- see below

    configs = [
        (200.0,  1.0,  800),
        (500.0,  1.0,  1500),
        (1000.0, 1.0,  3000),
        (2000.0, 1.0,  6000),
    ]
    plateau_eps = 0.01     # 1% relative-change-per-window threshold
    plateau_window = 4     # period-2-killing window

    n_cols = len(configs)
    fig, axes = plt.subplots(3, n_cols, figsize=(5 * n_cols, 11),
                             squeeze=False)
    rows = []

    for col, (la, lb, n_steps) in enumerate(configs):
        h = run(la, lb, eta, alpha0, beta0, n_steps)
        r = lb / la
        rho_0 = lb * beta0 / (la * eta)         # the meaningful smallness param
        floor = la * eta**2 / 8.0

        denom_a = h["alpha"]**2 + h["beta"]**2
        align = h["alpha"]**2 / np.maximum(denom_a, 1e-30)

        T1_theory = k
        T_star = detect_plateau_onset(h["loss"],
                                       window=plateau_window,
                                       rel_eps=plateau_eps)

        # Predicted curves anchored at the theoretical T_1
        beta_T1 = abs(h["beta"][T1_theory])
        decay = (1 - 2 * r)
        t_arr = h["step"].astype(float)
        beta_pred = np.where(t_arr >= T1_theory,
                             beta_T1 * decay ** np.maximum(t_arr - T1_theory, 0),
                             np.nan)
        # squared-rate alignment-error prediction in the cycle
        align_err_pred = beta_pred**2 / ((eta / 2.0) ** 2 + beta_pred**2)
        align_pred = 1.0 - align_err_pred

        meas_rate = measured_beta_rate(np.abs(h["beta"]), T1_theory)

        rows.append({
            "ratio": int(la / lb),
            "rho_0": rho_0,
            "T1_theory": T1_theory,
            "T_star_data": T_star,
            "match_T1_within_2": int(abs(T_star - T1_theory) <= 2),
            "loss_floor_pred": floor,
            "loss_at_T1": float(h["loss"][T1_theory]),
            "loss_at_end": float(h["loss"][-1]),
            "beta_decay_pred": decay,
            "beta_decay_meas": meas_rate,
            "rate_ratio_meas/pred": meas_rate / decay
                                     if not np.isnan(meas_rate) else float("nan"),
            "align_at_0": float(align[0]),
            "align_at_T1": float(align[T1_theory]),
            "align_at_end": float(align[-1]),
        })

        ratio_lab = fr"$\lambda_\alpha/\lambda_\beta = {int(la/lb)}$"

        # ---- Row 1: log-loss with predicted floor + plateau markers ----
        ax = axes[0, col]
        ax.semilogy(h["step"], h["loss"], color="tab:blue", lw=1.0,
                    label=r"$L_t$ measured")
        ax.axhline(floor, color="tab:red", ls="--", lw=1.2,
                   label=fr"$L_{{\mathrm{{floor}}}}=\lambda_\alpha\eta^2/8={floor:.2f}$")
        ax.axvspan(T_star, n_steps, color="orange", alpha=0.10,
                   label=f"plateau (data: $t^* = {T_star}$)")
        ax.axvline(T_star, color="orange", lw=1.4)
        ax.axvline(T1_theory, color="black", ls=":", lw=1.0,
                   label=f"$T_1 = k = {T1_theory}$ (theory)")
        ax.set_title(f"({chr(ord('a') + col)}) {ratio_lab}, "
                     fr"$\eta={eta}$, $k={k}$, $\rho_0={rho_0:.3g}$",
                     fontsize=9)
        ax.set_xlabel("step"); ax.set_ylabel("loss (log)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="upper right")

        # ---- Row 2: theorem alignment with prediction overlay ----
        ax = axes[1, col]
        ax.plot(h["step"], align, color="tab:green", lw=2.0,
                label=r"$\mathrm{align}_t = \alpha_t^2/(\alpha_t^2+\beta_t^2)$")
        with np.errstate(invalid="ignore"):
            ax.plot(h["step"], align_pred, color="tab:red", ls=":", lw=1.5,
                    label=r"prediction (squared rate)")
        ax.axvspan(T_star, n_steps, color="orange", alpha=0.10)
        ax.axvline(T_star, color="orange", lw=1.4)
        ax.axvline(T1_theory, color="black", ls=":", lw=1.0)
        ax.text(T_star + 0.02 * n_steps, 0.07,
                fr"$\mathrm{{align}}(t^*) = {align[T_star]:.3f}$",
                fontsize=8, color="black", va="bottom")
        ax.text(n_steps * 0.98, 0.95,
                f"start: {align[0]:.3f}\n"
                f"  $T_1$ : {align[T1_theory]:.3f}\n"
                f"end: {align[-1]:.3f}",
                fontsize=8, ha="right", va="top",
                bbox=dict(facecolor="white", edgecolor="darkgreen", alpha=0.85))
        ax.set_xlabel("step"); ax.set_ylabel(r"alignment")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Phase 1: alignment dips. "
                     "Plateau (shaded): alignment climbs to 1.",
                     fontsize=9)
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="center right")

        # ---- Row 3: log|beta_t| with predicted geometric decay ----
        ax = axes[2, col]
        ax.semilogy(h["step"], np.maximum(np.abs(h["beta"]), 1e-30),
                    color="tab:purple", lw=1.5,
                    label=r"$|\beta_t|$ measured")
        ax.semilogy(h["step"], beta_pred, color="tab:red", ls=":", lw=1.5,
                    label=r"$|\beta_{T_1}|(1-2\lambda_\beta/\lambda_\alpha)^{t-T_1}$")
        ax.axvspan(T_star, n_steps, color="orange", alpha=0.10)
        ax.axvline(T_star, color="orange", lw=1.4)
        rate_str = (f"meas/pred = {meas_rate / decay:.4f}"
                    if not np.isnan(meas_rate) else "n/a")
        ax.set_title(f"$\\beta$ decay: pred $={decay:.4f}$, meas $={meas_rate:.4f}$; "
                     f"{rate_str}", fontsize=9)
        ax.set_xlabel("step"); ax.set_ylabel(r"$|\beta_t|$ (log)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "M1 (2D anisotropic quadratic, NGD): plateau-with-feature-learning. "
        r"$\alpha_0 = (k+\frac{1}{2})\eta = 5.25$, $\beta_0 = 1$ (relaxed; "
        r"$\rho_0 \ll 1$ at all four anisotropies)." "\n"
        "Top: loss drops 3 orders of magnitude in Phase 1, then locks at the "
        "predicted floor.  Plateau onset detected from the loss trace "
        r"(orange line, $t^*$) matches $T_1 = k$ from theory (black dotted)." "\n"
        "Middle: alignment dips during Phase 1, then climbs to 1 inside the "
        r"locked plateau at the predicted squared rate." "\n"
        r"Bottom: $|\beta_t|$ matches the predicted geometric decay rate.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=140); plt.close(fig)
    print(f"Saved {out_png}")

    # Diagnostics
    keys = list(rows[0].keys())
    with open(out_csv, "w") as f:
        f.write(",".join(keys) + "\n")
        for r_row in rows:
            f.write(",".join(f"{r_row[k]:.6g}" if isinstance(r_row[k], float) else str(r_row[k])
                             for k in keys) + "\n")
    print(f"Saved {out_csv}")
    print()
    hdr = ("ratio | rho_0   | T1_th  T*_data  match | floor   loss_T1  loss_end |"
           " beta_pred  beta_meas  ratio  | align_0  align_T1  align_end")
    print(hdr); print("-" * len(hdr))
    for r_row in rows:
        print(f"{r_row['ratio']:>5d} | {r_row['rho_0']:.5f} | "
              f"{r_row['T1_theory']:>5d}  {r_row['T_star_data']:>6d}  {'OK' if r_row['match_T1_within_2'] else 'NO':>5s} | "
              f"{r_row['loss_floor_pred']:>5.2f}  {r_row['loss_at_T1']:>6.2f}  {r_row['loss_at_end']:>7.2f}  | "
              f"{r_row['beta_decay_pred']:.5f}   {r_row['beta_decay_meas']:.5f}   {r_row['rate_ratio_meas/pred']:.4f} | "
              f"{r_row['align_at_0']:.4f}  {r_row['align_at_T1']:.4f}  {r_row['align_at_end']:.4f}")


if __name__ == "__main__":
    main()
