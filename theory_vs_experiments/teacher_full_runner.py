import argparse, math, time
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent))
from m3_runner import make_problem, init_W
from teacher_activations import ACTIVATIONS


def relu(z): return np.maximum(z, 0.0)
def relu_p(z): return (z > 0).astype(z.dtype)


def loss_only(W, X, y, a_s, chunk=4096):
    n = X.shape[0]; total = 0.0
    for i in range(0, n, chunk):
        Xi = X[i:i+chunk]; yi = y[i:i+chunk]
        z = Xi @ W; h = relu(z); f = h @ a_s; err = f - yi
        total += float(np.sum(err**2))
    return total/n


def loss_and_grad(W, X, y, a_s, chunk=4096):
    n = X.shape[0]; total = 0.0; gW = np.zeros_like(W)
    for i in range(0, n, chunk):
        Xi = X[i:i+chunk]; yi = y[i:i+chunk]
        z = Xi @ W; h = relu(z); f = h @ a_s; err = f - yi
        total += float(np.sum(err**2))
        dphi = relu_p(z); tmp = err[:,None]*dphi*a_s[None,:]
        gW += Xi.T @ tmp
    return total/n, (2.0/n)*gW


def teacher_C(X, U, a_t, sigma_p):
    z = X @ U; dphi = sigma_p(z); S = dphi * a_t[None, :]
    return (S.T @ S) / X.shape[0]


def student_Cs(X, W, a_s):
    z = X @ W; dphi = relu_p(z); Ss = dphi * a_s[None, :]
    return (Ss.T @ Ss) / X.shape[0]


def matrix_pearson(A, B, eps=1e-12):
    a = A.ravel().astype(float); b = B.ravel().astype(float)
    a -= a.mean(); b -= b.mean()
    d = (a@a)**0.5 * (b@b)**0.5
    return float((a@b)/d) if d > eps else 0.0


def thr_rank_vs_l2(lam, thr_frac=0.5):
    s = np.sort(np.asarray(lam))[::-1]
    if len(s) < 2: return 1
    return int(np.sum(np.asarray(lam) >= thr_frac * max(s[1], 1e-30)))


def alignment_metrics(W, X, U_t, a_s, P_V, n_extra=3, n_top_lams=12):
    p, r_s = W.shape; r_t = U_t.shape[1]
    PVW = U_t.T @ W
    in_V = float(np.sum(PVW*PVW)) / float(np.sum(W*W))
    Cs = student_Cs(X, W, a_s)
    G_s = W @ Cs @ W.T
    G_s = 0.5 * (G_s + G_s.T)
    lam, V_eig = np.linalg.eigh(G_s)
    lam = np.clip(lam[::-1], 0, None); V_eig = V_eig[:, ::-1]
    s_lam = float(lam.sum()); s_lam2 = float((lam**2).sum())
    pr_eff_rank = (s_lam * s_lam) / max(s_lam2, 1e-30)
    V_top = V_eig[:, :r_t]
    QA, _ = np.linalg.qr(V_top); QB, _ = np.linalg.qr(U_t)
    cos_per = np.clip(np.linalg.svd(QA.T @ QB, compute_uv=False), 0.0, 1.0)
    n_track = r_t + n_extra
    in_V_per_full = np.einsum("ji,jk,ki->i", V_eig, P_V, V_eig)
    per_eigvec_in_V = in_V_per_full[:n_track]
    mass_in_V = float(np.sum(lam * in_V_per_full) / max(s_lam, 1e-30))
    K = min(n_top_lams, len(lam))
    top_lams_padded = np.zeros(n_top_lams, dtype=float)
    top_lams_padded[:K] = lam[:K]
    return {
        "in_V_frac": in_V,
        "cos_per": cos_per,
        "mean_cos2_AGOP": float((cos_per**2).mean()),
        "cos2_min_AGOP": float(cos_per.min()**2),
        "per_eigvec_in_V": per_eigvec_in_V,
        "top_lams": top_lams_padded,
        "mass_in_V_AGOP_p": mass_in_V,
        "thr50_l2": thr_rank_vs_l2(lam, 0.5),
        "pr_eff_rank": pr_eff_rank,
    }


def loss_V_optimal(W, X_tr, y_tr, X_te, y_te, U_t, p, chunk=4096):
    P_V = U_t @ U_t.T; W_V = P_V @ W
    r_s = W_V.shape[1]; sqp = math.sqrt(p)
    def feats(X):
        n = X.shape[0]; H = np.zeros((n, r_s))
        for i in range(0, n, chunk):
            Xi = X[i:i+chunk]; z = Xi @ W_V; H[i:i+chunk] = relu(z)/sqp
        return H
    M_tr = feats(X_tr)
    a_opt, _, _, _ = np.linalg.lstsq(M_tr, y_tr, rcond=None)
    L_tr = float(np.mean((M_tr @ a_opt - y_tr)**2))
    M_te = feats(X_te)
    L_te = float(np.mean((M_te @ a_opt - y_te)**2))
    return L_tr, L_te


def loss_AGOP_optimal(W, X_tr, y_tr, X_te, y_te, p, a_s, r_tilde, chunk=4096):
    Cs = student_Cs(X_tr, W, a_s)
    G_s = W @ Cs @ W.T; G_s = 0.5 * (G_s + G_s.T)
    lam, V_eig = np.linalg.eigh(G_s); V_eig = V_eig[:, ::-1]
    r_tilde = max(1, min(int(r_tilde), V_eig.shape[1]))
    V_top = V_eig[:, :r_tilde]
    P = V_top @ V_top.T
    W_hat = P @ W
    r_s = W_hat.shape[1]; sqp = math.sqrt(p)
    def feats(X):
        n = X.shape[0]; H = np.zeros((n, r_s))
        for i in range(0, n, chunk):
            Xi = X[i:i+chunk]; z = Xi @ W_hat; H[i:i+chunk] = relu(z)/sqp
        return H
    M_tr = feats(X_tr)
    a_opt, _, _, _ = np.linalg.lstsq(M_tr, y_tr, rcond=None)
    L_tr = float(np.mean((M_tr @ a_opt - y_tr)**2))
    M_te = feats(X_te)
    L_te = float(np.mean((M_te @ a_opt - y_te)**2))
    return L_tr, L_te


KEYS_SCALAR = ("step", "L_full", "L_test", "L_V_opt_train", "L_V_opt_test",
               "L_AGOP_opt_train", "L_AGOP_opt_test",
               "in_V_frac", "mean_cos2_AGOP", "cos2_min_AGOP",
               "mass_in_V_AGOP_p", "thr50_l2", "pr_eff_rank")

# Dense per-step keys: small, fast, recorded every step regardless of
# log_every. Used for the period-2 cycle signature
# rho_2 := -Corr(Delta L_t, Delta L_{t+1}) and the per-step Frobenius
# displacement check ||W_{t+1} - W_t||_F = eta * ||update||_F = eta * sqrt(M).
KEYS_DENSE = ("step_dense", "L_full_dense", "disp_dense")


def run_one(teacher_act, p, r_t, r_s, n, eta, seed, n_steps, log_every,
            ckpt_path, chunk_steps, budget_s):
    X, y, X_te, y_te, U_t, a_t = make_problem(p, r_t, n, n_test=5000,
                                              seed=seed, teacher_act=teacher_act)
    a_s = (1.0/math.sqrt(p)) * np.ones(r_s)
    P_V = U_t @ U_t.T

    if ckpt_path.exists():
        z = np.load(ckpt_path, allow_pickle=True)
        if int(z["next_t"]) >= n_steps:
            print(f"[skip] {ckpt_path.name} at {int(z['next_t'])}")
            return False
        W = z["W"].copy()
        log = {k: list(z[k]) for k in KEYS_SCALAR}
        cos_per_log = list(z["cos_per"])
        per_eigvec_log = list(z["per_eigvec_in_V"])
        top_lams_log = list(z["top_lams"])
        # Dense logs: backward-compatible -- if absent in old checkpoints,
        # start fresh empty lists (those checkpoints just won't have the
        # period-2 / displacement panels populated until resumed).
        dense = {k: list(z[k]) if k in z.files else []
                 for k in KEYS_DENSE}
        start_t = int(z["next_t"])
        print(f"[resume] {ckpt_path.name}: step {start_t}")
    else:
        rng = np.random.default_rng(seed + 17)
        W0, ia = init_W(p, r_s, U_t, rng=rng)
        print(f"[init {teacher_act} eta={eta}] {ckpt_path.name}: in-V={ia:.3f}")
        W = W0.copy()
        log = {k: [] for k in KEYS_SCALAR}
        cos_per_log = []; per_eigvec_log = []; top_lams_log = []
        dense = {k: [] for k in KEYS_DENSE}
        start_t = 0

    end_t = min(start_t + chunk_steps, n_steps)
    t_start = time.time()
    for t in range(start_t, end_t + 1):
        if time.time() - t_start > budget_s:
            print(f"[budget] stop at {t}"); end_t = t; break
        L, gW = loss_and_grad(W, X, y, a_s)
        Ug, _, Vhg = np.linalg.svd(gW, full_matrices=False)
        update = Ug @ Vhg
        # Dense per-step record: cheap (norm + scalar) regardless of log_every.
        # disp_dense[i] = ||W_{t+1} - W_t||_F = eta * ||Polar(grad)||_F = eta * sqrt(M).
        if t < end_t:
            dense["step_dense"].append(t)
            dense["L_full_dense"].append(L)
            dense["disp_dense"].append(float(eta * np.linalg.norm(update)))
        if t % log_every == 0 or t == end_t:
            am = alignment_metrics(W, X, U_t, a_s, P_V)
            L_te = loss_only(W, X_te, y_te, a_s)
            L_V_tr, L_V_te = loss_V_optimal(W, X, y, X_te, y_te, U_t, p)
            r_tilde = am["thr50_l2"]
            L_AG_tr, L_AG_te = loss_AGOP_optimal(W, X, y, X_te, y_te, p, a_s, r_tilde)
            log["step"].append(t); log["L_full"].append(L); log["L_test"].append(L_te)
            log["L_V_opt_train"].append(L_V_tr); log["L_V_opt_test"].append(L_V_te)
            log["L_AGOP_opt_train"].append(L_AG_tr); log["L_AGOP_opt_test"].append(L_AG_te)
            for k in ("in_V_frac","mean_cos2_AGOP","cos2_min_AGOP",
                      "mass_in_V_AGOP_p","thr50_l2","pr_eff_rank"):
                log[k].append(am[k])
            cos_per_log.append(am["cos_per"])
            per_eigvec_log.append(am["per_eigvec_in_V"])
            top_lams_log.append(am["top_lams"])
        if t == end_t: break
        W = W - eta * update

    save = {k: np.asarray(log[k]) for k in KEYS_SCALAR}
    save["W"] = W; save["next_t"] = end_t
    save["cos_per"] = np.stack(cos_per_log, axis=0)
    save["per_eigvec_in_V"] = np.stack(per_eigvec_log, axis=0)
    save["top_lams"] = np.stack(top_lams_log, axis=0)
    for k in KEYS_DENSE:
        save[k] = np.asarray(dense[k])
    np.savez(ckpt_path, **save)
    print(f"[done] {teacher_act} eta={eta} step={end_t} L={log['L_full'][-1]:.3f} "
          f"L_te={log['L_test'][-1]:.3f} L_Vopt={log['L_V_opt_train'][-1]:.4f} "
          f"in_V={log['in_V_frac'][-1]:.3f} mean_cos²={log['mean_cos2_AGOP'][-1]:.3f}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="teacher_full")
    ap.add_argument("--budget-s", type=float, default=38.0)
    ap.add_argument("--chunk-steps", type=int, default=2000)
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    seed = 5; p = 100; r_t = 4; r_s = 50; n = 15000; n_steps = 6000

    # Search for EoS regime: loss should plateau ABOVE L_V_opt for visible
    # plateau-with-feature-learning phenomenon
    configs = [
        # GELU sweep around EoS (L_V_opt = 0.06, want plateau at 0.3-1.5)
        ("gelu",       0.5),
        ("gelu",       0.7),
        ("gelu",       0.9),
        # SiLU (L_V_opt = 0.11)
        ("silu",       0.7),
        ("silu",       0.85),
        ("silu",       1.0),
        # tanh (L_V_opt = 0.85)
        ("tanh",       0.4),
        ("tanh",       0.5),
        ("tanh",       0.7),
    ]

    t_start = time.time()
    for ta, eta in configs:
        if time.time() - t_start > args.budget_s:
            print("[budget global] stop"); break
        ckpt = out_dir / f"teacher_{ta}_eta{eta}_p{p}_rt{r_t}_rs{r_s}_n{n}_seed{seed}.npz"
        run_one(ta, p, r_t, r_s, n, eta, seed, n_steps, log_every=20,
                ckpt_path=ckpt, chunk_steps=args.chunk_steps,
                budget_s=max(5.0, args.budget_s - (time.time() - t_start)))


if __name__ == "__main__":
    main()
