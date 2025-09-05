# coded by ChatGPT

import numpy as np
import torch
import torch.nn.functional as F
import cvxpy as cp
from eval_utils import STATBuilder, _to_inputs
from tqdm import tqdm

EPS = 1e-12

#TODO: Behavior of input when label is/is not included in a batch input
# Convergence criteria should be considered


@torch.no_grad()
def _trueclass_probs(models, inputs, labels, device="cuda"):
    B = labels.size(0)
    cols = []
    for m in models:
        m.to(device)  # move current model to GPU
        out = m(**inputs)  # safe: model and inputs both on GPU
        m.to("cpu")  # free GPU
        torch.cuda.empty_cache()
        #out = m(**inputs)
        p = F.softmax(out.logits, dim=-1)  # [B, C]
        cols.append(p[torch.arange(B, device=p.device), labels.to(p.device)])
    return torch.stack(cols, dim=1)  # [B, K]



class DCBuilder:
    r"""
    Builds per-domain matrices:
      A_k (N_k x K): rows a_i = \tilde Q(\cdot|x_i) ⊙ h(\cdot; x_i, y_i)
      B_k (N_k x K): rows b_i = \tilde Q(\cdot|x_i)
    where \tilde Q comes from beta_func(batch).

    Also provides:
      u_k(z) as cvxpy expressions,
      v_k(z_t) and grad v_k(z_t) in closed form.
    """

    def __init__(self, base_models, beta_func, device="cuda"):
        self.base_models = base_models
        self.beta_func = beta_func
        self.device = device
        self.A_by_dom = []
        self.B_by_dom = []
        for model in self.base_models:
            model.eval().to(device)

    @torch.no_grad()
    def precompute_AB(self, domain_loaders, domains, max_batches=None):
        self.A_by_dom.clear()
        self.B_by_dom.clear()

        for i, loader in enumerate(domain_loaders):
            a_rows, b_rows = [], []
            # Create a progress bar for this domain
            progress_bar = tqdm(
                enumerate(loader),
                total=len(loader) if max_batches is None else min(max_batches, len(loader)),
                desc=f"Evaluating domain {domains[i]}",
                leave=True  # Keep the progress bar after completion
            )
            for bi, batch in progress_bar:
                if max_batches is not None and bi >= max_batches:
                    break

                labels = batch["labels"].to(self.device)
                inputs = _to_inputs(batch, self.device)

                q_tilde = self.beta_func(inputs)          # [B,K]
                h_true  = _trueclass_probs(self.base_models, inputs, labels)  # [B,K]

                A = (q_tilde * h_true).clamp_min(EPS).cpu().double()      # [B,K]
                B = q_tilde.clamp_min(EPS).cpu().double()                 # [B,K]

                a_rows.append(A); b_rows.append(B)

            if len(a_rows) == 0:  # empty domain (shouldn't happen)
                self.A_by_dom.append(np.full((1, len(self.base_models)), EPS))
                self.B_by_dom.append(np.full((1, len(self.base_models)), EPS))
            else:
                self.A_by_dom.append(torch.cat(a_rows, 0).numpy())
                self.B_by_dom.append(torch.cat(b_rows, 0).numpy())

    # ------- helpers at a given z (numpy arrays) -------

    def _E_log_j(self, A, z): return np.mean(np.log(np.clip(A @ z, EPS, None)))
    def _E_log_k(self, B, z): return np.mean(np.log(np.clip(B @ z, EPS, None)))

    def _per_domain_E_log_k_over_j(self, z):
        vals = []
        for A, B in zip(self.A_by_dom, self.B_by_dom):
            j = A @ z; k = B @ z
            vals.append(np.mean(np.log(np.clip(k, EPS, None)) - np.log(np.clip(j, EPS, None))))
        return np.array(vals)  # [K]

    def _E_b_over_k_rows(self, z):
        rows = []
        for B in self.B_by_dom:
            k = B @ z
            rows.append((B / np.clip(k[:, None], EPS, None)).mean(axis=0))
        return np.vstack(rows)  # [K,K]

    def _E_a_over_j_rows(self, z):
        rows = []
        for A in self.A_by_dom:
            j = A @ z
            rows.append((A / np.clip(j[:, None], EPS, None)).mean(axis=0))
        return np.vstack(rows)  # [K,K]

    # ------- u_k(z) as cvxpy expressions -------

    def u_convex_exprs(self, z_var):
        exprs = []
        for A in self.A_by_dom:
            az = A @ z_var                        # affine (N_k,)
            exprs.append(cp.sum(-cp.log(az + EPS)) / A.shape[0])
        return exprs

    # ------- v_k(z_t) and grad v_k(z_t) -------

    def v_and_grad_at(self, z_t):
        # v_k(z) = E_{D_z}[log(k/j)] - E_{D_k}[log k]
        per_dom_E_log_k_over_j = self._per_domain_E_log_k_over_j(z_t)     # [K]
        Ez_log_k_over_j = float(per_dom_E_log_k_over_j @ z_t)              # scalar

        per_dom_E_log_k = np.array([self._E_log_k(B, z_t) for B in self.B_by_dom])  # [K]
        v_vals = Ez_log_k_over_j - per_dom_E_log_k                                   # [K]

        # gradient:
        E_b_over_k = self._E_b_over_k_rows(z_t)    # [K,K], row r = E_{D_r}[b/k]
        E_a_over_j = self._E_a_over_j_rows(z_t)    # [K,K], row r = E_{D_r}[a/j]
        shared = z_t @ (E_b_over_k - E_a_over_j)   # [K]

        grad = np.zeros((len(z_t), len(z_t)), dtype=np.float64)  # [K,K] (k,r)
        for k in range(len(z_t)):
            grad[k, :] = per_dom_E_log_k_over_j + shared - E_b_over_k[k, :]
        return v_vals, grad



def dc_cccp_step(z_t, u_exprs, v_vals, grad_v):
    K = len(z_t)
    z = cp.Variable(K)
    gamma = cp.Variable()
    cons = [z >= 0, cp.sum(z) == 1]
    for k in range(K):
        lin = v_vals[k] + grad_v[k] @ (z - z_t)
        cons.append(u_exprs[k] - lin <= gamma)

    prob = cp.Problem(cp.Minimize(gamma), cons)
    try:
        prob.solve(solver=cp.ECOS, abstol=1e-8, reltol=1e-8, feastol=1e-8, warm_start=True)
    except Exception:
        prob.solve(solver=cp.SCS, max_iters=10000, eps=1e-5, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"DC subproblem failed: {prob.status}")

    z_next = np.maximum(z.value, 0.0)
    s = z_next.sum()
    return (z_next / s) if s > 0 else np.ones_like(z_next) / K


# class DCBuilder(STATBuilder):
#     def __init__(self, base_models, beta_func, device="cuda"):
#         super().__init__(base_models, beta_func, device)
#
#     # ------- helpers at a given z (numpy arrays) -------
#     def _E_log_j(self, A, z): return np.mean(np.log(np.clip(A @ z, EPS, None)))
#     def _E_log_k(self, B, z): return np.mean(np.log(np.clip(B @ z, EPS, None)))
#
#     def _per_domain_E_log_k_over_j(self, z):
#         vals = []
#         for A, B in zip(self.A_by_dom, self.B_by_dom):
#             j = A @ z; k = B @ z
#             vals.append(np.mean(np.log(np.clip(k, EPS, None)) - np.log(np.clip(j, EPS, None))))
#         return np.array(vals)
#
#     def _E_b_over_k_rows(self, z):
#         rows = []
#         for B in self.B_by_dom:
#             k = B @ z
#             rows.append((B / np.clip(k[:, None], EPS, None)).mean(axis=0))
#         return np.vstack(rows)
#
#     def _E_a_over_j_rows(self, z):
#         rows = []
#         for A in self.A_by_dom:
#             j = A @ z
#             rows.append((A / np.clip(j[:, None], EPS, None)).mean(axis=0))
#         return np.vstack(rows)
#
#     # ------- u_k(z) as cvxpy expressions -------
#     def u_convex_exprs(self, z_var):
#         exprs = []
#         for A in self.A_by_dom:
#             az = A @ z_var
#             exprs.append(cp.sum(-cp.log(az + EPS)) / A.shape[0])
#         return exprs
#
#     # ------- v_k(z_t) and grad v_k(z_t) -------
#     def v_and_grad_at(self, z_t):
#         per_dom_E_log_k_over_j = self._per_domain_E_log_k_over_j(z_t)     # [K]
#         Ez_log_k_over_j = float(per_dom_E_log_k_over_j @ z_t)
#
#         per_dom_E_log_k = np.array([self._E_log_k(B, z_t) for B in self.B_by_dom])  # [K]
#         v_vals = Ez_log_k_over_j - per_dom_E_log_k
#
#         E_b_over_k = self._E_b_over_k_rows(z_t)
#         E_a_over_j = self._E_a_over_j_rows(z_t)
#         shared = z_t @ (E_b_over_k - E_a_over_j)
#
#         grad = np.zeros((len(z_t), len(z_t)), dtype=np.float64)
#         for k in range(len(z_t)):
#             grad[k, :] = per_dom_E_log_k_over_j + shared - E_b_over_k[k, :]
#         return v_vals, grad


def run_dc(domains, datamodule, base_models, beta_func,
           z0, num_iters=10, device="cuda", max_batches=None):
    loaders = [datamodule.get_dataloader(d, loader_type="val", is_training=False)
               for d in domains]

    dcb = DCBuilder(base_models, beta_func, device=device)
    z_t = np.asarray(z0, dtype=np.float64)
    zs = list()
    for t in range(num_iters):
        zs.append(z_t)
        print(f"Iteration {t + 1}/{num_iters}, z_t {z_t}")
        # --- NEW: ensure beta_func uses the current z_t ---
        if hasattr(beta_func, "set_z"):
            beta_func.set_z(torch.tensor(z_t, dtype=torch.float32, device=device))

        dcb.precompute_AB(loaders, domains, max_batches=max_batches)  # (optional: pass max_batches if supported)

        z_var = cp.Variable(len(z_t))
        u_exprs = dcb.u_convex_exprs(z_var)
        v_vals, grad_v = dcb.v_and_grad_at(z_t)

        u_vals_now = u_values_at(dcb, z_t)

        # --- print per-domain losses ---
        domain_loss = dcb._per_domain_E_log_k_over_j(z_t)
        mix_loss = float(domain_loss @ z_t)
        print(f"dc-solver: Round {t + 1}, domain-wise losses={domain_loss}, mix-loss={mix_loss}, z_t={z_t}")
        print(f"iter {t}: max_k[u_k - v_k] = {max(u - v for u, v in zip(u_vals_now, v_vals)):.6f}")

        z_next = dc_cccp_step(z_t, u_exprs, v_vals, grad_v)

        if np.linalg.norm(z_next - z_t, 1) < 1e-6:
            break
        z_t = z_next

    return zs


def u_values_at(dcb, z):
    return [float(-np.mean(np.log(np.clip(A @ z, EPS, None)))) for A in dcb.A_by_dom]
