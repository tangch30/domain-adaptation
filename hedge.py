# coded by ChatGPT
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from eval_utils import STATBuilder, _to_inputs, TorchBaseWrapper, SklearnBaseWrapper, TorchDomainWrapper, SklearnDomainWrapper, TorchBetaWrapper, SklearnBetaWrapper

EPS = 1e-12


def _to_inputs(batch, device):
    return {k: v.to(device) for k, v in batch.items() if k != "labels" and hasattr(v, "to")}


# @torch.no_grad()
# def _predict_probas(models, inputs):
#     cols = []
#     for m in models:
#         out = m(**inputs)                  # has .logits
#         p = F.softmax(out.logits, dim=-1)  # [B, C]
#         cols.append(p.unsqueeze(2))        # [B, C, 1]
#     return torch.cat(cols, dim=2)          # [B, C, K]

@torch.no_grad()
def _predict_probas(wrapped_models, inputs):
    cols = []
    for model in wrapped_models:
        p = model.predict_proba(inputs)
        cols.append(p.unsqueeze(2))
    return torch.cat(cols, dim=2)


def _cross_entropy(pred, labels):
    # pred: [B, C], labels: [B]
    logp = torch.log(pred.clamp_min(EPS))
    return F.nll_loss(logp, labels, reduction="mean")


def _abs_loss(pred, labels):
    return F.l1_loss(pred[:, 1], labels, reduction="mean")


class DabHedgeRunner:
    r"""
    dab-Hedge (Algorithm 1) for multiple source adaptation.

    Maintains per-domain losses, multiplicative weights, and produces
    ensemble predictors h_{λ_t}. Supports either:
      - di_fns: [K] functions D_i(x) (unnormalized scores),
      - or domain_clf: returns P(i|x) (DMSA approx).
    """

    def __init__(self, base_models, di_fns=None, domain_clf=None,
                 prior_pi=None, device="cuda", input_key='features'):
        self.device = device
        self.input_key = input_key
        self.K = len(base_models)
        self.prior_pi = prior_pi if prior_pi is not None else self.K

        # Wrap base models
        self.base_models = []
        for model in base_models:
            if isinstance(model, torch.nn.Module):
                self.base_models.append(TorchBaseWrapper(model, device))
            else:
                self.base_models.append(SklearnBaseWrapper(model, input_key, device))

        # Wrap domain functions
        self.di_fns = None
        if di_fns is not None:
            self.di_fns = []
            for fn in di_fns:
                if isinstance(fn, torch.nn.Module):
                    self.di_fns.append(TorchDomainWrapper(fn, device))
                else:
                    self.di_fns.append(SklearnDomainWrapper(fn, input_key, device))

        # Wrap domain classifier
        self.domain_clf = None
        if domain_clf is not None:
            # Wrap beta function
            if isinstance(domain_clf, torch.nn.Module):
                self.domain_clf = TorchBetaWrapper(domain_clf, device)
            else:
                self.domain_clf = SklearnBetaWrapper(domain_clf, input_key, device)


    def _beta_z(self, z, inputs):
        """
        Compute per-example mixture weights β_z(x,i).
        Ensures all outputs are float32 to avoid dtype mismatches.
        """
        z = torch.as_tensor(z, dtype=torch.float32, device=self.device)  # [K]

        if self.di_fns is not None:
            # score each domain D_i(x)
            scores = []
            for fn in self.di_fns:
                s = fn.predict_proba(inputs).squeeze(1)
                scores.append(s)
            scores = torch.stack(scores, dim=1)  # [B, K]
            num = z[None, :] * scores
        else:
            p = self.domain_clf(inputs)
            #prior_pi_tensor = torch.as_tensor(self.prior_pi, dtype=torch.float32, device=self.device)
            #num = z[None, :] * (p / prior_pi_tensor)
            num = z[None, :] * (p / self.prior_pi)

        den = num.sum(dim=1, keepdim=True).clamp_min(EPS)
        beta = num / den
        return beta.to(torch.float32)  # enforce float32

    @torch.no_grad()
    def ensemble_predict(self, z, inputs):
        """
        Compute mixture probabilities h_z(x) = sum_i β_z(x,i) h_i(x)
        """
        probas = _predict_probas(self.base_models, inputs)  # [B, C, K], float32
        betas = self._beta_z(z, inputs)  # [B, K], now float32
        #betas = betas.to(torch.float32)  # extra safety
        mix = torch.einsum("bck,bk->bc", probas, betas)  # [B, C]
        return mix

    def step_losses(self, z, loaders, domains, max_batches=None):
        losses = np.zeros(self.K, dtype=np.float32)
        for i, loader in enumerate(loaders):
            total, n = 0.0, 0
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
                pred = self.ensemble_predict(z, inputs)
                #total += float(_cross_entropy(pred, labels))
                total += float(_abs_loss(pred, labels))
                n += 1

            losses[i] = np.float32(total / max(1, n))
        return losses


def run_dab_hedge(domains, datamodule, base_models, z0=None,
                  di_fns=None, domain_clf=None, prior_pi=None, batch_size_eval=8,
                  beta=0.1, rho=1.0, T=50, device="cuda", max_batches=None,
                  use_fip=False):
    loaders = []
    datamodule.reset_batch_size(batch_size_eval)
    for domain in domains:
        #datamodule.setup("fit", "seq_pair_classif", domain)
        # loaders = [datamodule.get_dataloader(d, loader_type="train", is_training=True)
        #            for d in domains]
        loaders.append(datamodule.get_dataloader(domain, loader_type="val", is_training=False))

    #print(f"v1 domain {domains[0]}, data={loaders[0]} ")
    #print(f"v1 domain {domains[1]}, data={loaders[1]} ")

    runner = DabHedgeRunner(base_models, di_fns=di_fns,
                            domain_clf=domain_clf,
                            prior_pi=prior_pi,
                            device=device)

    K = len(base_models)
    # init uniform
    lambdas = []
    if z0 is None:
        w = np.ones(K, dtype=np.float32)
    else:
        w = z0

    res = dict()
    for t in range(T):
        z_t = w / w.sum()
        lambdas.append(z_t)
        losses = runner.step_losses(z_t, loaders, domains, max_batches=max_batches)
        if use_fip:
            w= w * losses + 0.001/K
        else:
            w = w * np.exp(beta * losses / rho)
        mix_loss = np.dot(z_t, losses)
        print(f"dab-hedge-v1: Round {t+1}/{T}, domain-wise losses={losses}, mix-loss={mix_loss}, z_t={z_t}")
        res[f"t-{t+1}:loss"] = losses.tolist()
        res[f"t-{t+1}:z_t"] = z_t.tolist()

    suffix = 'fip' if use_fip else 'hedge'
    with open(f"temp_{suffix}.json", 'w') as f:
        json.dump(res, f)

    return lambdas, runner


def predict_avg(runner, lambdas, x_batch, device="cuda"):
    """
    Average ensemble predictor: (1/T) Σ_t h_{λ_t}(x).
    x_batch: dict with "labels" removed, already on device.
    """
    T = len(lambdas)
    acc = None
    for z_t in lambdas:
        pred = runner.ensemble_predict(z_t, x_batch)  # [B,C]
        acc = pred if acc is None else acc + pred
    return acc / T

