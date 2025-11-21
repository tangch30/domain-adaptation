import numpy as np
from typing import List, Dict, Any
import torch, gc
import torch.nn.functional as F
from tqdm import tqdm
from data_loader import MNLIDataModule
from trainer import collect_domain_model_paths, load_model, LightDomainClassifier


EPS = 1e-12

def prepare_datamodule(data_dir: str, domains: List[str], batch_size: int = 8):
    """Prepare MNLI datamodule for given domains."""
    sentiments = ['entailment', 'contradiction', 'neutral']
    datamodule = MNLIDataModule(data_dir, domains, sentiments)
    datamodule.prepare_data()
    datamodule.reset_batch_size(batch_size)
    return datamodule


def load_base_models(domains: List[str], domain_models_dir: str, device: str):
    """Load domain-specific task models and return them as a list."""
    domain_model_paths = collect_domain_model_paths(domains, domain_models_dir)
    base_models = []
    for path in domain_model_paths:
        model = load_model(path)
        model.eval().to(device)
        base_models.append(model)
    print(f"Loaded {len(base_models)} base task models")
    return base_models, domain_model_paths


def load_domain_classifier(domain_model_paths: List[str], domain_classifier_path: str, device: str):
    """Load a trained LightDomainClassifier from checkpoint."""
    config = {
        "domain_model_paths": domain_model_paths,
        "optimizer_config": {"lr": 2e-5, "mu": 0.01},
        "freeze_domain_models": True
    }
    domain_classifier = LightDomainClassifier(**config)
    checkpoint = torch.load(domain_classifier_path, map_location=device)
    domain_classifier.load_state_dict(checkpoint['state_dict'])
    domain_classifier.eval().to(device)
    print("Loaded domain classifier from:", domain_classifier_path)
    return domain_classifier


def clear_torch_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _to_inputs(batch, device):
    return {k: v.to(device) for k, v in batch.items() if k != "labels" and hasattr(v, "to")}


def _extract_logits(output):
    """Handle different model output formats (Tensor, object with .logits, tuple)."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, tuple):
        return output[0]
    raise TypeError(f"Unsupported output type: {type(output)}")


class STATBuilder:
    def __init__(self, base_models_paths, beta_func, device="cuda"):
        self.base_models_paths = base_models_paths
        self.beta_func = beta_func
        self.device = device
        self.K = len(base_models_paths)
        self.A_by_dom = []
        self.B_by_dom = []


    @torch.no_grad()
    def precompute_AB(self, domain_loaders, domains, max_batches=None, use_amp=True):
        """
        Precompute A and B matrices domain-wise.
        Memory-friendly version: iterate over models first,
        so only one model is on GPU at a time.
        """
        self.A_by_dom.clear()
        self.B_by_dom.clear()

        # First compute B (domain classifier weights) once
        print("Building B")
        all_B = []
        self.beta_func.to(self.device).eval()
        for loader in domain_loaders:
            b_rows = []
            for bi, batch in enumerate(loader):
                if max_batches is not None and bi >= max_batches:
                    break
                #labels = batch["labels"].to(self.device, non_blocking=True)
                inputs = _to_inputs(batch, device=self.device)
                B = self.beta_func(inputs)        # [B,K]
                b_rows.append(B.clamp_min(EPS).cpu())
            if b_rows:
                all_B.append(torch.cat(b_rows, 0))
            else:
                all_B.append(torch.full((1, self.K), EPS, dtype=torch.float32))
        self.beta_func.to("cpu").detach()
        del self.beta_func

        # Initialize A as zeros, to be filled column by column
        all_A = [torch.zeros_like(B, dtype=torch.float32) for B in all_B]

        print("Building B")
        # Now loop over models, one at a time
        for m_idx, model_path in enumerate(self.base_models_paths):
            m = load_model(model_path)
            m.eval()
            m.to(self.device)

            for d_idx, loader in enumerate(domain_loaders):
                offset = 0
                progress_bar = tqdm(
                                enumerate(loader),
                                total=len(loader) if max_batches is None else min(max_batches, len(loader)),
                                desc=f"Evaluating domain {domains[d_idx]}",
                                leave=True  # Keep the progress bar after completion
                            )
                for bi, batch in progress_bar:
                    if max_batches is not None and bi >= max_batches:
                        break
                    labels = batch["labels"].to(self.device, non_blocking=True)
                    inputs = _to_inputs(batch, device=self.device)

                    out = m(**inputs)
                    logits = _extract_logits(out)
                    probs = F.softmax(logits, dim=-1)
                    true_probs = probs[torch.arange(len(labels), device=probs.device),
                                       labels] # B

                    # Write into the m_idx-th column of A for this domain
                    # TODO: Need checking
                    A_slice = true_probs.clamp_min(EPS).to("cpu", dtype=torch.float32)
                    all_A[d_idx][offset:offset + len(A_slice), m_idx] = (
                        A_slice * all_B[d_idx][offset:offset + len(A_slice), m_idx]
                    ) # B
                    offset += len(A_slice)

            m.to("cpu").detach()
            del m
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

        # Save results as numpy arrays
        self.A_by_dom = [A.numpy() for A in all_A] # a list of K matrices of size BxK
        self.B_by_dom = [B.numpy() for B in all_B]


# ==================== MODEL WRAPPERS ====================
class TorchBaseWrapper:
    """Wrapper for PyTorch base models"""
    def __init__(self, model, device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def predict_proba(self, inputs):
        with torch.no_grad():
            self.model.to(self.device)
            outputs = self.model(**inputs)
            self.model.to("cpu")
            return F.softmax(outputs.logits, dim=-1)

class SklearnBaseWrapper:
    """Wrapper for scikit-learn base models"""
    def __init__(self, model, input_key, device):
        self.model = model
        self.input_key = input_key
        self.device = device

    def predict_proba(self, inputs):
        x = inputs[self.input_key].cpu().numpy()
        probs = self.model.predict_proba(x).astype(np.float32)
        return torch.from_numpy(probs).to(self.device)

class TorchDomainWrapper:
    """Wrapper for PyTorch domain classifiers"""
    def __init__(self, model, device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def predict_proba(self, inputs):
        with torch.no_grad():
            self.model.to(self.device)
            outputs = self.model(inputs)
            self.model.to("cpu")
            return F.softmax(outputs, dim=1)

class SklearnDomainWrapper:
    """Wrapper for scikit-learn domain classifiers"""
    def __init__(self, model, input_key, device):
        self.model = model
        self.input_key = input_key
        self.device = device

    def predict_proba(self, inputs):
        x = inputs[self.input_key].cpu().numpy()
        probs = self.model.predict_proba(x).astype(np.float32)
        return torch.from_numpy(probs).to(self.device)


class TorchBetaWrapper:
    """Wrapper for PyTorch beta functions"""
    def __init__(self, model, device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def __call__(self, inputs):
        with torch.no_grad():
            self.model.to(self.device)
            outputs = self.model(inputs)
            self.model.to("cpu")
            return F.softmax(outputs, dim=1)


class SklearnBetaWrapper:
    """Wrapper for scikit-learn beta functions"""
    def __init__(self, model, input_key, device):
        self.model = model
        self.input_key = input_key
        self.device = device

    def __call__(self, inputs):
        x = inputs[self.input_key].cpu().numpy()
        probs = self.model.predict_proba(x).astype(np.float32)
        return torch.from_numpy(probs).to(self.device)

