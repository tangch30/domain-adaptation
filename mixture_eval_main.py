# mixture_eval_main.py
import os
import torch
import json
import argparse
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any

# Import necessary components from your existing code
from data_loader import MNLIDataModule
from trainer import (
    LightBERTSeqClass,
    LightDomainClassifier,
    load_model,
    collect_domain_model_paths
)
from mixture_eval import (
    evaluate_beta_weighted_ensemble,
    DomainClassifierToBeta
)
from tqdm import tqdm
from hedge import _to_inputs


EPS = 1e-12

def main(
        data_dir: str,
        domains: List[str],
        domain_models_dir: str,
        domain_classifier_path: str,
        z_vectors: List[List[float]],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        eval_mode = "domain-matrix",
        max_batches = None
    ) -> Dict[str, Any]:
    """
    Main evaluation function for beta-weighted ensemble

    Args:
        data_dir: Directory containing domain datasets
        domains: List of domain names
        domain_model_dir: Directory with domain-specific task models
        domain_classifier_path: Path to trained domain classifier checkpoint
        z_vectors: List of z-vectors to evaluate
        device: Device to run evaluation on

    Returns:
        Dictionary with evaluation results for each z-vector
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # 1. Prepare data module
    sentiments = ['entailment', 'contradiction', 'neutral']
    datamodule = MNLIDataModule(data_dir, domains, sentiments)
    datamodule.prepare_data()
    datamodule.reset_batch_size(8)

    # 2. Load domain-specific task models
    domain_model_paths = collect_domain_model_paths(domains, domain_models_dir)
    print("domain models path ", domain_model_paths)
    base_models = []
    for path in domain_model_paths:
        model = load_model(path)
        model.eval()
        model.to(device)
        base_models.append(model)

    print(f"Loaded {len(base_models)} base task models")

    # 3. Load domain classifier
    config = {
        "domain_model_paths": domain_model_paths,
        "optimizer_config": {"lr": 2e-5, "mu": 0.01},
        "freeze_domain_models": True
    }

    results = {}

    # 6. Update datamodule
    #datamodule.setup('fit', 'domain_classif', "all")
    if eval_mode == "ensemble":
        domain_classifier = LightDomainClassifier(**config)
        checkpoint = torch.load(domain_classifier_path, map_location=device) #TODO: change to cpu?
        # Use trained domain models for domain classification instead
        domain_classifier.load_state_dict(checkpoint['state_dict'])
        domain_classifier.eval()
        domain_classifier.to(device)

        print("Loaded domain classifier from:", domain_classifier_path)

        # 4. Prepare mixture lambda (uniform by default)
        # mixture_lambda = torch.ones(len(domains)) / len(domains)

        # 5. Evaluate each z-vector
        for i, z_list in enumerate(z_vectors):
            # Convert to tensor
            z_vector = torch.tensor(z_list, dtype=torch.float32)

            # Create beta function
            beta_func = DomainClassifierToBeta(
                domain_classifier=domain_classifier,
                initial_z=z_vector,
                trainable=False  # Evaluation only
            )
            beta_func.to(device)
            beta_func.eval()

            # Evaluate
            #TODO: debug here
            metrics = evaluate_beta_weighted_ensemble(
            base_models=base_models,
            beta_func=beta_func,
            datamodule=datamodule,
            mixture_lambda=z_vector,
            domains=domains,
            device=device
            )

            # Store results
            results[f"z_vector_{i}"] = {
                "z_vector": z_list,
                "mixture_lambda": z_list,
                "metrics": metrics
            }

            print(f"Evaluation for z_vector_{i} ({z_list}):")
            print(f"  Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
            return results

    elif eval_mode == "domain-matrix":
        data_domains = ["fiction", "government", "slate", "telephone", "travel"]
        loaders = []
        for domain in data_domains:
            datamodule.setup("fit", "seq_pair_classif", domain)
            datamodule.reset_batch_size(8)
            loaders.append(datamodule.val_dataloader())

        k, t = len(data_domains), len(base_models)
        losses = torch.zeros((k, t), dtype=torch.float32, device=device)
        acc = torch.zeros((k, t), dtype=torch.float32, device=device)
        for i, domain in enumerate(data_domains):
            loader = loaders[i]
            # Create a progress bar for this domain
            for j, model in enumerate(base_models):
                total_loss, total_correct, total_samples = 0.0, 0, 0
                progress_bar = tqdm(
                    enumerate(loader),
                    total=len(loader) if max_batches is None else min(max_batches, len(loader)),
                    desc=f"Evaluating model {domains[j]} on domain {domain}",
                    leave=True  # Keep the progress bar after completion
                )
                for bi, batch in progress_bar:
                    if max_batches is not None and bi >= max_batches:
                        break
                    labels = batch["labels"].to(device)
                    batch = _to_inputs(batch, device)  # drop labels
                    outputs = model(**batch)
                    probs = F.softmax(outputs.logits, dim=-1)  # [B, C]
                    logp = torch.log(probs.clamp_min(EPS))
                    loss_batch = F.nll_loss(logp, labels, reduction="mean")
                    _, predicted = torch.max(outputs.logits, 1)
                    total_correct += (predicted == labels).sum().item()
                    total_loss += float(loss_batch.item()) * labels.size(0)
                    total_samples += labels.size(0)
                    #print(f"domain={domains[i]}, model={j}, total_loss={total_loss}")

                losses[i, j] = total_loss / max(total_samples, 1)
                acc[i, j] = total_correct / max(total_samples, 1)


        losses = np.array(losses.to('cpu'))
        acc = np.array(acc.to('cpu'))
        for i, domain in enumerate(data_domains):
            for j, model in enumerate(base_models):
                results[f"losses:{domain}-{domains[j]}"] = losses[i, j].item()
                results[f"acc:{domain}-{domains[j]}"] = acc[i, j].item()
        return results





if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Evaluate beta-weighted ensemble with different z-vectors")
    parser.add_argument("--data_dir", required=False,
                        help="Path to dataset directory",
                        default=r'C:\Users\tangc\PycharmProjects\domain-adaptation\datasets')
    parser.add_argument("--domain", type=str, required=True, help="Target domain(s) for inference")
    parser.add_argument("--domain_models_dir", type=str, default=None,
                        help="Directory containing trained domain-specific models")
    parser.add_argument("--domain_classifier_path", type=str, required=True,
                        help="Path to domain classifier checkpoint")
    parser.add_argument("--z_vectors", type=str, required=True, help="JSON file containing list of z-vectors")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file for results")

    args = parser.parse_args()

    # Load z-vectors from JSON file
    with open(args.z_vectors, 'r') as f:
        z_vectors = json.load(f)

    # Validate z-vectors
    if not isinstance(z_vectors, list) or not all(isinstance(z, list) for z in z_vectors):
        raise ValueError("z_vectors should be a list of lists of numbers")

    # Run evaluation
    results = main(
        data_dir=args.data_dir,
        domains=args.domain.split("-"),
        domain_models_dir=args.domain_models_dir,
        domain_classifier_path=args.domain_classifier_path,
        z_vectors=z_vectors
    )

    # Save results
    if isinstance(results, dict):
        print(results)
        with open(args.output+"_"+args.domain+".json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Evaluation complete. Results saved to {args.output}")
    elif isinstance(results, tuple):
        for obj in results:
            print(obj)