# mixture_eval.py
import torch
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader
from transformers import BertModel
from data_loader import MNLIDataModule, collate_fn, input_handler
from trainer import LightBERTSeqClass, LightDomainClassifier, load_model, collect_domain_model_paths, BERT_BASE_DIR


def evaluate_classifier_on_mixture(
        classifier,
        datamodule,
        mixture_lambda,
        domains,
        device="cuda"
    ):
    """
    Evaluates a single classifier on a mixture of k sources

    Args:
        classifier: Trained classifier model
        datamodule: Data module containing source domains
        mixture_lambda: Weight vector for the mixture distribution
        domains: List of domain names
        device: Device to run evaluation on
    """
    classifier = classifier.to(device)
    classifier.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    # Create domain to index mapping
    domain_to_idx = {domain: idx for idx, domain in enumerate(domains)}

    # Evaluate per domain
    for domain in domains:
        # Setup datamodule for current domain (use validation data)
        datamodule.setup('fit', 'seq_pair_classif', domain)
        loader = datamodule.val_dataloader()

        domain_loss = 0
        domain_correct = 0
        domain_samples = 0

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['labels']
            batch_size = len(labels)

            with torch.no_grad():
                outputs = classifier(batch)
                loss = F.cross_entropy(outputs.logits, labels)
                preds = torch.argmax(outputs.logits, dim=1)
                correct = (preds == labels).sum().item()

            domain_loss += loss.item() * batch_size # average --> total loss
            domain_correct += correct
            domain_samples += batch_size

        # Calculate domain average loss and accuracy
        domain_avg_loss = domain_loss / domain_samples
        domain_avg_acc = domain_correct / domain_samples

        # Apply mixture weighting
        domain_weight = mixture_lambda[domain_to_idx[domain]]
        total_loss += domain_weight * domain_avg_loss
        total_correct += domain_weight * domain_avg_acc
        total_samples += 1  # Count domains for weighting

    return {
        "loss": total_loss,
        "accuracy": total_correct
    }


def evaluate_beta_weighted_ensemble(
        base_models,
        beta_func,
        datamodule,
        mixture_lambda,
        domains,
        device="cuda"
    ):
    #TODO: Current implementation averages logits not probs (not faithful)
    """
    Evaluates beta-weighted ensemble on mixture distribution

    Args:
        base_models: List of k base classifiers
        beta_func: Function that computes weights from input batch
        datamodule: Data module containing source domains
        mixture_lambda: Weight vector for mixture distribution
        domains: List of domain names
        device: Device to run evaluation on
    """
    # Move models to device
    base_models = [model.to(device).eval() for model in base_models]
    beta_func = beta_func.to(device).eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0
    domain_to_idx = {domain: idx for idx, domain in enumerate(domains)}

    #assert sum(mixture_lambda) == 1, "The mixture vector must sum to one"

    for domain in domains:
        datamodule.setup('fit', 'seq_pair_classif', domain)
        loader = datamodule.val_dataloader()
        datamodule.reset_batch_size(8) #TODO

        domain_loss = 0
        domain_correct = 0
        domain_samples = 0

        for batch in loader:
            batch, labels = input_handler(batch, device=device)
            batch_size = len(labels)

            with torch.no_grad():
                # Get predictions from all base models
                model_preds = []
                for model in base_models:
                    outputs = model(**batch)
                    model_preds.append(outputs.logits)

                # Stack predictions: [batch_size, num_domains, num_classes]
                stacked_preds = torch.stack(model_preds, dim=1)

                # Compute weighted prediction using beta function
                mixture_weights = beta_func(batch)
                weighted_pred = torch.einsum('bk,bkc->bc', mixture_weights, stacked_preds)

                # Compute loss
                loss = F.cross_entropy(weighted_pred, labels)
                preds = torch.argmax(weighted_pred, dim=1)
                correct = (preds == labels).sum().item()

            domain_loss += loss.item() * batch_size
            domain_correct += correct
            domain_samples += batch_size

        # Calculate domain average loss and accuracy
        domain_avg_loss = domain_loss / domain_samples
        domain_avg_acc = domain_correct / domain_samples

        # Apply mixture weighting
        domain_weight = mixture_lambda[domain_to_idx[domain]]
        total_loss += domain_weight * domain_avg_loss
        total_correct += domain_weight * domain_avg_acc
        total_samples += 1  # Count domains for weighting

    if isinstance(total_loss, torch.Tensor):
        total_loss = total_loss.item()
    if isinstance(total_correct, torch.Tensor):
        total_correct = total_correct.item()

    return {
        "loss": total_loss,
        "accuracy": total_correct,
        "total_samples": total_samples,
    }


class DomainClassifierToBeta(torch.nn.Module):
    def __init__(self, domain_classifier, initial_z, trainable=False):
        super().__init__()
        self.domain_classifier = domain_classifier
        self.trainable = trainable

        if not isinstance(initial_z, torch.Tensor):
            initial_z = torch.tensor(initial_z, dtype=torch.float32)

        if trainable:
            self.z = torch.nn.Parameter(initial_z.clone())
        else:
            self.register_buffer('z', initial_z.clone().detach())


    def forward(self, batch):
        """
        Compute beta weights for a batch of inputs
        Implements:
        beta_i(x) = ( z_i * (Qhat(i|x)) ) /
                    sum_j z_j * (Qhat(j|x))
        """
        #TODO: prior is not needed ANYMORE
        #TODO: check if logits should be used OR probability in both versions
        with torch.no_grad():
            logits = self.domain_classifier(batch)  # [B, k]
            q_post = F.softmax(logits, dim=1)          # [B, k], Qhat(i|x)

        if self.trainable:
            z_val = F.softplus(self.z)
        else:
            z_val = self.z

        # Compute numerator: z_i * Qhat(i|x)
        numerator = z_val.unsqueeze(0) * q_post

        # Denominator: sum over j
        denominator = numerator.sum(dim=1, keepdim=True)  # [B, 1]

        beta = numerator / (denominator + 1e-12)
        return beta

    def get_z(self):
        if self.trainable:
            return F.softplus(self.z).detach().clone()
        return self.z.detach().clone()

    def set_z(self, new_z: torch.Tensor):
        """
        Update the mixture weights z used in beta(x).
        Expects a 1D tensor on the correct device.
        """
        if not isinstance(new_z, torch.Tensor):
            new_z = torch.tensor(new_z, dtype=torch.float32)
        new_z = new_z.to(self.z.device)
        if self.trainable:
            self.z.data = new_z.clone()
        else:
            self.z.data = new_z.clone().detach()


class BERTBetaFunction(torch.nn.Module):
    """BERT-based beta function that outputs domain weights"""

    def __init__(self, model_name="bert-base-uncased", num_domains=5, unfreeze_layers=None):
        """
        Args:
            model_name: Hugging Face model identifier
            num_domains: Number of domains (k)
            unfreeze_layers: List of layer indices to unfreeze (e.g., [10, 11] for last two layers)
        """
        super().__init__()
        # Load BERT encoder
        self.encoder = BertModel.from_pretrained(model_name)
        self.num_domains = num_domains

        # Classification head
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, num_domains)

        # Freeze all layers initially
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze specified layers
        if unfreeze_layers:
            self._unfreeze_layers(unfreeze_layers)


    def _unfreeze_layers(self, layer_indices):
        """Unfreeze specific encoder layers"""
        for i in layer_indices:
            # Handle embeddings
            if i == "embeddings":
                for param in self.encoder.embeddings.parameters():
                    param.requires_grad = True

            # Handle encoder layers
            elif isinstance(i, int) and 0 <= i < self.encoder.config.num_hidden_layers:
                for param in self.encoder.encoder.layer[i].parameters():
                    param.requires_grad = True

    def forward(self, batch):
        # Extract inputs
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch.get('token_type_ids', None)

        # Pass through BERT encoder
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        # Get [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # Compute domain logits
        logits = self.classifier(cls_embedding)  # [batch_size, num_domains]

        # Convert to weights (probabilities)
        weights = F.softmax(logits, dim=1)  # [batch_size, num_domains]

        return weights


# Modified BetaTrainer
class BetaTrainer(L.LightningModule):
    def __init__(self, base_models, beta_module, optimizer_config):
        """
        Args:
            base_models: List of k fixed task classifiers
            beta_module: Beta weighting module (e.g., BERTBetaFunction)
            optimizer_config: Dictionary with optimizer parameters
        """
        super().__init__()
        self.base_models = torch.nn.ModuleList(base_models)
        self.beta_module = beta_module
        self.optimizer_config = optimizer_config

        # Freeze base models
        for model in self.base_models:
            model.requires_grad_(False)
            model.eval()

    def forward(self, batch):
        weights = self.beta_module(batch)  # [batch_size, k]

        # Get predictions from all base models
        model_preds = []
        for model in self.base_models:
            with torch.no_grad():
                outputs = model(batch)
                model_preds.append(outputs.logits)

        stacked_preds = torch.stack(model_preds, dim=1)  # [batch_size, k, num_classes]
        weighted_pred = torch.einsum('bk,bkc->bc', weights, stacked_preds)
        return weighted_pred

    def training_step(self, batch, batch_idx):
        weighted_pred = self(batch)
        loss = F.cross_entropy(weighted_pred, batch['labels'])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.beta_module.parameters(),
            lr=self.optimizer_config.get('lr', 2e-5),
            weight_decay=0.01
        )


def discriminator_weighted_ensemble_loss(
        base_models,
        beta_func,
        datamodule,
        mixture_lambda,  # Now a torch.Tensor (trainable)
        domains,
        device="cuda"
    ):
    """
    Evaluates beta-weighted ensemble on mixture distribution

    Args:
        base_models: List of k base classifiers
        beta_func: Function that computes weights from input batch
        datamodule: Data module containing source domains
        mixture_lambda: Weight vector for mixture distribution (trainable Tensor)
        domains: List of domain names
        device: Device to run evaluation on
    """
    # Ensure mixture_lambda is on correct device
    mixture_lambda = mixture_lambda.to(device)

    # Move models to device
    base_models = [model.to(device).eval() for model in base_models]
    beta_func = beta_func.to(device).eval()

    total_loss = torch.tensor(0.0, device=device)  # Keep as tensor for gradients
    total_accuracy = 0.0  # Track separately (not differentiable)
    result = dict()
    domain_to_idx = {domain: idx for idx, domain in enumerate(domains)}

    for domain in domains:
        datamodule.setup('fit', 'seq_pair_classif', domain)
        loader = datamodule.val_dataloader()
        datamodule.reset_batch_size(8)  # TODO: Adjust domain

        domain_total_loss = 0.0
        domain_correct = 0
        domain_samples = 0

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            labels = batch['labels']
            batch_size = len(labels)

            # Compute without gradients for base models/beta_func
            with torch.no_grad():
                model_preds = []
                for model in base_models:
                    outputs = model(batch)
                    model_preds.append(outputs.logits)

                stacked_preds = torch.stack(model_preds, dim=1)
                mixture_weights = beta_func(batch)
                weighted_pred = torch.einsum('bk,bkc->bc', mixture_weights, stacked_preds)

                loss_batch = F.cross_entropy(weighted_pred, labels)
                preds = torch.argmax(weighted_pred, dim=1)
                correct_batch = (preds == labels).sum().item()

            # Accumulate domain stats
            domain_total_loss += loss_batch.item() * batch_size
            domain_correct += correct_batch
            domain_samples += batch_size

        # Calculate domain average loss/accuracy
        domain_avg_loss = domain_total_loss / domain_samples
        domain_avg_accuracy = domain_correct / domain_samples

        result["domain-{}:loss".format(domain)] = domain_avg_loss
        result["domain-{}:acc".format(domain)] = domain_avg_accuracy

        # Differentiable part: Convert to tensor and weight by mixture_lambda
        idx = domain_to_idx[domain]
        total_loss = total_loss + mixture_lambda[idx] * torch.tensor(domain_avg_loss, device=device)

        # Non-differentiable accuracy (use detached value)
        total_accuracy += mixture_lambda[idx].detach().item() * domain_avg_accuracy

    result["loss"] = total_loss
    result["accuracy"] = total_accuracy
    result["total_samples"] = len(domains)

    return result


# Modified training function
def train_beta_weighting(
        base_models,
        beta_model,  # Pre-configured beta model (e.g., BERTBetaFunction)
        datamodule,
        mixture_lambda,
        domains,
        optimizer_config,
        train_params,
        logger=None
):
    """
    Trains beta weighting function

    Args:
        beta_model: Pre-initialized beta model (must be trainable)
        ... (other params unchanged)
    """
    model = BetaTrainer(
        base_models=base_models,
        beta_module=beta_model,
        optimizer_config=optimizer_config
    )

    # Configure mixture sampling
    datamodule.set_sampler_lambda(
        mixture_lambda,
        loader_type='train',
        size=train_params.get('num_train_samples', 10000)
    )
    datamodule.setup('fit', 'seq_pair_classif', "all")

    trainer = L.Trainer(
        max_epochs=train_params.get('num_epochs', 5),
        logger=logger,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        enable_progress_bar=True
    )

    trainer.fit(model, datamodule.train_dataloader())
    return model.beta_module


class DomainAttentionBeta(torch.nn.Module):
    """Trainable attention-based beta function"""

    def __init__(self, num_domains, hidden_size):
        super().__init__()
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(num_domains, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, num_domains)
        )

    def forward(self, stacked_preds):
        # TODO: there should be task labels
        # stacked_preds: [batch_size, num_domains, num_classes]
        batch_size = stacked_preds.size(0)

        # Compute attention weights
        attn_input = stacked_preds.mean(dim=-1)  # [batch_size, num_domains]
        attn_weights = self.attention(attn_input)
        attn_weights = F.softmax(attn_weights, dim=-1)  # [batch_size, num_domains]

        # Compute weighted prediction
        return torch.einsum('bk,bkc->bc', attn_weights, stacked_preds)


def main():
    # Configuration
    data_dir = r'C:\Users\tangc\PycharmProjects\domain-adaptation\datasets'  # Replace with your data directory
    domains = ["fiction", "travel", "government"]  # Replace with actual domain names
    sentiments = ["entailment", "contradiction", "neutral"]
    mixture_lambda = [0.3, 0.3, 0.4]  # Sum to 1, matches DOMAINS order
    domain_models_dir = r"C:\Users\tangc\PycharmProjects\domain-adaptation\logs\domain_models"
    domain_model_paths = collect_domain_model_paths(domains, domain_models_dir)

    PRETRAINED_MODEL_NAME = "bert-base-uncased"
    batch_size = 16
    NUM_EPOCHS = 10
    LEARNING_RATE = 3e-5
    NUM_TRAIN_SAMPLES = 10000  # Size of the training mixture

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize data module
    datamodule = MNLIDataModule(
        data_dir=data_dir,
        domains=domains,
        sentiments=sentiments,
        val_size=1000  # Default validation size
    )
    datamodule.prepare_data()  # Preprocess data if needed

    # Set batch size
    datamodule.reset_batch_size(batch_size)

    # Load base classifiers (task models)
    base_models = []
    for ckpt_path in domain_model_paths:
        # Assuming load_model is a function that loads a Lightning model from checkpoint
        model = load_model(ckpt_path, pretrained_path=BERT_BASE_DIR, num_labels=3)
        model.to(device)
        model.eval()
        base_models.append(model)

    # Initialize BERT encoder for beta function
    #bert_encoder = AutoModel.from_pretrained(PRETRAINED_MODEL_NAME)
    #beta_model = BERTBetaFunction(bert_encoder, num_domains=len(DOMAINS))
    beta_model = BERTBetaFunction(num_domains=len(domains))
    beta_model.to(device)

    # Training configuration
    optimizer_config = {"lr": LEARNING_RATE}
    train_params = {
        "num_epochs": NUM_EPOCHS,
        "num_train_samples": NUM_TRAIN_SAMPLES
    }

    # Setup logger (TensorBoard in this example)
    logger = TensorBoardLogger("logs", name="beta_training")

    # Train beta function
    print("Training beta function...")
    trained_beta = train_beta_weighting(
        base_models=base_models,
        beta_model=beta_model,
        datamodule=datamodule,
        mixture_lambda=mixture_lambda,
        domains=domains,
        optimizer_config=optimizer_config,
        train_params=train_params,
        logger=logger
    )

    # Save trained beta model
    torch.save(trained_beta.state_dict(), "trained_beta_model.pt")
    print("Saved beta model to 'trained_beta_model.pt'")

    # Evaluate ensemble performance on the mixture validation
    # We'll create a validation mixture using the same lambda
    print("Evaluating ensemble on mixture...")

    # Set up datamodule for evaluation
    datamodule.set_sampler_lambda(
        mixture_lambda,
        loader_type='val',
        size=1000  # Validation set size
    )

    # Evaluate using the beta-weighted ensemble
    eval_results = evaluate_beta_weighted_ensemble(
        base_models=base_models,
        beta_func=trained_beta,
        datamodule=datamodule,
        mixture_lambda=mixture_lambda,
        domains=domains,
        device=device
    )

    print("\nEvaluation Results:")
    print(f"Mixture Loss: {eval_results['loss']:.4f}")
    print(f"Mixture Accuracy: {eval_results['accuracy']:.4f}")
    print(f"Based on {eval_results['total_samples']} domains")


if __name__ == "__main__":
    from pytorch_lightning.loggers import TensorBoardLogger

    main()