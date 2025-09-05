"""Train domain classifier and task predictor"""
## DeepSeek is used during coding

import lightning as L
import torch
import argparse, os, json
import torch.nn.functional as F
from datetime import datetime
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoConfig, BertForSequenceClassification, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from data_loader import MNLIDataModule, mnli_preprocess

#TODO: Change validation metric, check log frequency (you haven't written any data to event files -tfb)
BERT_BASE_DIR=r'C:\Users\tangc\PycharmProjects\domain-adaptation\bert-base-uncased-cache'


def collect_domain_model_paths(domains, domain_models_dir):
    """
    Collect model checkpoint paths for given domains from a directory.

    Args:
        domains: List of domain names
        domain_models_dir: Directory containing model checkpoints

    Returns:
        List of paths to model checkpoints in the same order as domains
    """
    domain_model_paths = []
    domain_files = {}

    print(f"Searching in directory: {domain_models_dir}")
    print(f"Directory exists? {os.path.exists(domain_models_dir)}")
    print(f"Files in directory: {os.listdir(domain_models_dir)}")

    # Collect all checkpoint files in directory
    for root, _, files in os.walk(domain_models_dir):
        print(root)
        print(files)
        for file in files:
            if file.endswith(".ckpt"):
                full_path = os.path.join(root, file)
                # Try to find matching domain in filename
                for domain in domains:
                    if domain.lower() in file.lower():
                        domain_files[domain] = full_path
                        break

    # Create ordered list based on input domains
    for domain in domains:
        if domain in domain_files:
            domain_model_paths.append(domain_files[domain])
        else:
            raise FileNotFoundError(
                f"No checkpoint found for domain '{domain}' in {domain_models_dir}. "
                "Filenames should contain domain names."
            )

    print(f"Loaded domain models:")
    for domain, path in zip(domains, domain_model_paths):
        print(f"  {domain}: {path}")

    return domain_model_paths


# def load_model(ckpt_path, pretrained_path=BERT_BASE_DIR, num_labels=3):
#     model = AutoModelForSequenceClassification.from_pretrained(
#         pretrained_model_name_or_path=pretrained_path,
#         num_labels=num_labels
#     )
#     # Load checkpoint
#     checkpoint = torch.load(ckpt_path, map_location='cpu')
#     # Extract state dict
#     state_dict = checkpoint['state_dict']
#     for key in state_dict.keys():
#         print(key)
#     state_dict = {k.replace('model.', ''): v
#                   for k, v in state_dict.items()
#                   if k.startswith('model.')
#                   }
#
#     missing, unexpected = model.load_state_dict(state_dict, strict=False)
#     print(f"Missing keys: {missing}")
#     print(f"Unexpected keys: {unexpected}")
#     return model


# Update the load_model function to handle different checkpoint formats
def load_model(ckpt_path, pretrained_path=BERT_BASE_DIR, num_labels=3):
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=pretrained_path,
        num_labels=num_labels
    )

    if ckpt_path is None:
        return model

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['state_dict']

    # Remove all prefixes from keys (model., domain_models.domain.)
    new_state_dict = {}
    for key in list(state_dict.keys()):
        # Handle shared model prefix (model.)
        if key.startswith('model.'):
            new_key = key.replace('model.', '', 1)
        # Handle domain-specific prefix (domain_models.domain_name.)
        elif key.startswith('domain_models.'):
            parts = key.split('.')
            # Keep only keys after the domain name
            new_key = '.'.join(parts[2:])
        else:
            new_key = key

        # Only keep keys that exist in the model
        if any(new_key.startswith(prefix) for prefix in ['bert', 'classifier', 'dropout', 'fc']):
            new_state_dict[new_key] = state_dict[key]

    # Load filtered state dict
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    #print(f"Successfully loaded keys: {list(new_state_dict.keys())}")
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    return model


class LightBERTSeqClass(L.LightningModule):
    def __init__(self, domains, optimizer_config=None, shared_model=False, ckpt_path=None,
                 freeze_mode=None):
        super().__init__()
        self.optimizer_config = optimizer_config
        self.shared_model = shared_model  # New flag for shared model
        self.freeze_mode = freeze_mode
        pretrained_path = None
        if ckpt_path is None:
            pretrained_path = BERT_BASE_DIR


        if shared_model:
            # Single shared model for all domains
            if pretrained_path is not None:
                # load from pretrained
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    pretrained_model_name_or_path=pretrained_path,
                    num_labels=3
                )
            else:
                # load from checkpoint
                assert ckpt_path is not None
                self.model = load_model(ckpt_path, pretrained_path=BERT_BASE_DIR, num_labels=3)
            self._apply_freeze_mode(self.model)
        else:
            # Separate models per domain (original behavior)
            self.domain_models = torch.nn.ModuleDict()
            for domain in domains:
                if pretrained_path is not None:
                    model = AutoModelForSequenceClassification.from_pretrained(
                        pretrained_model_name_or_path=pretrained_path,
                        num_labels=3
                    )
                else:
                    assert ckpt_path is not None
                    model = load_model(ckpt_path, pretrained_path=BERT_BASE_DIR, num_labels=3)
                self._apply_freeze_mode(model)
                self.domain_models[domain] = model

        self.curr_domain = None
        self.print_trainable_params()


    def _apply_freeze_mode(self, model):
        """Apply freezing based on selected mode"""
        if self.freeze_mode is None:
            return  # No freezing, all parameters trainable

        if self.freeze_mode == "freeze_bert":
            # Freeze all parameters first
            for param in model.parameters():
                param.requires_grad = False

            # Unfreeze classifier layers
            classifier_layers = [
                'classifier', 'cls', 'fc', 'out_proj', 'qa_outputs'
            ]

            for name, param in model.named_parameters():
                if any(layer in name for layer in classifier_layers):
                    param.requires_grad = True
                    print(f"Unfreezing classifier layer: {name}")

        elif self.freeze_mode == "freeze_classifier":
            # Freeze classifier layers
            classifier_layers = [
                'classifier', 'cls', 'fc', 'out_proj', 'qa_outputs'
            ]

            for name, param in model.named_parameters():
                if any(layer in name for layer in classifier_layers):
                    param.requires_grad = False
                    print(f"Freezing classifier layer: {name}")


    def print_trainable_params(self):
        """Print trainable parameters count and names"""
        if self.shared_model:
            model = self.model
        else:
            # Use the first domain model
            model = self.domain_models[list(self.domain_models.keys())[0]]

        total_params = 0
        trainable_params = 0
        trainable_names = []

        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                trainable_names.append(name)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%})")
        print("Trainable layers:")
        for name in trainable_names[:10]:  # Show first 10 for brevity
            print(f"  - {name}")
        if len(trainable_names) > 10:
            print(f"  ... and {len(trainable_names) - 10} more layers")


    def reset_domain(self, domain):
        self.curr_domain = domain


    def forward(self, batch, return_dict=True):
        # Get the domain-specific model
        if self.shared_model:
            model = self.model
        else:
            model = self.domain_models[self.curr_domain]

        # Prepare inputs
        inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'token_type_ids': batch['token_type_ids'],
            'return_dict': return_dict
        }

        # Add labels only if they exist in the batch
        if 'labels' in batch:
            inputs['labels'] = batch['labels']

        return model(**inputs)


    def training_step(self, batch, batch_idx):
        outputs = self(batch) # ???
        loss = outputs.loss
        # Logging training loss
        self.log(
                f"train_loss/{self.curr_domain}",
                 loss,
                 on_step=True,
                 #on_epoch=True,
                 prog_bar=True
                 )
        return loss


    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        # Logging validation loss
        #self.log(f"val_loss/{self.curr_domain}", loss,
        #         on_epoch=True, prog_bar=True)

        # Optional: Log accuracy
        logits = outputs.logits
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch['labels']).float().mean()
        # Log validation metrics with unified keys
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        return loss


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Forward pass without labels
        outputs = self.forward(batch, return_dict=True)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        return preds


    def configure_optimizers(self):
        # TODO
        lr = self.optimizer_config.get('lr', 2e-5)
        opt_name = self.optimizer_config.get('opt_name', 'AdamW')
        assert opt_name in ['Adam', 'AdamW', 'SGD']

        # Get current model parameters
        if self.shared_model:
            model = self.model
        else:
            model = self.domain_models[self.curr_domain]

        if opt_name == 'Adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                weight_decay=0.01,
                lr=lr
            )
        elif opt_name == 'AdamW':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=0.01,
                eps=1e-6,
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr
            )

        total_steps = self.optimizer_config.get('total_steps', None)
        epochs = self.optimizer_config.get("epochs", None)
        warmup_ratio = self.optimizer_config.get('warmup_ratio', 0.01)
        #epochs = self.optimizer_config.get('num_train_epochs', None)
        #steps_per_epoch = self.optimizer_config.get('steps_per_epoch', None)

        if total_steps is None and hasattr(self, 'trainer') and self.trainer is not None:
            #TODO: need extra debugging
            total_steps = self.trainer.estimated_stepping_batches

        assert total_steps is not None, "Total number of steps cannot be determined!"

        assert total_steps is not None
        print(f"Total train steps: {total_steps}")

        steps_per_epoch = None
        if epochs is not None:
            assert total_steps % epochs == 0
            steps_per_epoch = total_steps // epochs

        print(f"Epochs {epochs} | Steps per epoch {steps_per_epoch}")

        if steps_per_epoch is not None:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,  # Peak learning rate
                total_steps=total_steps,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=warmup_ratio,
                anneal_strategy='linear'
            )
        else:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,  # Peak learning rate
                total_steps=total_steps,
                pct_start=warmup_ratio,
                anneal_strategy='linear'
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }


class LightDomainClassifier(L.LightningModule):
    def __init__(self, domain_model_paths, optimizer_config=None, freeze_domain_models=True):
        super().__init__()
        self.optimizer_config = optimizer_config
        self.mu = optimizer_config["mu"]
        self.freeze_domain_models = freeze_domain_models

        # Load domain-specific BERT models and freeze them
        self.domain_models = torch.nn.ModuleList()
        for path in domain_model_paths:
            model = load_model(path)
            # Always freeze classifier parameters since we don't use them
            self._freeze_classifier_parameters(model)
            if self.freeze_domain_models:
                model.requires_grad_(False)  # Freeze all parameters
            self.domain_models.append(model)

        # Get hidden size from first model
        hidden_size = self.domain_models[0].config.hidden_size

        # Initialize SINGLE weight vector (hidden_size)
        self.w = torch.nn.Parameter(torch.randn(hidden_size))


    def _freeze_classifier_parameters(self, model):
        """Freeze classifier parameters since they're not used in domain classification"""
        classifier_layers = [
            'classifier', 'cls', 'fc', 'out_proj', 'qa_outputs'
        ]

        for name, param in model.named_parameters():
            if any(layer in name for layer in classifier_layers):
                param.requires_grad = False
                print(f"Freezing classifier layer: {name}")


    def forward(self, batch, return_dict=True):
        # Collect features from all domain-specific models
        domain_features = []
        for model in self.domain_models:
            # Use checkpointing only for unfrozen models
            if not self.freeze_domain_models:
                pooled_output = torch.utils.checkpoint.checkpoint(
                    self._forward_model,
                    model,
                    batch['input_ids'],
                    batch['attention_mask'],
                    batch['token_type_ids'],
                    use_reentrant=False,
                )
                # outputs = model.bert(
                #     input_ids=batch['input_ids'],
                #     attention_mask=batch['attention_mask'],
                #     token_type_ids=batch['token_type_ids'],
                #     return_dict=True
                # )
                # pooled_output = outputs.pooler_output

            else:
                # For frozen models or during eval
                with torch.set_grad_enabled(False):
                    outputs = model.bert(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        token_type_ids=batch['token_type_ids'],
                        return_dict=True
                    )
                    pooled_output = outputs.pooler_output
            domain_features.append(pooled_output)

        # Stack features: (batch_size, num_domains, hidden_size)
        features = torch.stack(domain_features, dim=1)

        # Compute logits: w^T * f(x, k) for each domain k
        logits = torch.einsum('h,bkh->bk', self.w, features)
        return logits


    def _forward_model(self, model, input_ids, attention_mask, token_type_ids):
        """Helper function for gradient checkpointing"""
        outputs = model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        return outputs.pooler_output


    def training_step(self, batch, batch_idx):
        logits = self(batch)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, batch['labels'])

        # L2 regularization: μ||w||^2
        reg_loss = self.mu * torch.sum(self.w ** 2)

        total_loss = loss + reg_loss
        self.log("train_loss", total_loss, prog_bar=True)

        return total_loss

    # def on_after_backward(self):
    #     """CORRECT place to check gradients"""
    #     print("\n=== GRADIENT ANALYSIS ===")
    #     if self.global_step >= 5:
    #         return
    #     # 1. Check w gradients
    #     if self.w.grad is None:
    #         print("❌ w.grad is None")
    #     else:
    #         w_norm = self.w.grad.norm().item()
    #         print(f"✅ w.grad exists | norm: {w_norm}")
    #         self.log("grad_norm/w", w_norm, prog_bar=True)
    #
    #     # 2. Check domain model gradients
    #     for i, model in enumerate(self.domain_models):
    #         total_norm = 0
    #         total_params = 0
    #         grad_params = 0
    #
    #         for name, param in model.named_parameters():
    #             total_params += 1
    #             if param.grad is not None:
    #                 grad_params += 1
    #                 total_norm += param.grad.norm().item() ** 2
    #
    #         total_norm = total_norm ** 0.5 if grad_params > 0 else 0
    #         print(f"Domain {i} | Grad params: {grad_params}/{total_params} | Norm: {total_norm:.6f}")
    #         self.log(f"grad_norm/domain_{i}", total_norm, prog_bar=True)


    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        loss = F.cross_entropy(logits, batch['labels'])
        reg_loss = self.mu * torch.sum(self.w ** 2)
        total_loss = loss + reg_loss

        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch['labels']).float().mean()

        self.log("val_loss", total_loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return total_loss


    def configure_optimizers(self):
        #params = [self.w]
        #for model in self.domain_models:
        #    params += list(model.parameters())
        trainable_params = [self.w]
        if not self.freeze_domain_models:
            for model in self.domain_models:
                trainable_params += [p for p in model.parameters() if p.requires_grad]

        lr = self.optimizer_config.get('lr', 2e-5)
        optimizer = torch.optim.AdamW(
            trainable_params,
            #lr=lr,
            weight_decay=0,
            eps=1e-6,
        )

        total_steps = self.optimizer_config.get('total_steps', None)
        epochs = self.optimizer_config.get("epochs", None)
        warmup_ratio = self.optimizer_config.get('warmup_ratio', 0.01)
        #epochs = self.optimizer_config.get('num_train_epochs', None)
        #steps_per_epoch = self.optimizer_config.get('steps_per_epoch', None)

        if total_steps is None and hasattr(self, 'trainer') and self.trainer is not None:
            #TODO: need extra debugging
            total_steps = self.trainer.estimated_stepping_batches

        assert total_steps is not None, "Total number of steps cannot be determined!"

        print(f"Total train steps: {total_steps}")

        steps_per_epoch = None
        if epochs is not None:
            assert total_steps % epochs == 0
            steps_per_epoch = total_steps // epochs

        print(f"Epochs {epochs} | Steps per epoch {steps_per_epoch}")

        if steps_per_epoch is not None:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,  # Peak learning rate
                total_steps=total_steps,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=warmup_ratio,
                anneal_strategy='linear'
            )
        else:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,  # Peak learning rate
                total_steps=total_steps,
                pct_start=warmup_ratio,
                anneal_strategy='linear'
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }


# Modified training function to return final validation metrics
def seq_pair_classif_training(data_dir, domains, optimizer_config, train_params,
                              logger=None, train_ratio=0.05, val_ratio=0.3,
                              grad_accum_steps=1, ckpt_path=None, freeze_mode=None,
                              ):
    # train a single model
    batch_size, max_steps = train_params['batch_size'], train_params['num_train_steps']
    max_epochs = train_params.get('num_train_epochs', -1)

    # Determine if using shared model
    shared_model = (len(domains) > 1)

    # Pass shared_model flag to model
    task_pl = LightBERTSeqClass(
        domains,
        optimizer_config=optimizer_config,
        shared_model=shared_model,
        ckpt_path=ckpt_path,
        freeze_mode=freeze_mode
    )


    # Rest of the function remains unchanged
    dm = MNLIDataModule(data_dir, domains,
                        ['entailment', 'contradiction', 'neutral'])

    dm.prepare_data()
    dm.reset_batch_size(batch_size)

    if shared_model:
        dm.setup('fit', 'seq_pair_classif', "all")
    else:
        dm.setup('fit', 'seq_pair_classif', domains[0])
        task_pl.reset_domain(domains[0])

    # Create checkpoint directory
    checkpoint_dir = os.path.join(logger.log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor='val_acc',  # Metric to monitor
        mode='max',  # Maximize validation accuracy
        save_top_k=1,  # Save only the best model
        filename='best-{epoch}-{val_acc:.2f}',
        save_last=True,  # Also save last checkpoint
        auto_insert_metric_name=False  # Cleaner filename
    )
    if max_epochs == -1:
        trainer = L.Trainer(
            accelerator="auto",
            logger=logger,
            gradient_clip_val=1.0,
            accumulate_grad_batches=grad_accum_steps,
            limit_train_batches=train_ratio,
            limit_val_batches=val_ratio,
            max_steps=max_steps,
            enable_progress_bar=True,
            log_every_n_steps=100, #TODO
            callbacks=[checkpoint_callback]
            #callbacks=[L.pytorch.callbacks.ModelCheckpoint(monitor='val_acc', mode='max')]
        )
    else:
        # epoch-based training is the priority
        trainer = L.Trainer(
            accelerator="auto",
            logger=logger,
            gradient_clip_val=1.0,
            accumulate_grad_batches=grad_accum_steps,
            limit_train_batches=train_ratio,
            limit_val_batches=val_ratio,
            max_epochs=max_epochs,
            enable_progress_bar=True,
            log_every_n_steps=100, #TODO
            callbacks=[checkpoint_callback]
        )


    # Add hyperparameters to TensorBoard
    if logger:
        logger.log_hyperparams({
            "warmup_ratio": optimizer_config["warmup_ratio"],
            "lr": optimizer_config['lr'],
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "num_train_steps": max_steps,
            "opt_name": optimizer_config.get('opt_name', 'AdamW'),
            "domain": "-".join(domains),
            "ckpt_path": ckpt_path,
            "freeze_mode": freeze_mode
        })

    trainer.fit(task_pl, dm.train_dataloader(), dm.val_dataloader())

    # Return final validation metrics
    val_metrics = trainer.validate(task_pl, dataloaders=dm.val_dataloader())

    # After training, get best model path
    best_model_path = checkpoint_callback.best_model_path

    # Return metrics AND best model path
    val_metrics = val_metrics[0] if val_metrics else {}
    val_metrics['best_model_path'] = best_model_path
    return val_metrics


def domain_classif_training(data_dir, domains, optimizer_config, train_params,
                            domain_model_paths, logger=None, train_ratio=0.05,
                            val_ratio=0.3, grad_accum_steps=1, freeze_domain_models=True):
    """
    TODO: test code sanity and tuner
    Train a domain classifier using pre-trained domain-specific models

    Args:
        domain_model_paths: List of paths to pre-trained domain-specific models
        Other args same as seq_pair_classif_training
    """
    batch_size, max_steps = train_params['batch_size'], train_params['num_train_steps']
    max_epochs = train_params.get('num_train_epochs', -1)

    # Create domain classifier model
    domain_classifier = LightDomainClassifier(
        domain_model_paths=domain_model_paths,
        optimizer_config=optimizer_config,
        freeze_domain_models=freeze_domain_models
    )

    # Setup data module for domain classification
    dm = MNLIDataModule(data_dir, domains, ['entailment', 'contradiction', 'neutral'])
    dm.prepare_data()
    dm.reset_batch_size(batch_size)
    dm.setup('fit', 'domain_classif', "all")  # Use all domains for domain classification

    # Create checkpoint directory
    checkpoint_dir = os.path.join(logger.log_dir, "checkpoints") if logger else "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor='val_acc',  # Metric to monitor
        mode='max',  # Maximize validation accuracy
        save_top_k=1,  # Save only the best model
        filename='domain_classif-best-{epoch}-{val_acc:.2f}',
        save_last=True,  # Also save last checkpoint
        auto_insert_metric_name=False
    )

    # Configure trainer
    if max_epochs == -1:

        trainer = L.Trainer(
            accelerator="auto",
            logger=logger,
            gradient_clip_val=1.0,
            accumulate_grad_batches=grad_accum_steps,
            limit_train_batches=train_ratio,
            limit_val_batches=val_ratio,
            max_steps=max_steps,
            enable_progress_bar=True,
            log_every_n_steps=10,  # TODO: set values back to original
            val_check_interval=50,
            callbacks=[checkpoint_callback]
            # callbacks=[L.pytorch.callbacks.ModelCheckpoint(monitor='val_acc', mode='max')]
        )

    else:
        trainer = L.Trainer(
            accelerator="auto",
            logger=logger,
            gradient_clip_val=1.0,
            accumulate_grad_batches=grad_accum_steps,
            limit_train_batches=train_ratio,
            limit_val_batches=val_ratio,
            max_epochs=max_epochs,
            enable_progress_bar=True,
            log_every_n_steps=100,
            val_check_interval=100,
            callbacks=[checkpoint_callback]
        )

    # Add hyperparameters to TensorBoard
    if logger:
        logger.log_hyperparams({
            "lr": optimizer_config.get('lr', 1e-3),
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "num_domains": len(domains),
            "mu": optimizer_config.get('mu', 0)  # Regularization strength
        })

    # Print out a sample batch
    sample_batch = next(iter(dm.train_dataloader()))
    print("Sample batch keys:", sample_batch.keys())
    print("Input IDs shape:", sample_batch['input_ids'].shape)
    print("Labels:", sample_batch['labels'][:10])

    # Train the model
    trainer.fit(domain_classifier, dm.train_dataloader(), dm.val_dataloader())

    # Return final validation metrics
    val_metrics = trainer.validate(domain_classifier, dataloaders=dm.val_dataloader())

    # After training, get best model path
    best_model_path = checkpoint_callback.best_model_path

    # Return metrics AND best model path
    val_metrics = val_metrics[0] if val_metrics else {}
    val_metrics['best_model_path'] = best_model_path
    return val_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=False, type=str, default="seq_pair_classif")
    parser.add_argument("--data_dir", required=False,
                        help="Path to dataset directory",
                        default=r'C:\Users\tangc\PycharmProjects\domain-adaptation\datasets')
    parser.add_argument("--domain_models_dir", type=str, default=None,
                        help="Directory containing trained domain-specific models")
    #TODO: beta
    parser.add_argument("--single_bert_path", type=str, default=None,
                        help="Path to a single pre-trained BERT model to be used as k copies for k domains")
    parser.add_argument("--unfreeze_domain_models", action="store_false", default=True)
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Path to checkpoint directory")
    parser.add_argument("--freeze_mode", type=str, default=None,
                        choices=["freeze_bert", "freeze_classifier"],
                        help="Freezing mode: none, freeze_bert, or freeze_classifier")
    parser.add_argument("--domain", type=str, required=True, help="Target domain for tuning")
    parser.add_argument("--port", type=int, default=6006, help="TensorBoard port")
    parser.add_argument("--warmup_ratio", type=float, default=0.06, help="Warmup ratio")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--mu", type=float, required=False,
                        default=0.0,
                        help="regularization hp for domain classification")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--train_ratio", type=float, default=1.0)
    parser.add_argument("--num_train_steps", type=int, default=200)
    parser.add_argument("--num_train_epochs", type=int)

    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = args.data_dir
    lr = args.lr
    batch_size = args.batch_size
    effective_batch_size = batch_size * args.grad_accum_steps
    if not args.num_train_epochs:
        version = f"warmup_ratio_{args.warmup_ratio}_lr_{lr}_ebs_{effective_batch_size}_steps_{args.num_train_steps}"
    else:
        version = f"warmup_ratio_{args.warmup_ratio}_lr_{lr}_ebs_{effective_batch_size}_epochs_{args.num_train_epochs}"
    log_dir = f"logs/tuning_{args.domain}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    result_file = os.path.join(log_dir, f"tuning_results_{args.domain}.json")
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=args.domain,
        version=version,
        default_hp_metric=False  # We'll add our own
    )

    # set training configs
    domain = args.domain.split("-")
    train_params = {
        "batch_size": batch_size,
        "num_train_steps": args.num_train_steps
        }
    optimizer_config = {
            "warmup_ratio": args.warmup_ratio,
            "lr": lr,
        }

    if args.num_train_epochs:
        train_params['num_train_epochs'] = args.num_train_epochs
        optimizer_config["epochs"] = args.num_train_epochs

    if args.task == "seq_pair_classif":
        val_metrics = seq_pair_classif_training(
            data_dir=data_dir,
            domains=domain,
            optimizer_config=optimizer_config,
            train_params=train_params,
            logger=logger,
            train_ratio=args.train_ratio,
            val_ratio=1.0,
            grad_accum_steps=args.grad_accum_steps,
            ckpt_path=args.ckpt_path,
            freeze_mode=args.freeze_mode,
        )

    else:
        # get all model address from given folder
        # TODO： beta
        if args.single_bert_path:
            # Create k copies of the same model path
            domain_model_paths = [args.single_bert_path] * len(domain)
            print(f"Using single BERT model at {args.single_bert_path} for all {len(domain)} domains")
        else:
            # Original behavior - collect domain-specific models
            domain_model_paths = collect_domain_model_paths(domain, args.domain_models_dir)
            print(f"Using domain-specific models: {domain_model_paths}")

        val_metrics = domain_classif_training(
            data_dir,
            domain,
            optimizer_config,
            train_params,
            domain_model_paths,
            logger=logger,
            train_ratio=args.train_ratio,
            val_ratio=1.0,
            grad_accum_steps=args.grad_accum_steps,
            freeze_domain_models=args.unfreeze_domain_models,
        )


    result = {
        "task": args.task,
        "lr": args.lr,
        "mu": args.mu,
        "grad_accum_steps": args.grad_accum_steps,
        "batch_size": args.batch_size,
        "train_ratio": args.train_ratio,
        "num_train_steps": args.num_train_steps,
        "num_train_epochs": args.num_train_epochs,
        "version": version,
        "ckpt_path": args.ckpt_path,
        "freeze_mode": args.freeze_mode,
        **val_metrics  # Include all validation metrics
    }

    with open(result_file, "w") as f:
        json.dump(result, f)

    print(val_metrics)