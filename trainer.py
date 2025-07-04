"""Train domain-wise predictor and task predictor"""
## DeepSeek is used during coding

# input: data loader (domain name), pre-trained bert
# save fine-tuned bert
# should i install HF-transformer locally
# load bert-model from HF-transformer github

import lightning as L
import torch
import argparse, os, json
import random
from datetime import datetime
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoModel, AutoModelForSequenceClassification
from transformers import BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from data_loader import MNLIDataModule, mnli_preprocess

#TODO: Change validation metric, check log frequency (you haven't written any data to event files -tfb)


class LightBERTSeqClass(L.LightningModule):
    def __init__(self, domains, optimizer_config=None, shared_model=False):
        super().__init__()
        self.optimizer_config = optimizer_config
        self.shared_model = shared_model  # New flag for shared model

        if shared_model:
            # Single shared model for all domains
            self.model = AutoModelForSequenceClassification.from_pretrained(
                r'C:\Users\tangc\PycharmProjects\domain-adaptation\bert-base-uncased-cache',
                num_labels=3
            )
        else:
            # Separate models per domain (original behavior)
            self.domain_models = torch.nn.ModuleDict()
            for domain in domains:
                model = AutoModelForSequenceClassification.from_pretrained(
                    r'C:\Users\tangc\PycharmProjects\domain-adaptation\bert-base-uncased-cache',
                    num_labels=3
                )
                self.domain_models[domain] = model

        self.curr_domain = None


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


    # def forward(self, batch, return_dict=True):
    #     #input_ids, attention_mask, token_type_ids, target = inputs
    #     #inputs, target = batch
    #     #input_ids, attention_mask, token_type_ids = inputs
    #     input_ids = batch['input_ids']
    #     attention_mask = batch['attention_mask']
    #     token_type_ids = batch['token_type_ids']
    #     labels = batch['labels']
    #     #print("Input_ids: ", input_ids)
    #     #return self.seq_classifiers[self.curr_domain](input_ids, attention_mask,
    #     #                                              token_type_ids, labels=labels,
    #     #                                              return_dict=return_dict)
    #     # Pass through shared base model
    #     outputs = self.base_model(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #           token_type_ids=token_type_ids,
    #         return_dict=return_dict
    #     )
    #
    #     # Get pooled output (CLS token representation)
    #     pooled_output = outputs.pooler_output
    #
    #     # Pass through domain-specific classifier head
    #     logits = self.classifier_heads[self.curr_domain](pooled_output)
    #
    #     # Calculate loss
    #     loss_fct = torch.nn.CrossEntropyLoss()
    #     loss = loss_fct(logits.view(-1, 3), labels.view(-1))
    #
    #     # Return in standard Hugging Face output format
    #     if return_dict:
    #         return SequenceClassifierOutput(
    #             loss=loss,
    #             logits=logits,
    #             hidden_states=outputs.hidden_states,
    #             attentions=outputs.attentions
    #         )
    #     return (loss, logits)


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


class LightBERTDomainClass(L.LightningModule):
    def __init__(self, domains):
        super().__init__()
        self.domain_dict = dict()
        for idx, domain in enumerate(domains):
            self.domain_dict[domain] = idx
        self.domain_classifiers = dict()
        for domain in domains:
            self.domain_classifiers[domain] = AutoModelForSequenceClassification.from_pretrained(r'C:\Users\tangc\PycharmProjects\domain-adaptation\bert-base-uncased-cache')

    def reset_domain(self, domain):
        self.curr_domain = domain

    def forward(self, inputs, targets):
        #TODO: inputs, target <--- batch suggested by DeepSeek
        input_ids, attention_mask, token_type_ids = inputs
        return self.domain_classifiers[self.curr_domain](input_ids, attention_mask,
                                                      token_type_ids, labels=targets)

    def training_step(self, batch_dict, batch_idx):
        # The dataloader will load a batch for each domain
        # extend batch: determine one domain and sample from any of other domains
        # randomly shuffle postive and negative domain examples
        #outputs = self(batch) #TODO ?
        #loss = outputs.loss #TODO: check
        this_batch = batch_dict[self.curr_domain]
        other_domain = None
        rint = random.randint(1, len(self.domain_classifiers.keys()))
        count = 1
        for domain in self.domain_classifiers.keys():
            if domain == self.curr_domain:
                continue
            if count == rint:
                other_domain = domain
                break
            count += 1
        assert other_domain is not None and other_domain != self.curr_domain
        other_batch = batch_dict[other_domain]
        this_target = torch.tensor([self.domain_dict[self.curr_domain]] * len(this_batch))
        other_target = torch.tensor([self.domain_dict[other_domain]] * len(other_batch))
        batch = torch.cat([this_batch, other_batch], axis=0)
        batch_y = torch.cat([this_target, other_target], axis=0)
        idx = torch.randperm(len(batch))
        batch = batch[idx, :]
        batch_y = batch_y[idx, :]
        outputs = self(batch, batch_y)
        loss = outputs.loss
        return loss


    def configure_optimizers(self):
        return torch.optim.SGD(self.domain_classifiers[self.curr_domain].parameters(), lr=0.1) #TODO: configure optimizer


class BaseLearner(object):
    def __init__(self, data_dir, domains, sentiments,
                 seq_classif_params=None,
                 domain_classif_params=None):
        self.data_dir = data_dir
        self.domains = domains
        #self.sentiments = sentiments
        self.seq_classif_params = seq_classif_params

        #self.domain_pl = None
        self.task_pl = LightBERTSeqClass(domains)
        #self.domain_predictors = None # TODO
        #self.task_predictors = None # TODO: get the automodel after training

        self.dm = MNLIDataModule(data_dir, domains, sentiments)
        self.dm.prepare_data()


    def seq_pair_classif_training(self, domain, logger):
        train_params = self.seq_classif_params[domain]
        batch_size, max_epochs = train_params['batch_size'], train_params['max_epochs']
        self.dm.reset_batch_size(batch_size)
        self.dm.setup('fit', 'seq_pair_classif', domain)
        self.task_pl.reset_domain(domain)
        trainer = L.Trainer(
            logger=logger,
            #accelerator="gpu",  # Explicitly use GPU
            #devices=1,  # Use first available GPU
            #strategy="auto",  # Automatically select best strategy
            #precision="16-true",  # Mixed precision for efficiency
            gradient_clip_val=1.0,  # Prevent exploding gradients
            max_epochs=max_epochs,  # Prevent infinite training
            enable_progress_bar=True,
            log_every_n_steps=100,
            #detect_anomaly=True  # Helpful for debugging
        ) # TODO readthedoc: args and where is model kept
        trainer.fit(self.task_pl, self.dm.train_dataloader(), self.dm.val_dataloader())


    #def seq_pair_classif_predict(self):
    #    pass

    def domain_training(self, domain):
        # at each step load a pair of data from domain A and not domain A
        pass

    def clean_up(self):
        # also remove tmp files
        pass


# Modified training function to return final validation metrics
def seq_pair_classif_training(data_dir, domains, optimizer_config, train_params,
                              logger=None, train_ratio=0.05, val_ratio=0.3,
                              grad_accum_steps=1
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
        shared_model=shared_model
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
        task_pl.reset_domain(domain)

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
            logger=logger,
            gradient_clip_val=1.0,
            accumulate_grad_batches=grad_accum_steps,
            limit_train_batches=train_ratio,
            limit_val_batches=val_ratio,
            max_steps=max_steps,
            enable_progress_bar=True,
            log_every_n_steps=100,
            callbacks=[checkpoint_callback]
            #callbacks=[L.pytorch.callbacks.ModelCheckpoint(monitor='val_acc', mode='max')]
        )
    else:
        # epoch-based training is the priority
        trainer = L.Trainer(
            logger=logger,
            gradient_clip_val=1.0,
            accumulate_grad_batches=grad_accum_steps,
            limit_train_batches=train_ratio,
            limit_val_batches=val_ratio,
            max_epochs=max_epochs,
            enable_progress_bar=True,
            log_every_n_steps=100,
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
            "domain": "-".join(domains)
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


######## call after training...from DeepSeek
"""
import json
from datetime import datetime


def save_metadata_for_reproducibility(config, metrics):
    # for code reproducibility
    repo = git.Repo(search_parent_directories=True)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "git": {
            "commit": repo.head.commit.hexsha,
            "branch": repo.active_branch.name,
            "dirty": repo.is_dirty()
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "libraries": {
                "torch": torch.__version__,
                "transformers": transformers.__version__
            }
        },
        "config": vars(config) if hasattr(config, '__dict__') else config,
        "metrics": metrics
    }

    os.makedirs(config.output_dir, exist_ok=True)
    with open(f"{config.output_dir}/run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=False,
                        help="Path to dataset directory",
                        default=r'C:\Users\tangc\PycharmProjects\domain-adaptation\datasets')
    parser.add_argument("--domain", type=str, required=True, help="Target domain for tuning")
    parser.add_argument("--port", type=int, default=6006, help="TensorBoard port")
    parser.add_argument("--warmup_ratio", type=float, default=0.06, help="Warmup ratio")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Number of gradient accumulation steps")
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


    val_metrics = seq_pair_classif_training(
        data_dir=data_dir,
        domains=domain,
        optimizer_config=optimizer_config,
        train_params=train_params,
        logger=logger,
        train_ratio=1.0,
        val_ratio=1.0,
        grad_accum_steps=args.grad_accum_steps
    )

    result = {
        "lr": args.lr,
        "grad_accum_steps": args.grad_accum_steps,
        "batch_size": args.batch_size,
        "num_train_steps": args.num_train_steps,
        "num_train_epochs": args.num_train_epochs,
        "version": version,
        **val_metrics  # Include all validation metrics
    }

    with open(result_file, "w") as f:
        json.dump(result, f)

    print(val_metrics)