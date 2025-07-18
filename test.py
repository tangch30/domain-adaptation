import lightning as L
import os, json, datetime
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from trainer import LightBERTSeqClass, domain_classif_training, collect_domain_model_paths
from data_loader import mnli_preprocess, MNLIDataModule, MNLIDataset
from torch.utils.data import DataLoader, ConcatDataset


def test_lightning_module():
    """Test function for LightBERTSeqClass"""
    # Setup test environment
    data_dir = r'C:\Users\tangc\PycharmProjects\domain-adaptation\datasets'
    domains = ['telephone']
    sentiments = ['entailment', 'contradiction', 'neutral']

    # Create data module
    dm = MNLIDataModule(data_dir, domains, sentiments)
    dm.prepare_data()
    dm.reset_batch_size(4)
    dm.setup('fit', 'seq_pair_classif', 'telephone')

    # Create model
    optimizer_config = {
        'lr': 2e-5,
        'warmup_ratio': 0.06,
        'total_steps': 100,
        'epochs': 1
    }
    model = LightBERTSeqClass(domains, optimizer_config, shared_model=True)
    model.reset_domain('telephone')

    # Create trainer
    trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=5,  # Only 5 batches for testing
        limit_val_batches=2,  # Only 2 batches for validation
        enable_progress_bar=True,
        logger=False,
        enable_checkpointing=False
    )

    # Test training
    print("\n=== Testing Training ===")
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

    # Test validation
    print("\n=== Testing Validation ===")
    val_results = trainer.validate(model, dataloaders=dm.val_dataloader())
    print("Validation results:", val_results)

    # Test prediction
    print("\n=== Testing Prediction ===")
    dm.setup('predict', 'seq_pair_classif', 'telephone')
    predictions = trainer.predict(model, dataloaders=dm.predict_dataloader())
    print(f"Predicted {len(predictions)} batches")
    print("First batch predictions:", predictions[0])

    print("\nAll tests completed successfully!")



def seq_pair_classif_predict_mnli(data_dir, domains, ckpt_path=None):
    # Pass shared_model flag to model
    task_pl = LightBERTSeqClass(
        domains,
        ckpt_path=ckpt_path,
    )

    # Rest of the function remains unchanged
    dm = MNLIDataModule(data_dir, domains,
                        ['entailment', 'contradiction', 'neutral'])
    dm.prepare_data()
    dm.reset_batch_size(16)
    dm.setup('fit', 'seq_pair_classif', domains[0])

    task_pl.reset_domain(domains[0])

    trainer = L.Trainer(
            accelerator="auto",
            enable_progress_bar=True
    )


    # Return final validation metrics
    val_metrics = trainer.validate(task_pl, dataloaders=dm.val_dataloader())

    # Return metrics AND best model path
    #val_metrics = val_metrics[0] if val_metrics else {}
    return val_metrics


def test_domain_classification():
    print("\n" + "=" * 50)
    print("Testing Domain Classification Data Loading")
    print("=" * 50)

    # Setup test environment
    domains = ['telephone', 'oup']
    sentiments = ["entailment", "contradiction", "neutral"]
    file_dir = r'C:\Users\tangc\PycharmProjects\domain-adaptation\datasets'

    # Create domain dictionary mapping
    domain_dict = {domain: idx for idx, domain in enumerate(domains)}

    # 1. Test data preprocessing
    print("\nTesting mnli_preprocess for domain classification...")
    mnli_preprocess(file_dir, domains)

    # Verify temporary files were created
    for domain in domains:
        for suffix in ['train', 'dev_matched', 'dev_mismatched', 'test_matched', 'test_mismatched']:
            filename = f"{domain}.{suffix}.tmp"
            file_path = os.path.join(file_dir, filename)
            assert os.path.exists(file_path), f"Preprocessing failed: {filename} not created"
            print(f"Verified: {filename} exists")

    # 2. Test MNLIDataset for domain classification
    print("\nTesting MNLIDataset for domain classification...")

    # Test training dataset
    train_dataset = MNLIDataset(
        file_dir,
        domains,
        domain_dict,
        {},  # Sentiment dict not used
        'train',
        'domain_classif'
    )
    print(f"Domain train set size: {len(train_dataset)}")

    # Verify first item has domain label
    first_item = train_dataset[0]
    print(first_item)
    assert 'labels' in first_item, "Domain label missing in training item"
    assert first_item['labels'] in domain_dict.values(), "Invalid domain label value"
    print(f"First training item domain label: {first_item['labels']}")

    # Verify tokenization
    for key in ['input_ids', 'attention_mask', 'token_type_ids']:
        assert key in first_item, f"Missing key in item: {key}"
        assert len(first_item[key]) > 0, f"Empty {key} in item"
    print("Tokenization verified")

    # Test validation dataset
    val_dataset = MNLIDataset(
        file_dir,
        domains,
        domain_dict,
        {},
        'dev_matched',
        'domain_classif'
    )
    print(f"Domain validation set size: {len(val_dataset)}")

    # Test test dataset (unlabeled)
    test_dataset = MNLIDataset(
        file_dir,
        domains,
        domain_dict,
        {},
        'test_matched',
        'domain_classif',
        has_labels=False
    )
    assert 'labels' not in test_dataset[0], "Labels should not be present in test set"
    print("Unlabeled test set verified")

    # 3. Test MNLIDataModule for domain classification
    print("\nTesting MNLIDataModule for domain classification...")

    # Create data module
    dm = MNLIDataModule(file_dir, domains, sentiments)
    dm.prepare_data()  # Should be idempotent
    dm.reset_batch_size(2)

    # Test setup for training
    dm.setup('fit', 'domain_classif', "all")

    # Verify dataset sizes
    assert len(dm.train) > 0, "Training dataset is empty"
    assert len(dm.valid) > 0, "Validation dataset is empty"
    print(f"Training set size: {len(dm.train)}")
    print(f"Validation set size: {len(dm.valid)}")

    # Test training dataloader
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print("\nTraining batch structure:")
    for key, value in batch.items():
        print(f"{key}: shape {value.shape}")

    assert 'input_ids' in batch
    assert 'attention_mask' in batch
    assert 'token_type_ids' in batch
    assert 'labels' in batch
    assert batch['labels'].shape[0] == 2, "Batch size incorrect"

    # Test validation dataloader
    val_loader = dm.val_dataloader()
    val_batch = next(iter(val_loader))
    assert 'labels' in val_batch, "Labels missing in validation batch"

    # Test setup for prediction
    dm.setup('predict', 'domain_classif', "all")
    predict_loader = dm.predict_dataloader()
    predict_batch = next(iter(predict_loader))

    assert 'input_ids' in predict_batch
    assert 'attention_mask' in predict_batch
    assert 'token_type_ids' in predict_batch
    assert 'labels' not in predict_batch, "Labels should not be in prediction set"
    print("Prediction dataloader verified")

    # 4. Test domain label assignment
    print("\nTesting domain label assignment...")
    domain_label_counts = {domain_id: 0 for domain_id in domain_dict.values()}

    # Check training set domain distribution
    for i in range(len(train_dataset)):
        item = train_dataset[i]
        domain_label_counts[item['labels']] += 1

    print("Domain distribution in training set:")
    for domain, domain_id in domain_dict.items():
        count = domain_label_counts[domain_id]
        print(f"  {domain}: {count} examples")
        #assert count > 0, f"No examples found for domain: {domain}"

    # 5. Test ConcatDataset behavior
    print("\nTesting ConcatDataset behavior...")
    matched_dataset = MNLIDataset(
        file_dir, domains, domain_dict, {}, 'dev_matched', 'domain_classif'
    )
    mismatched_dataset = MNLIDataset(
        file_dir, domains, domain_dict, {}, 'dev_mismatched', 'domain_classif'
    )
    concat_dataset = ConcatDataset([matched_dataset, mismatched_dataset])

    assert len(concat_dataset) == len(matched_dataset) + len(mismatched_dataset)
    print(f"ConcatDataset size verified: {len(concat_dataset)}")

    # Verify items from both datasets are present
    first_matched = matched_dataset[0]
    last_mismatched = mismatched_dataset[-1]

    concat_item1 = concat_dataset[0]
    concat_item2 = concat_dataset[len(matched_dataset)]

    assert torch.equal(
        torch.tensor(first_matched['input_ids']),
        torch.tensor(concat_item1['input_ids'])
    ), "ConcatDataset first item mismatch"

    # Verify last item from second dataset (correct boundary)
    concat_last_idx = len(matched_dataset) + len(mismatched_dataset) - 1
    concat_last_item = concat_dataset[concat_last_idx]

    last_mismatched = mismatched_dataset[-1]
    assert torch.equal(
        torch.tensor(last_mismatched['input_ids']),
        torch.tensor(concat_last_item['input_ids'])
    ), "ConcatDataset last item mismatch"

    print("ConcatDataset tests passed.")

    print("All domain classification tests passed successfully!")


def test_domain_classifier(data_dir, domain, domain_models_dir, logger=None):
    """Test domain classifier training with fixed parameters"""
    print(f"\n=== Testing domain classifier for {domain} ===")

    # Fixed hyperparameters for test
    optimizer_config = {
        "lr": 1e-3,
        "mu": 0.01  # Regularization strength
    }
    train_params = {
        "batch_size": 16,
        "num_train_epochs": 1  # Short test run
    }

    # Collect domain model paths
    domains = domain.split("-")
    domain_model_paths = collect_domain_model_paths(domains, domain_models_dir)

    # Run training
    val_metrics = domain_classif_training(
        data_dir,
        domains,
        optimizer_config,
        train_params,
        domain_model_paths,
        logger=logger,
        train_ratio=0.01,  # Small subset
        val_ratio=1.0,
        grad_accum_steps=1
    )

    print(f"Domain classifier test complete | Val Acc: {val_metrics.get('val_acc', 'N/A'):.4f}")
    return val_metrics


if __name__ == '__main__':
    #test_lightning_module()
    #data_dir = r'C:\Users\tangc\PycharmProjects\domain-adaptation\datasets'
    #from tuner import BERT_ALL_FT_PATH
    #val_metrics = seq_pair_classif_predict_mnli(data_dir, ["slate"], ckpt_path=BERT_ALL_FT_PATH)
    #print(f"val: {val_metrics}")


    # Add domain classification tests
    # Run domain classifier test
    data_dir = r'C:\Users\tangc\PycharmProjects\domain-adaptation\datasets'
    domain_models_dir = r'C:\Users\tangc\PycharmProjects\domain-adaptation\logs\domain_models'
    domain = ["fiction", "telephone", "government", "slate", "travel"]
    if not domain_models_dir:
        raise ValueError("--domain_models_dir required for domain_classif task")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/domain_test_all_domains_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    # Minimal logger
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=domain,
        version="test_run",
        default_hp_metric=False
    )

    # Run test
    results = test_domain_classifier(
        data_dir,
        domain,
        domain_models_dir,
        logger=logger
    )

    # Save results
    with open(os.path.join(log_dir, "test_results.json"), "w") as f:
        json.dump(results, f)

    print(f"Domain classifier test results: {results}")
    print(f"Logs saved to: {log_dir}")





