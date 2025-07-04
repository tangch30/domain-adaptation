import lightning as L
from trainer import LightBERTSeqClass
from data_loader import MNLIDataModule
from torch.utils.data import DataLoader


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


if __name__ == '__main__':
    test_lightning_module()






