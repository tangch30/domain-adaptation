"""
Note: created largely by DeepSeek
"""

import argparse
import os
import itertools
import pandas as pd
import torch
import datetime
import traceback
import sys
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
from data_loader import MNLIDataModule
from trainer import LightBERTSeqClass


def debug_training(data_dir, domains, config):
    """Run training with detailed debugging information"""
    domain_str = "-".join(domains) if isinstance(domains, list) else domains
    print("\n" + "=" * 80)
    print(f"🚨 DEBUGGING TRIAL: {config}")
    print(f"Domains: {domain_str}")
    print("=" * 80)

    try:
        # 1. Test data module initialization
        print("\n🔍 Testing data module initialization...")
        dm = MNLIDataModule(data_dir, domains, ['entailment', 'contradiction', 'neutral'])
        dm.prepare_data()
        dm.reset_batch_size(config['batch_size'])

        # Determine if we're using shared model (multiple domains)
        shared_model = len(domains) > 1
        if shared_model:
            print("🔧 Setting up for shared model (multiple domains)")
            dm.setup('fit', 'seq_pair_classif', "all")
        else:
            dm.setup('fit', 'seq_pair_classif', domains[0])

        print("✅ Data module initialized successfully!")

        # 2. Test model initialization
        print("\n🔍 Testing model initialization...")
        task_pl = LightBERTSeqClass(domains,
                                    optimizer_config={
                                        "lr": config["lr"],
                                        "opt_name": config["opt_name"]
                                    },
                                    shared_model=shared_model)  # Pass shared_model flag

        # For single domain, set current domain
        if not shared_model:
            task_pl.reset_domain(domains[0])

        print(f"✅ Model initialized: {task_pl.__class__.__name__}")
        print(f"Shared model: {shared_model}")

        # 3. Test logger initialization
        print("\n🔍 Testing logger...")
        logger = TensorBoardLogger(
            save_dir="logs/debug",
            name=domain_str,
            version="debug_run"
        )
        logger.log_hyperparams(config)
        print("✅ Logger initialized")

        # 4. Test trainer initialization
        print("\n🔍 Testing trainer setup...")
        # Create checkpoint callback
        checkpoint_dir = os.path.join(logger.log_dir, "checkpoints")
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor='val_acc',
            mode='max',
            save_top_k=1,
            filename='debug-best-{epoch}-{step}-{val_acc:.2f}',
            save_last=True,
            every_n_train_steps=5  # Save every 5 steps for debugging
        )

        # Calculate appropriate training batches
        num_training_batches = min(5, len(dm.train_dataloader()))  # Use at most 5 batches
        if num_training_batches < 1:
            num_training_batches = 1

        trainer = L.Trainer(
            logger=logger,
            gradient_clip_val=1.0,
            accumulate_grad_batches=config["grad_accum_steps"],
            limit_train_batches=num_training_batches,
            limit_val_batches=1,
            max_steps=10,
            val_check_interval=1,  # Validate every 2 steps
            enable_progress_bar=True,
            log_every_n_steps=1,
            callbacks=[checkpoint_callback],
            fast_dev_run=False,
            detect_anomaly=True,
            overfit_batches=1
        )
        print("✅ Trainer initialized with checkpoint callback")

        print(f" Estimated total steps: {trainer.estimated_stepping_batches}")

        # 6. Test data loader
        print("\n🔍 Testing data loader...")
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        print(f"✅ Got batch: {type(batch)} with keys: {list(batch.keys())}")
        print(f"Batch sizes: input_ids={batch['input_ids'].shape}, labels={batch['labels'].shape}")

        # 7. Test model forward pass
        print("\n🔍 Testing model forward pass...")
        task_pl.eval()
        if shared_model:
            # For shared model, we don't need to reset domain
            outputs = task_pl.training_step(batch, 0)
        else:
            task_pl.reset_domain(domains[0])
            with torch.no_grad():
                outputs = task_pl.training_step(batch, 0)

        print(f"✅ Forward pass successful. Output type: {type(outputs)}")
        if isinstance(outputs, torch.Tensor):
            print(f"Loss value: {outputs.item()}")
        else:
            print(f"Output keys: {list(outputs.keys())}")

        # 8. Test full training run
        print("\n🔍 Starting test training run (10 steps)...")
        trainer.fit(task_pl, train_loader, dm.val_dataloader())
        print("✅ Test training completed successfully!")

        # 9. Enhanced checkpoint verification
        print("\n🔍 Verifying checkpoint saving...")
        print(f"Expected checkpoint directory: {checkpoint_dir}")

        # List all files in the checkpoint directory
        if os.path.exists(checkpoint_dir):
            print("✅ Checkpoint directory exists")
            files = os.listdir(checkpoint_dir)
            print(f"Found {len(files)} files: {files}")

            # Check for checkpoint files
            ckpt_files = [f for f in files if f.endswith('.ckpt')]
            if ckpt_files:
                print(f"✅ Found {len(ckpt_files)} checkpoint files")
                for ckpt in ckpt_files:
                    print(f"  - {ckpt}")

                # Check best model path
                best_path = checkpoint_callback.best_model_path
                if best_path:
                    print(f"Best model path: {best_path}")
                    if os.path.exists(best_path):
                        print("✅ Best model file exists")
                    else:
                        print("❌ Best model file does not exist")
                else:
                    print("❌ No best model path recorded")
            else:
                print("❌ No checkpoint files found")
        else:
            print(f"❌ Checkpoint directory does not exist: {checkpoint_dir}")
            # Debug directory structure
            parent_dir = os.path.dirname(checkpoint_dir)
            print(f"\n📂 Parent directory contents: {os.listdir(parent_dir)}")

            # Check TensorBoard log dir
            tb_log_dir = logger.log_dir
            print(f"\n📊 TensorBoard log dir: {tb_log_dir}")
            print(f"TensorBoard dir exists: {os.path.exists(tb_log_dir)}")
            print(f"TensorBoard dir contents: {os.listdir(tb_log_dir) if os.path.exists(tb_log_dir) else 'N/A'}")

        return True

    except Exception as e:
        print("\n❌ DEBUGGING FAILED!")
        print(f"ERROR TYPE: {type(e).__name__}")
        print(f"ERROR MESSAGE: {str(e)}")
        print("\nFULL TRACEBACK:")
        traceback.print_exc()

        # Additional diagnostics
        print("\n⚙️ SYSTEM INFO:")
        print(f"Python: {sys.version}")
        print(f"PyTorch: {torch.__version__}")
        print(f"Lightning: {L.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA devices: {torch.cuda.device_count()}")

        return False


def debug_hyperparameters(data_dir, domains):
    """Run debugging with a default configuration"""
    # Convert single domain string to list
    if isinstance(domains, str):
        domains = [domains]

    default_config = {
        "grad_accum_steps": 2,
        "lr": 5e-5,
        "batch_size": 16,
        "num_train_steps": 100,
        "num_train_epochs": 2,
        "opt_name": "AdamW"
    }

    domain_str = "-".join(domains)
    print(f"🔧 Starting debugging for domains: {domain_str}")
    success = debug_training(data_dir, domains, default_config)

    if success:
        print("\n🎉 Debugging completed successfully! All components seem functional.")
        print("Possible issues with your tuning script:")
        print("1. Check if your data directory is accessible: ", os.path.abspath(data_dir))
        print("2. Verify domain names match your dataset")
        print("3. Review hyperparameter ranges (especially learning rates)")
    else:
        print("\n🔴 Debugging failed. Please check the error messages above.")
        print("Common solutions:")
        print("1. Verify dataset path and structure")
        print("2. Check model compatibility with your data")
        print("3. Ensure all required dependencies are installed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=False,
                        help="Path to dataset directory",
                        default=r'C:\Users\tangc\PycharmProjects\domain-adaptation\datasets'
                        )
    parser.add_argument("--domains", required=True,
                        type=str,
                        help="Target domain(s) for tuning (hyphen-separated)")
    args = parser.parse_args()

    # Parse domains argument
    domains = args.domains.split("-")

    debug_hyperparameters(args.data_dir, domains)