import os, datetime, json
import argparse
import pandas as pd
import itertools
import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from data_loader import MNLIDataModule
from trainer import seq_pair_classif_training, domain_classif_training, collect_domain_model_paths


BERT_ALL_FT_PATH=r'C:\Users\tangc\PycharmProjects\domain-adaptation\logs\ALL20250703134849\all\wpr0.01lr4e05ebs32epochs3\checkpoints\best\best-0-0.83.ckpt'


def tune_hyperparameters(data_dir, domain, configs):
    results = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/tuning_{domain}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    results_file = os.path.join(log_dir, f"tuning_results_{domain}.csv")
    for i, config in enumerate(configs):
        print(f"\n=== Trial {i + 1}/{len(configs)}: {config} ===")

        try:
            # Create unique logger with descriptive version
            effective_batch_size = 16*config['grad_accum_steps']
            #version = f"warmup_{config['warmup_ratio']}_lr_{config['lr']}_ebs_{effective_batch_size}_freeze_mode_{str(config['freeze_mode'])}_steps_{config['num_train_steps']}_opt_{config['opt_name']}"
            version = list()
            for key, value in config.items():
                version.append(key + "_" + str(value))

            version = "_".join(version)

            logger = TensorBoardLogger(
                save_dir=log_dir,
                name=domain,
                version=version,
                default_hp_metric=False  # We'll add our own
            )

            # Run training
            domains = domain.split("-")
            val_metrics = seq_pair_classif_training(
                data_dir=data_dir,
                domains=domains,
                optimizer_config={
                    "warmup_ratio": config["warmup_ratio"],
                    "lr": config["lr"],
                    "opt_name": config["opt_name"]
                },
                train_params={
                    "batch_size": 16,
                    "num_train_steps": config["num_train_steps"]
                },
                logger=logger,
                train_ratio=config.get("train_ratio", 0.05),
                grad_accum_steps=config['grad_accum_steps'],
                ckpt_path=BERT_ALL_FT_PATH
            )
            # Record results with all metrics
            result = {
                "trial": i + 1,
                #"warmup_ratio": config["warmup_ratio"],
                #"lr": config["lr"],
                "effective_batch_size": effective_batch_size,
                #"num_train_steps": config["num_train_steps"],
                #"opt_name": config["opt_name"],
                "version": version,
                **val_metrics  # Include all validation metrics
            }

            for key, value in config.items():
                result[key] = value
            results.append(result)

            # Save incremental results
            df = pd.DataFrame(results)
            df.to_csv(results_file, index=False)

            print(f"Trial complete | Val Acc: {val_metrics.get('val_acc', 'N/A'):.4f} | Logs: {logger.log_dir}")

        except Exception as e:
            print(f"Trial failed: {str(e)}")
            results.append({
                "trial": i + 1,
                "error": str(e),
                **config
            })

    # Clear GPU memory
    torch.cuda.empty_cache()
    # Find best configuration based on validation accuracy
    successful_runs = [r for r in results if 'val_acc' in r]
    if successful_runs:
        best_run = max(successful_runs, key=lambda x: x['val_acc'])
        print(f"\n Best configuration: acc={best_run['val_acc']:.4f}")
        print(f"Parameters: warmup={best_run["warmup_ratio"]}, lr={best_run['lr']}, ebs={best_run['effective_batch_size']}, "
                  f"steps={best_run['num_train_steps']}, opt={best_run['opt_name']}")
        print(f"Log directory: {os.path.join(log_dir, domain, best_run['version'])}")

        # Save best config separately
        with open(os.path.join(log_dir, "best_config.txt"), "w") as f:
            f.write(str(best_run))

    return log_dir


def start_tensorboard(log_dir, port=6006):
    """Start TensorBoard in the background"""
    import subprocess
    print(f"\n Starting TensorBoard on port {port}...")
    print(f"View at: http://localhost:{port}")
    subprocess.Popen(["tensorboard", "--logdir", log_dir, "--port", str(port)])
    print("Use 'pkill -f tensorboard' to stop TensorBoard when finished")


if __name__ == '__main__':
    # # In your training method
    # logger = TensorBoardLogger("logs", name="domain_adaptation")
    #
    # # Start TensorBoard in a background thread
    # def start_tensorboard(log_dir):
    #     import os
    #     os.system(f"tensorboard --logdir={log_dir} --port=6006")
    #
    # t = threading.Thread(target=start_tensorboard, args=("logs/domain_adaptation",))
    # t.daemon = True
    # t.start()
    #
    # ### 1. Tune sequence classifier on a specific domain
    # args = init_parser()
    #
    # domain = args.domain
    # task = args.task
    #
    # train_params = {
    #     "batch_size": args.batch_size,
    #     "num_train_steps": args.num_train_steps
    # }
    # optimizer_config = {
    #     "lr": args.lr,
    #     "opt_name": args.opt_name
    # }
    # #bl = BaseLearner(r'C:\Users\tangc\PycharmProjects\domain-adaptation\datasets',
    # #                 ['telephone', 'government', 'travel', 'fiction', 'slate'],
    # #                 ['entailment', 'contradiction', 'neutral'])
    # seq_pair_classif_training(r'C:\Users\tangc\PycharmProjects\domain-adaptation\datasets',
    #                           args.domain,
    #                           optimizer_config,
    #                           train_params,
    #                           logger=logger
    #                           )

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=False,
                        help="Path to dataset directory",
                        default=r'C:\Users\tangc\PycharmProjects\domain-adaptation\datasets')
    parser.add_argument("--domain", required=True, help="Target domain for tuning")
    parser.add_argument("--port", type=int, default=6006, help="TensorBoard port")
    parser.add_argument("--task", choices=["seq_pair_classif", "domain_classif"],  # New argument
                        default="seq_pair_classif",
                        help="Task to tune")
    parser.add_argument("--domain_models_dir", type=str,  # New argument
                        help="Directory with domain models (required for domain_classif)")
    parser.add_argument("--single_bert_path", action="store_true", default=False,
                        help="Whether a single pre-trained BERT model to be used as k copies for k domains")
    args = parser.parse_args()

    if args.task == "domain_classif":
        # Define search space for domain classifier hyperparameters
        search_space = {
            "lr": [1e-5],  # Learning rate for domain classifier
            "mu": [0.0],  # Regularization strength
            "grad_accum_steps": [16],
            "batch_size": [2],
            "num_train_steps": [500],
            "train_ratio": [0.05, 0.2, 1.0]
        }

        # Generate all combinations of hyperparameters
        configs = [dict(zip(search_space.keys(), values))
                   for values in itertools.product(*search_space.values())]

        print(f"Starting hyperparameter tuning for domain classifier")
        print(f"Testing {len(configs)} configurations...")

        # Create results directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/domain_classif_tuning_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        results_file = os.path.join(log_dir, f"domain_classif_results.csv")

        results = []
        for i, config in enumerate(configs):
            print(f"\n=== Trial {i + 1}/{len(configs)}: {config} ===")

            try:
                # Create unique logger
                version = "_".join([f"{k}_{v}" for k, v in config.items()])
                logger = TensorBoardLogger(
                    save_dir=log_dir,
                    name="domain_classif",
                    version=version,
                    default_hp_metric=False
                )

                # Collect domain model paths
                domains = args.domain.split("-")
                if args.domain_models_dir is not None:
                    domain_model_paths = collect_domain_model_paths(
                    domains, args.domain_models_dir
                    )
                elif args.single_bert_path:
                    domain_model_paths = [None] * len(domains)
                    print(f"Using single pretrained BERT model for all {len(domains)} domains")

                optimizer_config = {
                    "lr": config["lr"],
                    "mu": config["mu"]
                }

                train_params = {
                    "batch_size": config["batch_size"],
                    "num_train_steps": config["num_train_steps"]
                }

                if "num_train_epochs" in config:
                    train_params['num_train_epochs'] = config["num_train_epochs"]
                    optimizer_config["epochs"] = config["num_train_epochs"]

                # Run domain classifier training
                val_metrics = domain_classif_training(
                    data_dir=args.data_dir,
                    domains=domains,
                    optimizer_config=optimizer_config,
                    train_params=train_params,
                    domain_model_paths=domain_model_paths,
                    logger=logger,
                    train_ratio=config["train_ratio"],
                    grad_accum_steps=config["grad_accum_steps"],
                    freeze_domain_models=False,
                )

                # Record results
                result = {
                    "trial": i + 1,
                    **config,
                    **val_metrics
                }
                results.append(result)

                # Save incremental results
                df = pd.DataFrame(results)
                df.to_csv(results_file, index=False)
                print(f"Trial complete | Val Acc: {val_metrics.get('val_acc', 'N/A'):.4f}")

            except Exception as e:
                print(f"Trial failed: {str(e)}")
                results.append({
                    "trial": i + 1,
                    "error": str(e),
                    **config
                })

        # Find best configuration
        successful_runs = [r for r in results if 'val_acc' in r]
        if successful_runs:
            best_run = max(successful_runs, key=lambda x: x['val_acc'])
            print(f"\nBest configuration: acc={best_run['val_acc']:.4f}")
            print(f"Parameters: lr={best_run['lr']}, mu={best_run['mu']}, "
                  f"batch_size={best_run['batch_size']}, epochs={best_run['num_train_steps']}")

            # Save best config
            with open(os.path.join(log_dir, "best_config.txt"), "w") as f:
                f.write(str(best_run))

    else:  # Original sequence classification tuning
        search_space = {
            "lr": [2e-6, 4e-6, 1e-5, 4e-5],
            "grad_accum_steps": [2],
            "warmup_ratio": [0.01],
            "freeze_mode": [None, "freeze_bert", "freeze_classifier"],
            "num_train_steps": [1500],
            "train_ratio": [0.05, 0.01, 0.2],
            "opt_name": ["AdamW"]
        }

        configs = [dict(zip(search_space.keys(), values))
                   for values in itertools.product(*search_space.values())]

        print(f"Starting hyperparameter tuning for domain: {args.domain}")
        print(f"Testing {len(configs)} configurations...")

        log_dir = tune_hyperparameters(
            data_dir=args.data_dir,
            domain=args.domain,
            configs=configs
        )

