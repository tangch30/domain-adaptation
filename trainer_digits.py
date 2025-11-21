## code by ChatGPT

import argparse
import itertools
import datetime
import os
import copy
import math
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.loggers import TensorBoardLogger
from data_loader import DigitDataModule
#from trainer_digits_ror_wrn import RoR_WRN


# -----------------------------
# Model definition
# -----------------------------
class DigitCNN(nn.Module):
    def __init__(self, n1=64, n2=128, n3=256, n4=128, kernel_size=5, num_classes=10):
        super().__init__()
        assert kernel_size in [3, 5], "Please provide a legal kernel size"
        if kernel_size == 5:
            padding = 2
        else:
            padding = 1
        self.features = nn.Sequential(
            nn.Conv2d(3, n1, kernel_size=kernel_size, stride=1, padding=padding),  # out: 32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(n1, n2, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(n2, n3, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 4x4

            nn.Conv2d(n3, n4, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),  # 1x1
        )
        self.classifier = nn.Linear(n4, num_classes)

    def forward(self, x):
        feats = self.features(x)  # [B, 128, 1, 1]
        feats = feats.view(x.size(0), -1)
        logits = self.classifier(feats)
        return logits, feats  # return both for later domain transfer


class ResidualBlock(nn.Module):
    def __init__(self, n=18, out_channels=16, sd=False, is_training=False):
        super().__init__()
        assert out_channels in [16, 32, 64]
        if out_channels == 16:
            in_channels = 16
            downsample_stride = 1
        elif out_channels == 32:
            in_channels = 16
            downsample_stride = 2
        else:
            in_channels = 32
            downsample_stride = 2

        #self.layer_dict = dict()
        self.n = n
        self.drop_probs = dict()

        self.sd = sd
        self.is_training = is_training

        if sd:
            delta = 0.5 / self.n
            for layer in range(self.n):
                self.drop_probs[layer] = 1 - (layer+1)*delta

        self.layers = nn.ModuleList()

        for layer in range(n):
            if layer == 0:
                conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                  stride=downsample_stride, padding=1)
            else:
                conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

            bn1 = nn.BatchNorm2d(out_channels)
            relu1 = nn.ReLU(inplace=False)
            conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            bn2 = nn.BatchNorm2d(out_channels)
            relu2 = nn.ReLU(inplace=False)

            if layer == 0:
                convShortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                         stride=downsample_stride, bias=False)
            else:
                convShortcut = None

            self.layers.append(
                nn.ModuleDict({
                    "main": nn.Sequential(conv1, bn1, relu1, conv2, bn2),
                    "shortcut": convShortcut,
                    "relu_out": relu2
                })
            )

    def forward(self, x, is_training=False):
        out = x
        for layer in range(self.n):
            block = self.layers[layer]
            conv_out = block["main"](out)

            if self.sd and is_training:
                conv_out = torch.bernoulli(self.drop_probs[layer]) * conv_out

            shortcut = block["shortcut"]
            if shortcut is None:
                out = block["relu_out"](conv_out + out)
            else:
                out = block["relu_out"](conv_out + shortcut(out))
        return out


class ResNet110(nn.Module):
    def __init__(self, n=18, n_classes=10, sd=False):
        super().__init__()
        self.layer1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.block1 = ResidualBlock(n=n, out_channels=16, sd=sd)
        self.block2 = ResidualBlock(n=n, out_channels=32, sd=sd)
        self.block3 = ResidualBlock(n=n, out_channels=64, sd=sd)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x, is_training=False):
        out = self.layer1(x)
        out = self.block1.forward(out, is_training=is_training)
        out = self.block2.forward(out, is_training=is_training)
        out = self.block3.forward(out, is_training=is_training)
        out = self.pool(out) # NxCx1x1
        out = torch.flatten(out, 1)
        return self.classifier(out)



# -----------------------------
# Training loop
# -----------------------------
def test_loop(model, device, test_loader, criterion):
    model.eval()
    total_loss=0.0
    correct=0
    total=0
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device).long()
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += images.size(0)
    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    print(f"Test: Loss={avg_loss:.4f}, Acc={acc:.2f}%")
    return avg_loss, acc


def train_model(datamodule, domain,
                config=None,
                use_val=False,
                logger=None,
                device="cuda",
                train_log_interval=10, val_interval=50,
                train_ratio=None,
                max_batches=None):

    if logger is not None:
        writer = logger.experiment
    else:
        writer = None

    # Model, optimizer, loss
    model = DigitCNN(num_classes=10).to(device)
    #model = RoR_WRN().to(device)
    criterion = nn.CrossEntropyLoss()

    grad_accum_steps = config["grad_accum_steps"]
    epochs = config["epochs"]

    optimizer_cfg = copy.deepcopy(config["optimizer_cfg"])
    opt_name = optimizer_cfg.pop("name")
    optimizer_cls = getattr(optim, opt_name)
    optimizer = optimizer_cls(model.parameters(), **optimizer_cfg)

    train_loader = datamodule.get_dataloader(domain, loader_type='train', is_training=True)

    scheduler_cfg = config.get("scheduler_cfg", None)
    scheduler = None
    if scheduler_cfg is not None and scheduler_cfg["name"] is not None:
        scheduler_cfg = copy.deepcopy(scheduler_cfg)
        scheduler_cls = getattr(optim.lr_scheduler, scheduler_cfg.pop("name"))
        scheduler_cfg["max_lr"] = optimizer_cfg["lr"]
        scheduler_cfg["total_steps"] = math.ceil(len(train_loader) / grad_accum_steps) * epochs
        scheduler = scheduler_cls(optimizer, **scheduler_cfg)

    val_loader = None
    if use_val:
        val_loader = datamodule.get_dataloader(domain, loader_type="val", is_training=False)

    test_loader = datamodule.get_dataloader(domain, loader_type="test", is_training=False)

    print(f"Training model with data loader batch size {datamodule.batch_size} and effective batch size {datamodule.batch_size*grad_accum_steps}")

    # Training
    global_step = 0
    log_flag = True
    per_epoch_steps = math.ceil(len(train_loader) / grad_accum_steps)
    per_epoch_true_steps = len(train_loader)
    total_steps = per_epoch_steps*epochs
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        step_count = 0
        optimizer.zero_grad()
        for imgs, labels in train_loader:
            if max_batches is not None and step_count == max_batches:
                break

            if train_ratio is not None and step_count == math.ceil(per_epoch_true_steps * train_ratio):
                break

            imgs, labels = imgs.to(device), labels.to(device)

            #outputs, _ = model(imgs)
            outputs = model(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            (loss / grad_accum_steps).backward()

            if (step_count + 1) % grad_accum_steps == 0:
                optimizer.step()
                global_step += 1
                log_flag = True
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad() #TODO: Should I remove this?

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            if global_step % train_log_interval == 0 and log_flag:
                print(f"Global step /Total steps: {global_step}/{total_steps}")
                print(f"Global step {global_step} | Average running loss {running_loss / total:.4f}")
                print(f"Global step {global_step} | Average accuracy {correct / total:.4f}")
                print(f"Global step {global_step} | Batch loss {loss.item()}")
                if writer is not None:
                    writer.add_scalar("train/loss", loss.item(), global_step)

            if global_step % val_interval == 0 and log_flag:
                # Evaluate
                val_acc = None
                if val_loader is not None:
                    model.eval()
                    val_loss, correct, total = 0.0, 0, 0
                    with torch.no_grad():
                        for imgs, labels in val_loader:
                            imgs, labels = imgs.to(device), labels.to(device)
                            #outputs, _ = model(imgs)
                            outputs = model(imgs)
                            if isinstance(outputs, tuple):
                                outputs = outputs[0]
                            _, preds = outputs.max(1)
                            vloss = criterion(outputs, labels)
                            val_loss += vloss.item() * labels.size(0)
                            correct += preds.eq(labels).sum().item()
                            total += labels.size(0)

                    val_acc = correct / total
                    avg_val_loss = val_loss / total

                    if writer is not None:
                        writer.add_scalar("val/accuracy", val_acc, global_step)
                        writer.add_scalar("val/loss", avg_val_loss, global_step)

            log_flag = False
            step_count += 1

        if step_count % grad_accum_steps != 0:
            optimizer.step() # lingering parameter update for tail batch
            global_step += 1
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        #TODO
        if val_acc is not None:
            print(f"Epoch {epoch} ends | Latest validation accuracy {val_acc:.4f}")
        if avg_val_loss is not None:
            print(f"Epoch {epoch} ends | Latest validation loss {avg_val_loss:.4f}")

    test_loss, test_acc = test_loop(model, device, test_loader, criterion)


    return model, val_acc, test_loss, test_acc


# IDE is running low on memory (???)
def tune_hyperparameters(domain, configs, train_ratio=None, max_batches=None):
    results = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/tuning_{domain}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    results_file = os.path.join(log_dir, f"tuning_results_{domain}.csv")
    best_acc = 0.0
    for i, config in enumerate(configs):
        print(f"\n=== Trial {i + 1}/{len(configs)}: {config} ===")

        try:
            # Create unique logger with descriptive version
            #version = f"warmup_{config['warmup_ratio']}_lr_{config['lr']}_ebs_{effective_batch_size}_freeze_mode_{str(config['freeze_mode'])}_steps_{config['num_train_steps']}_opt_{config['opt_name']}"
            version = list()
            for key, value in config.items():
                if key == "optimizer_cfg" or key == "scheduler_cfg":
                    for k, v in config[key].items():
                        if k == "name":
                            k = ""
                        if v is None:
                            continue
                        version.append(k+"_"+str(v))
                else:
                    version.append(key + "_" + str(value))

            if max_batches is not None:
                version.append("max_batches" + "_" + str(max_batches))
                version.append("max_batches" + "_" + str(max_batches))

            if train_ratio is not None:
                version.append("train_ratio" + "_" + str(train_ratio))
                version.append("train_ratio" + "_" + str(train_ratio))

            version = "_".join(version)

            logger = TensorBoardLogger(
                save_dir=log_dir,
                name=domain,
                version=version,
                default_hp_metric=False  # We'll add our own
            )

            grad_accum_steps = config.get("grad_accum_steps", 4)
            epochs = config.get("epochs", 1)
            config["grad_accum_steps"] = grad_accum_steps
            config["epochs"] = epochs
            optimizer_cfg = config.get("optimizer_cfg", None)
            # Flexible optimizer config
            if optimizer_cfg is None or "lr" not in optimizer_cfg or "name" not in optimizer_cfg:
                config["optimizer_cfg"] = {"name": "Adam", "lr": 0.001}


            # Set up data module
            ddm = DigitDataModule()
            ddm.prepare_data()
            ddm.reset_batch_size(8)
            ddm.setup('fit', task='digit_classif', domain="all", val_ratio=0.2)

            # Run training
            model, val_acc, test_loss, test_acc = train_model(ddm, domain,
                        config=config,
                        use_val=True,
                        logger=logger,
                        device="cuda",
                        train_log_interval=50, val_interval=300,
                        train_ratio=train_ratio,
                        max_batches=max_batches)
            results.append({"val_acc": val_acc, "test_loss": test_loss, "test_acc":test_acc,
                            "config": config, "version":version})

            # if val_acc > best_acc:
            #     best_acc = val_acc
            #     torch.save({'state_dict': model.state_dict()}, "best_model.ckpt")
            #     print(f"Saved best model with acc {best_acc:.2f}%")

        except Exception as e:
            print(f"Trial failed: {str(e)}")
            traceback.print_exc()
            results.append({
                "trial": i + 1,
                "error": str(e),
                **config
            })

    # Clear GPU memory
    torch.cuda.empty_cache()

    # Find best configuration based on validation accuracy
    successful_runs = [r for r in results if 'test_acc' in r]
    if successful_runs:
        best_run = max(successful_runs, key=lambda x: x['test_acc'])
        print(f"\n Best configuration: acc={best_run['test_acc']:.4f}")
        print(f"Parameters: {best_run["config"]}")
        final_log_dir = os.path.join(log_dir, domain, best_run['version'])
        print(f"Log directory: {final_log_dir}")

        # Save best config separately
        with open(os.path.join(final_log_dir, "best_config.txt"), "w") as f:
            f.write(str(best_run))
    return log_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="svhn",
                        help="Training domain (e.g., svhn, mnist, usps, all)")
    parser.add_argument("--max_batches", type=int, default=None)

    args = parser.parse_args()

    # search_space = {
    #     # learning rate, learning rate schedule, batch size, optimizer
    #     "grad_accum_steps": [16, 32],
    #     "optimizer_cfg": [{"name": "Adam", "lr": 1e-3},
    #                       {"name": "Adam", "lr": 5e-4},
    #                       {"name": "AdamW", "lr": 1e-3},
    #                       {"name": "AdamW", "lr": 5e-4},
    #                       ],
    #     "scheduler_cfg": [{"name": None},
    #                       {"name": "OneCycleLR", "pct_start": 0.1},
    #                       {"name": "OneCycleLR", "pct_start": 0.2},
    #                       {"name": "OneCycleLR", "pct_start": 0.3},
    #                       ],
    # }

    search_space = {
        # learning rate, learning rate schedule, batch size, optimizer
        "grad_accum_steps": [16, 32],
        "epochs": [50],
        "optimizer_cfg": [
                          #{"name": "AdamW", "lr": 3e-4},
                          {"name": "AdamW", "lr": 5e-4},
                          ],
        "scheduler_cfg": [
                          {"name": "OneCycleLR", "pct_start": 0.1},
                          ],
    }

    configs = [dict(zip(search_space.keys(), values))
               for values in itertools.product(*search_space.values())]

    print(f"Starting hyperparameter tuning for domain: {args.domain}")
    print(f"Testing {len(configs)} configurations...")

    log_dir = tune_hyperparameters(
        domain=args.domain,
        configs=configs,
        train_ratio=0.35,
        max_batches=None,
    )

    # Model saving (better practice: save full checkpoint with optimizer state)
    #torch.save(trained_model.state_dict(), f"digit_cnn_{args.domain}.pth")
