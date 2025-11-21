# train_ror_wrn_svhn.py
"""
RoR-3 on WRN58-4 + Stochastic Depth (training recipe for SVHN)
Based on "Residual Networks of Residual Networks" (RoR) — SVHN config:
- architecture: RoR-3-WRN58-4+SD (paper achieves 1.59% error on SVHN).
- SVHN training: batch_size=128, epochs=50, LR starts 0.1, divide by 10 after epoch 30 and 35.
- weight_decay=1e-4, momentum=0.9, Nesterov.
Reference: residual-of-residual-network.pdf (uploaded). :contentReference[oaicite:1]{index=1}
"""

import math
import argparse
import datetime
from typing import List
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_loader import DigitDataModule
from trainer_digits import DigitCNN, ResNet110, test_loop

import random
import numpy as np
from collections import defaultdict

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you use multi-GPU

    # For full determinism (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 5

set_seed(SEED)



# --------------------------
# Stochastic depth helper
# --------------------------
class StochasticDepth(nn.Module):
    """
    Performs stochastic depth on the *residual branch*.
    drop_prob: probability to DROP the residual branch (0..1).
    forward(x, residual) -> x + residual_or_scaled_masked_residual
    """
    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x, residual):
        # If eval or drop_prob==0, deterministic addition
        if (not self.training) or (self.drop_prob == 0.0):
            return x + residual
        keep_prob = 1.0 - self.drop_prob
        batch_size = residual.shape[0]
        device = residual.device
        # TODO: wrong implementation
        # per-sample binary mask to drop residual path for some samples
        mask = torch.rand(batch_size, 1, 1, 1, device=device) < keep_prob
        mask = mask.float()
        # Scale the kept residuals to preserve expectation
        residual = residual * mask / keep_prob
        return x + residual


# --------------------------
# Basic Wide Residual block (pre-activation layout)
# --------------------------
class BasicBlockWRN(nn.Module):
    """
    Pre-activation WRN basic block: BN -> ReLU -> conv -> BN -> ReLU -> conv
    Applies a 1x1 projection on the identity (x) if in/out channels or stride differ.
    Accepts drop_prob for stochastic depth (applied to residual branch).
    """
    def __init__(self, in_planes, out_planes, stride, drop_prob=0.0):
        super().__init__()
        self.equalInOut = (in_planes == out_planes) and (stride == 1)
        # pre-activation components
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        # projection conv for identity if needed (apply to x)
        if not self.equalInOut:
            self.convShortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        else:
            self.convShortcut = None

        self.sd = StochasticDepth(drop_prob)

    def forward(self, x):
        # Pre-activation path
        #out = self.relu1(self.bn1(x))
        out = self.relu1(x)
        # identity should be projection of x (not of out)
        identity = x if self.equalInOut else self.convShortcut(x)
        out = self.conv1(out)
        #out = self.conv2(self.relu2(self.bn2(out)))
        out = self.conv2(self.relu2(out))
        # stochastic depth expects (identity, residual)
        #return self.sd(identity, out)
        return identity+out


# --------------------------
# RoR-WRN backbone (RoR-3 wrapped around a WRN)
# --------------------------
class RoR_WRN(nn.Module):
    """
    RoR built on a WRN-like backbone (Pre-activation blocks).
    Default: depth=58, widen_factor=4 approximates WRN58-4.
    Implements:
      - final-level residual blocks (BasicBlockWRN)
      - middle-level projections (one per group) that project earlier group inputs
        to the final group's shape, with correct channel & spatial alignment
      - root-level projection from initial conv output to final group channels
    """
    def __init__(self, depth=28, widen_factor=4, num_classes=10, drop_prob_last=0.5):
        super().__init__()
        assert (depth - 4) % 6 == 0, "depth must be 6n + 4 for WRN"
        n = (depth - 4) // 6
        k = widen_factor
        self.n = n

        # channel plan: [16, 16*k, 32*k, 64*k]
        nChannels = [16, 16 * k, 32 * k, 64 * k]

        # initial conv
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)

        # set up blocks with linearly increasing drop probability (0 -> p_l)
        p_l = float(drop_prob_last)
        #total_blocks = n * 3

        # build groups
        self.group1 = self._make_group(nChannels[0], nChannels[1], n, stride=1, start_idx=0, p_l=p_l)
        self.group2 = self._make_group(nChannels[1], nChannels[2], n, stride=2, start_idx=n, p_l=p_l)
        self.group3 = self._make_group(nChannels[2], nChannels[3], n, stride=2, start_idx=2*n, p_l=p_l)

        # Middle-level projections
        self.g1_proj = nn.Conv2d(nChannels[0], nChannels[1], kernel_size=1, stride=1, bias=False)
        self.g2_proj = nn.Conv2d(nChannels[1], nChannels[2], kernel_size=1, stride=2, bias=False)
        self.g3_proj = nn.Conv2d(nChannels[2], nChannels[3], kernel_size=1, stride=2, bias=False)


        # root-level projection: from initial conv output (nChannels[0]) -> final group channels (nChannels[3])
        self.root_proj = nn.Conv2d(nChannels[0], nChannels[3], kernel_size=1, stride=4, bias=False)

        # final BN/ReLU + pooling + fc
        self.bn_last = nn.BatchNorm2d(nChannels[3])
        self.relu_last = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(nChannels[3], num_classes)


    def _make_group(self, in_planes, out_planes, num_blocks, stride, start_idx=0, p_l=0.5):
        layers = []
        total_blocks = self.n * 3
        for i in range(num_blocks):
            layer_idx = start_idx + i
            drop_prob = p_l * (float(layer_idx) / max(1, total_blocks - 1))
            s = stride if i == 0 else 1
            in_ch = in_planes if i == 0 else out_planes
            layers.append(BasicBlockWRN(in_ch, out_planes, stride=s, drop_prob=drop_prob))
        return nn.Sequential(*layers)


    def forward(self, x):
        # initial conv (before groups)
        x0 = self.conv1(x)  # shape: [B, 16, 32, 32] for CIFAR/SVHN sized input

        g1 = self.group1(x0)
        # g1 = g1 + self.g1_proj(x0)
        #
        g2 = self.group2(g1)
        # g2 = g2 + self.g2_proj(g1)
        #
        g3 = self.group3(g2)
        # g3 = g3 + self.g3_proj(g2)
        #
        # y = g3 + self.root_proj(x0)

        y = g3

        #out = self.relu_last(self.bn_last(y))
        out = self.relu_last(y)
        out = self.avgpool(out)  # [B, C, 1, 1]
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out




# --------------------------
# Training + evaluation loops
# --------------------------

def log_gradient_norms_safe(model):
    """Safer gradient monitoring that definitely doesn't modify anything"""
    total_norm = 0.0
    param_count = 0
    # Use no_grad to ensure we don't interfere with computation graph
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Clone and detach to be absolutely safe
                grad_norm = param.grad.detach().clone().norm(2).item()
                total_norm += grad_norm ** 2
                param_count += 1
                #writer.add_scalar(f'Gradients/{name}', grad_norm, global_step)

        if param_count > 0:
            total_norm = total_norm ** 0.5
            #writer.add_scalar('Gradients/Total_Norm', total_norm, global_step)
            #writer.add_scalar('Gradients/Param_Count', param_count, global_step)
        else:
            total_norm = 0.0
            #writer.add_scalar('Gradients/Total_Norm', 0.0, global_step)
            print("⚠️ WARNING: No gradients found!")
    return total_norm


def log_grad_norms_by_layer(model, writer=None, global_step=None):
    layer_norms = defaultdict(float)
    for name, param in model.named_parameters():
        if param.grad is not None:
            layer_name = name.split('.')[0]  # e.g., "conv1.weight" → "conv1"
            layer_norms[layer_name] += param.grad.data.norm(2).item()
    for layer, norm in layer_norms.items():
        if writer:
            writer.add_scalar(f'grad_norm/{layer}', norm, global_step)
        else:
            print(f"{layer}: grad_norm = {norm:.4f}")


activation_stats = {}

def record_activation_stats(name):
    def hook(module, input, output):
        # output is a tensor (or tuple)
        if isinstance(output, torch.Tensor):
            act = output.detach()
            activation_stats[name] = {
                'mean': act.mean().item(),
                'std': act.std().item(),
                'sparsity': (act == 0).float().mean().item()
            }
    return hook

def register_activation_hooks(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            module.register_forward_hook(record_activation_stats(name))


def train_one_epoch(model, device, writer, train_loader, val_loader,
                    optimizer, criterion,
                    grad_accum_steps, epoch,
                    train_log_interval, val_interval, scheduler=None,
                    max_batches=None, best_val_loss=None):
    model.train()
    total_steps = math.ceil(len(train_loader) / grad_accum_steps)
    total_loss = 0.0
    correct = 0
    total = 0
    step_count = 0
    grad_step = 0
    log_flag = True

    register_activation_hooks(model)

    for i, (images, targets) in enumerate(train_loader):
        if max_batches is not None and grad_step == max_batches:
            break

        images = images.to(device)
        targets = targets.to(device).long()  
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        # if (step_count + 1) % grad_accum_steps == 0:
        #     # you can now inspect activation_stats
        #     if grad_step > 0 and grad_step % train_log_interval == 0:
        #         for layer, stats in activation_stats.items():
        #             print(f"{layer}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, sparsity={stats['sparsity']:.4f}")
        #             writer.add_scalar("train/layer_mean", stats['mean'], grad_step + epoch*total_steps)

        loss = criterion(outputs, targets)
        #loss.backward()
        (loss / grad_accum_steps).backward()

        if (step_count + 1) % grad_accum_steps == 0:
            optimizer.step()
            grad_step += 1
            if grad_step % train_log_interval == 0:
                 #total_norm = log_gradient_norms_safe(model)
                 #print(f"Global step {grad_step}, total norm {total_norm}")
                 #writer.add_scalar("train/grad_norm", total_norm,
                 #                  grad_step + epoch*total_steps)

                 log_grad_norms_by_layer(model, writer, grad_step + epoch*total_steps)

            log_flag = True
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()  # TODO: Should I remove this?

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += images.size(0)

        if grad_step > 0 and grad_step % train_log_interval == 0 and log_flag:
            print(f"Global step /Total steps: {grad_step}/{total_steps}")
            print(f"Global step {grad_step} | Average running loss {total_loss / total:.4f}")
            print(f"Global step {grad_step} | Average accuracy {correct / total:.4f}")
            #print(f"Global step {grad_step} | Batch mean {images.mean().item()} | Batch std {images.std().item()}")
            print(f"Global step {grad_step} | Batch loss {loss.item()}")
            if writer is not None:
                writer.add_scalar("train/loss", loss.item(),
                                  grad_step + epoch*total_steps)


        if grad_step % val_interval == 0 and log_flag:
            # Evaluate
            if val_loader is not None:
                model.eval()
                val_loss, correct, total = 0.0, 0, 0
                with torch.no_grad():
                    for imgs, labels in val_loader:
                        imgs, labels = imgs.to(device), labels.to(device)
                        outputs = model(imgs)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        _, preds = outputs.max(1)
                        vloss = criterion(outputs, labels)
                        val_loss += vloss.item() * labels.size(0)
                        correct += preds.eq(labels).sum().item()
                        total += labels.size(0)
                    #total_norm = log_gradient_norms_safe(model)
                    #print(f"Global step {grad_step}, total norm {total_norm}")

                val_acc = correct / total
                avg_val_loss = val_loss / total

                if writer is not None:
                    #writer.add_scalar("val/total_norm", total_norm, grad_step+epoch*total_steps)
                    writer.add_scalar("val/accuracy", val_acc, grad_step+epoch*total_steps)
                    writer.add_scalar("val/loss", avg_val_loss, grad_step+epoch*total_steps)

                if best_val_loss is None or avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    # save best
                    torch.save({'epoch': epoch, 'global_step':grad_step+epoch*total_steps,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict()},
                               config["val_ckpt"])
                    print(f"Saved best val model with acc {best_val_loss:.2f}%")

                model.train()

        log_flag = False
        step_count += 1

    if max_batches is None and step_count % grad_accum_steps != 0:
        optimizer.step()  # lingering parameter update for tail batch
        grad_step += 1
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    print(f"Train Epoch {epoch}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")
    print(f"Train Epoch {epoch}: val Loss={avg_val_loss:.4f}, val Acc={val_acc:.2f}%")
    return best_val_loss


# --------------------------
# Main training entrypoint
# --------------------------
def train_model(datamodule, domain,
                config=None,
                logger=None,
                device="cuda",
                train_log_interval=10, val_interval=50,
                max_batches=None):

    print("Device:", device)
    train_loader = datamodule.get_dataloader(domain, seed=SEED, loader_type='train', is_training=True)
    val_loader = datamodule.get_dataloader(domain, loader_type="val", is_training=False)
    #TODO: now using test data for plotting
    test_loader = datamodule.get_dataloader(domain, loader_type="test", is_training=False)

    if logger is not None:
        writer = logger.experiment
    else:
        writer = None

    #model = RoR_WRN().to(device)
    #model = DigitCNN(num_classes=10).to(device)
    model = ResNet110(n=9, n_classes=10).to(device)

    # Loss, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"],
                          momentum=0.9, weight_decay=1e-4, nesterov=True)

    # LR schedule per paper: start 0.1, divide by 10 after epoch 30 and 35 (for SVHN)
    def adjust_lr(optimizer, epoch):
        if epoch < 30:
            lr = config["lr"]
        elif epoch < 35:
            lr = config["lr"] * 0.1
        else:
            lr = config["lr"] * 0.01
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    best_test_acc, best_test_epoch = 0.0, None
    best_val_loss = None
    for epoch in range(1, config["epochs"] + 1):
        lr = adjust_lr(optimizer, epoch)
        print(f"Epoch {epoch}/{config["epochs"]} — lr={lr:.5f}")

        best_val_loss = train_one_epoch(model, device, writer, train_loader, val_loader,
                    optimizer, criterion,
                    config["grad_accum_steps"], epoch,
                    train_log_interval, val_interval,
                        scheduler=None,
                        max_batches=max_batches,
                        best_val_loss=best_val_loss)

        #TODO: potentially remove this
        print(f"Epoch {epoch}, evaluating on test")
        test_loss, acc = test_loop(model, device, test_loader, criterion)
        writer.add_scalar("test/loss", test_loss, epoch)
        writer.add_scalar("test/accuracy", acc, epoch)
        if acc > best_test_acc:
            best_test_acc = acc
            best_test_epoch = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       config["test_ckpt"])
    print("Training finished. Epoch {} best test acc: {}".format(best_test_epoch, best_test_acc))


if __name__ == "__main__":
    # TODO: add extra data in SVHN for training
    parser = argparse.ArgumentParser(description="RoR-3 WRN58-4 + SD on SVHN")
    parser.add_argument('--domain', type=str)
    args = parser.parse_args()

    domain = "svhn"
    device = "cuda"
    config = {
        "lr": 0.1,
        "epochs": 50,
        "grad_accum_steps": 16,
        "val_ckpt": "best_val_model.ckpt",
        "test_ckpt": "best_test_model.ckpt"
    }

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/tuning_{domain}_{timestamp}"
    version = list()
    for key, value in config.items():
        if key == "optimizer_cfg" or key == "scheduler_cfg":
            for k, v in config[key].items():
                if k == "name":
                    k = ""
                if v is None:
                    continue
                version.append(k + "_" + str(v))
        else:
            version.append(key + "_" + str(value))

    version = "_".join(version)

    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=domain,
        version=version,
        default_hp_metric=False  # We'll add our own
    )

    ddm = DigitDataModule()
    ddm.prepare_data()
    ddm.reset_batch_size(8)
    ddm.setup('fit', task='digit_classif', domain="all", val_ratio=0.1)
    train_model(ddm, domain,
                config=config,
                logger=logger,
                device=device,
                train_log_interval=300, val_interval=400,
                max_batches=None)
