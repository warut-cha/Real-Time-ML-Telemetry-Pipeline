"""
  - Every step:   loss, accuracy, grad_norm  (TrainingEvent)
  - Every N steps: weight + gradient tensors (TensorSnapshot, type=WEIGHT/GRADIENT)
  - Every forward: activation maps           (TensorSnapshot, type=ACTIVATION)
  - Activation snapshots carry the correct step number via hook.set_step()

Architecture:
  features.0  Conv2d(1,32,3×3)   weight shape [32,1,3,3]
  features.1  ReLU
  features.2  MaxPool2d(2)
  features.3  Conv2d(32,64,3×3)  weight shape [64,32,3,3]
  features.4  ReLU
  features.5  MaxPool2d(2)
  classifier.0 Flatten
  classifier.1 Linear(3136,128)  weight shape [128,3136]
  classifier.2 ReLU
  classifier.3 Dropout(0.5)
  classifier.4 Linear(128,10)    weight shape [10,128]
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from hook import OmnistreamHook


class MnistCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # features.0
            nn.ReLU(inplace=True),                         # features.1
            nn.MaxPool2d(2),                               # features.2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # features.3
            nn.ReLU(inplace=True),                         # features.4
            nn.MaxPool2d(2),                               # features.5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                  # classifier.0
            nn.Linear(64 * 7 * 7, 128),                  # classifier.1
            nn.ReLU(inplace=True),                         # classifier.2
            nn.Dropout(0.5),                               # classifier.3
            nn.Linear(128, 10),                           # classifier.4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def train_one_epoch(
    model:          nn.Module,
    loader:         DataLoader,
    criterion:      nn.Module,
    optimizer:      optim.Optimizer,
    hook:           OmnistreamHook | None,
    epoch:          int,
    device:         torch.device,
    global_step:    int,
    snapshot_every: int,
) -> tuple[float, float, int]:
    model.train()
    total_loss = total_correct = total_samples = 0

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        if hook:
            hook.set_step(global_step)

        optimizer.zero_grad()
        logits = model(x)           # activation hooks fire here, step is correct
        loss   = criterion(logits, y)
        loss.backward()

        grad_norm = OmnistreamHook.compute_grad_norm(model) if hook else 0.0
        optimizer.step()

        preds    = logits.argmax(dim=1)
        correct  = (preds == y).sum().item()
        accuracy = correct / len(y)

        total_loss    += loss.item()
        total_correct += correct
        total_samples += len(y)

        if hook:
            hook.on_step(
                global_step, loss.item(), accuracy, grad_norm,
                epoch=epoch,
                learning_rate=optimizer.param_groups[0]['lr'],
            )
            # Weight + gradient snapshots every N steps
            if global_step % snapshot_every == 0:
                hook.capture_snapshot(model, global_step)

        if batch_idx % 100 == 0:
            print(f"  epoch {epoch:2d} | batch {batch_idx:4d}/{len(loader)} | "
                  f"loss {loss.item():.4f} | acc {accuracy:.3f} | "
                  f"grad_norm {grad_norm:.4f}")

        global_step += 1

    return total_loss / len(loader), total_correct / total_samples, global_step


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = total_samples = 0
    with torch.no_grad():
        for x, y in loader:
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            total_loss    += criterion(logits, y).item()
            total_correct += (logits.argmax(1) == y).sum().item()
            total_samples += len(y)
    return total_loss / len(loader), total_correct / total_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",          type=int,   default=1)
    parser.add_argument("--batch-size",      type=int,   default=64)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--endpoint",        type=str,   default="tcp://127.0.0.1:5555")
    parser.add_argument("--no-stream",       action="store_true")
    parser.add_argument("--data-dir",        type=str,   default="./data")
    parser.add_argument("--snapshot-every",  type=int,   default=25)
    parser.add_argument("--snapshot-sample", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device} epochs={args.epochs} batch={args.batch_size}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds     = datasets.MNIST(args.data_dir, train=True,  download=True, transform=transform)
    test_ds      = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=256,             shuffle=False, num_workers=2)

    model     = MnistCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(f"[train] parameters={sum(p.numel() for p in model.parameters()):,}")

    hook: OmnistreamHook | None = None
    activation_handles = []

    if not args.no_stream:
        hook = OmnistreamHook(
            zmq_endpoint    = args.endpoint,
            snapshot_layers = ['features.0', 'features.3', 'classifier.1', 'classifier.4'],
            snapshot_every  = args.snapshot_every,
            snapshot_sample = args.snapshot_sample,
        )
        hook.start()
        # Register forward hooks — they read hook._current_step when they fire
        activation_handles = hook.register_activation_hooks(model)
        print(f"[train] streaming to {args.endpoint}")
        print(f"[train] snapshots every {args.snapshot_every} steps "
              f"({args.snapshot_sample*100:.0f}% sample)")

    global_step = 0
    t_start     = time.perf_counter()

    for epoch in range(args.epochs):
        t_epoch = time.perf_counter()
        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer,
            hook, epoch, device, global_step, args.snapshot_every,
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        if hook:
            hook.on_epoch_end(epoch)

        print(f"[epoch {epoch:2d}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"{time.perf_counter()-t_epoch:.1f}s")

    print(f"\n[train] finished {args.epochs} epochs in {time.perf_counter()-t_start:.1f}s")

    if hook:
        for h in activation_handles:
            h.remove()
        hook.stop()


if __name__ == "__main__":
    main()
