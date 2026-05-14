from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dl_stereo_matching_pytorch.data import DATA_FOLDERS, build_patch_datasets
from dl_stereo_matching_pytorch.models import create_model, get_network_spec, soft_target_cross_entropy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PyTorch stereo matching model.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-iter", type=int, default=40000)
    parser.add_argument("--model-dir", type=str, default="model")
    parser.add_argument("--data-version", type=str, default="kitti2012", choices=sorted(DATA_FOLDERS))
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--util-root", type=str, required=True)
    parser.add_argument("--net-type", type=str, default="win37_dep9", choices=["win19_dep9", "win37_dep9"])
    parser.add_argument("--num-tr-img", type=int, default=160)
    parser.add_argument("--num-val-img", type=int, default=34)
    parser.add_argument("--patch-size", type=int, default=37)
    parser.add_argument("--num-val-loc", type=int, default=50000)
    parser.add_argument("--train-samples-per-epoch", type=int, default=50000)
    parser.add_argument("--disp-range", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adagrad", "sgd"])
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def validate_patch_size(net_type: str, patch_size: int) -> None:
    receptive_field = get_network_spec(net_type).receptive_field
    if patch_size != receptive_field:
        raise ValueError(f"patch_size must be {receptive_field} for {net_type}, got {patch_size}")


def make_loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle, num_workers=0)


def infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


class TeeStream:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def save_checkpoint(model_dir: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, args: argparse.Namespace) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        model_dir / "checkpoint.pt",
    )
    (model_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    params = model.parameters()
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.optimizer == "adagrad":
        return torch.optim.Adagrad(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    return torch.optim.SGD(params, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)


def update_learning_rate(optimizer: torch.optim.Optimizer, base_lr: float, step: int) -> float:
    lr = base_lr
    completed_steps = step - 1
    if completed_steps >= 24000:
        lr /= 5.0
        lr /= 5.0 ** ((completed_steps - 24000) // 8000)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def main() -> None:
    args = parse_args()
    validate_patch_size(args.net_type, args.patch_size)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    log_path = model_dir / "train.log"
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("a", encoding="utf-8") as log_file:
        tee = TeeStream(original_stdout, log_file)
        sys.stdout = tee
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            run_training(args, model_dir)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def run_training(args: argparse.Namespace, model_dir: Path) -> None:
    print(f"logging to {model_dir / 'train.log'}")

    train_dataset, val_dataset = build_patch_datasets(
        data_version=args.data_version,
        data_root=args.data_root,
        util_root=args.util_root,
        num_train_images=args.num_tr_img,
        num_val_images=args.num_val_img,
        num_val_locations=args.num_val_loc,
        patch_size=args.patch_size,
        disp_range=args.disp_range,
        train_samples_per_epoch=args.train_samples_per_epoch,
    )

    device = torch.device(args.device)
    model = create_model(args.net_type, train_dataset.cache.num_channels).to(device)
    optimizer = build_optimizer(args, model)

    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True)
    train_iter = infinite_loader(train_loader)

    losses = []
    for step in range(1, args.num_iter + 1):
        lr = update_learning_rate(optimizer, args.learning_rate, step)
        left, right, targets = next(train_iter)
        left = left.to(device)
        right = right.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(left, right)
        loss = soft_target_cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        if step % args.eval_every == 0:
            mean_loss = sum(losses) / len(losses)
            print(f"step={step} loss={mean_loss:.6f} lr={lr:.6e}")
            save_checkpoint(model_dir, model, optimizer, step, args)
            losses.clear()

    if losses:
        mean_loss = sum(losses) / len(losses)
        print(f"final_step={args.num_iter} loss={mean_loss:.6f}")
        save_checkpoint(model_dir, model, optimizer, args.num_iter, args)

    print(f"training samples={len(train_dataset)} validation samples={len(val_dataset)}")


if __name__ == "__main__":
    main()
