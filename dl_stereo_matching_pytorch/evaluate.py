from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dl_stereo_matching_pytorch.data import DATA_FOLDERS, build_patch_datasets
from dl_stereo_matching_pytorch.models import create_model, get_network_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the PyTorch stereo matching model.")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--data-version", type=str, default="kitti2012", choices=sorted(DATA_FOLDERS))
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--util-root", type=str, required=True)
    parser.add_argument("--net-type", type=str, default="win37_dep9", choices=["win19_dep9", "win37_dep9"])
    parser.add_argument("--num-tr-img", type=int, default=160)
    parser.add_argument("--num-val-img", type=int, default=34)
    parser.add_argument("--patch-size", type=int, default=37)
    parser.add_argument("--num-val-loc", type=int, default=50000)
    parser.add_argument("--disp-range", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    log_path = model_dir / "eval.log"
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("a", encoding="utf-8") as log_file:
        tee = TeeStream(original_stdout, log_file)
        sys.stdout = tee
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            run_evaluation(args, model_dir)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def run_evaluation(args: argparse.Namespace, model_dir: Path) -> None:
    print(f"logging to {model_dir / 'eval.log'}")

    receptive_field = get_network_spec(args.net_type).receptive_field
    if args.patch_size != receptive_field:
        raise ValueError(f"patch_size must be {receptive_field} for {args.net_type}, got {args.patch_size}")

    _, val_dataset = build_patch_datasets(
        data_version=args.data_version,
        data_root=args.data_root,
        util_root=args.util_root,
        num_train_images=args.num_tr_img,
        num_val_images=args.num_val_img,
        num_val_locations=args.num_val_loc,
        patch_size=args.patch_size,
        disp_range=args.disp_range,
        train_samples_per_epoch=None,
    )

    checkpoint = torch.load(model_dir / "checkpoint.pt", map_location="cpu")
    device = torch.device(args.device)
    model = create_model(args.net_type, val_dataset.cache.num_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    correct = 0
    total = 0
    loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for left, right, _targets in loader:
            logits = model(left.to(device), right.to(device))
            predictions = logits.argmax(dim=1).cpu()
            correct += int((predictions - val_dataset.center_label).abs().le(3).sum().item())
            total += int(predictions.numel())

    accuracy = 100.0 * correct / max(total, 1)
    print(f"3-pixel accuracy={accuracy:.3f}% ({correct}/{total})")


if __name__ == "__main__":
    main()
