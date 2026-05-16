from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dl_stereo_matching_pytorch.data import DATA_FOLDERS, load_file_ids, read_stereo_pair, read_stereo_pair_by_name
from dl_stereo_matching_pytorch.models import build_cost_volume, create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-image inference with the PyTorch stereo model.")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--data-version", type=str, default="kitti2012", choices=sorted(DATA_FOLDERS))
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--util-root", type=str, required=True)
    parser.add_argument("--net-type", type=str, default="win37_dep9", choices=["win19_dep9", "win37_dep9"])
    parser.add_argument("--disp-range", type=int, default=256)
    parser.add_argument("--num-imgs", type=int, default=5)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def save_disparity(out_path: Path, disparity_map: np.ndarray, disp_range: int) -> None:
    scale = 255.0 / max(disp_range - 1, 1)
    image = np.clip(disparity_map * scale, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(out_path)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    util_root = Path(args.util_root) if args.util_root else None
    if util_root and (util_root / "myPerm.bin").exists():
        file_ids = load_file_ids(util_root)
        first_left, _ = read_stereo_pair(Path(args.data_root), args.data_version, int(file_ids[0]))
        sample_keys = [int(file_id) for file_id in file_ids[args.start_id : args.start_id + args.num_imgs]]
        use_raw_layout = False
    else:
        left_folder, right_folder, raw_channels = DATA_FOLDERS[args.data_version]
        root = Path(args.data_root) / "testing"
        stems = sorted(p.stem for p in (root / left_folder).glob("*_10.png"))
        if not stems:
            raise ValueError(f"No testing samples found under {root / left_folder}")
        sample_keys = stems[args.start_id : args.start_id + args.num_imgs]
        first_left, _ = read_stereo_pair_by_name(root, left_folder, right_folder, sample_keys[0], raw_channels)
        use_raw_layout = True
    in_channels = first_left.shape[2]

    checkpoint = torch.load(Path(args.model_dir) / "checkpoint.pt", map_location="cpu")
    device = torch.device(args.device)
    model = create_model(args.net_type, in_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        for sample_key in sample_keys:
            if use_raw_layout:
                left_folder, right_folder, _ = DATA_FOLDERS[args.data_version]
                testing_root = Path(args.data_root) / "testing"
                left_img, right_img = read_stereo_pair_by_name(testing_root, left_folder, right_folder, sample_key, in_channels)
                output_stem = sample_key
            else:
                file_id = int(sample_key)
                left_img, right_img = read_stereo_pair(Path(args.data_root), args.data_version, file_id)
                output_stem = f"{file_id:06d}_10"
            left_tensor = torch.from_numpy(np.transpose(left_img, (2, 0, 1))).unsqueeze(0).float().to(device)
            right_tensor = torch.from_numpy(np.transpose(right_img, (2, 0, 1))).unsqueeze(0).float().to(device)

            left_features, right_features = model.extract_feature_maps(left_tensor, right_tensor)
            cost_volume = build_cost_volume(left_features, right_features, args.disp_range)
            disparity = cost_volume.argmax(dim=1).squeeze(0).cpu().numpy()

            save_disparity(out_dir / f"disp_map_{output_stem}.png", disparity, args.disp_range)
            print(f"processed sample={output_stem}")


if __name__ == "__main__":
    main()
