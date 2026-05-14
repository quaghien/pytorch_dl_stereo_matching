from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dl_stereo_matching_pytorch.data import (
    KittiStereoCache,
    RawKittiPatchDataset,
    RawKittiStereoCache,
    StereoPatchDataset,
    build_target_distribution,
)


def _write_fake_kitti(tmp_path: Path, data_version: str = "kitti2012") -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    util_root = tmp_path / "util"
    util_root.mkdir(parents=True)

    left_dir = data_root / "image_0"
    right_dir = data_root / "image_1"
    left_dir.mkdir(parents=True)
    right_dir.mkdir(parents=True)

    left = np.arange(30 * 30, dtype=np.uint8).reshape(30, 30)
    right = np.flipud(left)
    Image.fromarray(left).save(left_dir / "000000_10.png")
    Image.fromarray(right).save(right_dir / "000000_10.png")

    np.array([0], dtype="<f4").tofile(util_root / "myPerm.bin")
    return data_root, util_root


def test_build_target_distribution() -> None:
    target = build_target_distribution(9)
    assert torch.isclose(target.sum(), torch.tensor(1.0))
    assert torch.argmax(target).item() == 4


def test_patch_extraction_horizontal_and_vertical(tmp_path: Path) -> None:
    data_root, util_root = _write_fake_kitti(tmp_path)
    cache = KittiStereoCache(
        data_version="kitti2012",
        data_root=data_root,
        util_root=util_root,
        num_train_images=1,
        num_val_images=0,
    )

    locations = np.array(
        [
            [0, 1, 11, 11, 13],
            [0, 2, 11, 11, 13],
        ],
        dtype=np.int32,
    )
    dataset = StereoPatchDataset(cache, locations=locations, patch_size=5, disp_range=5)

    left_h, right_h, target_h = dataset[0]
    left_v, right_v, target_v = dataset[1]

    assert left_h.shape == (1, 5, 5)
    assert right_h.shape == (1, 5, 9)
    assert left_v.shape == (1, 5, 5)
    assert right_v.shape == (1, 5, 9)
    assert torch.equal(target_h, target_v)

    horizontal_np = left_h.squeeze(0).numpy()
    vertical_np = left_v.squeeze(0).numpy()
    assert np.allclose(vertical_np, horizontal_np.T)


def test_raw_kitti_patch_dataset(tmp_path: Path) -> None:
    data_root = tmp_path / "raw"
    for sub in ["training/image_0", "training/image_1", "training/disp_noc"]:
        (data_root / sub).mkdir(parents=True)

    left = np.tile(np.arange(80, dtype=np.uint8), (50, 1))
    right = np.roll(left, shift=-2, axis=1)
    disp = np.zeros((50, 80), dtype=np.uint16)
    disp[:, 20:70] = 2 * 256

    Image.fromarray(left).save(data_root / "training/image_0/000000_10.png")
    Image.fromarray(right).save(data_root / "training/image_1/000000_10.png")
    Image.fromarray(disp).save(data_root / "training/disp_noc/000000_10.png")

    cache = RawKittiStereoCache(data_version="kitti2012", data_root=data_root)
    dataset = RawKittiPatchDataset(
        cache=cache,
        patch_size=19,
        disp_range=5,
        sample_stems=["000000_10"],
        samples_per_epoch=8,
        deterministic=True,
    )
    left_patch, right_patch, target = dataset[0]
    assert left_patch.shape == (1, 19, 19)
    assert right_patch.shape == (1, 19, 23)
    assert target.shape == (5,)
