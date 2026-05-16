from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


DATA_FOLDERS = {
    "kitti2012": ("image_0", "image_1", 1),
    "kitti2015": ("image_2", "image_3", 3),
}


def _validate_data_version(data_version: str) -> Tuple[str, str, int]:
    if data_version not in DATA_FOLDERS:
        raise ValueError("data_version must be 'kitti2012' or 'kitti2015'")
    return DATA_FOLDERS[data_version]


def load_file_ids(util_root: Path) -> np.ndarray:
    file_ids = np.fromfile(util_root / "myPerm.bin", dtype="<f4")
    return np.rint(file_ids).astype(np.int32)


def load_locations(util_root: Path, split: str, num_images: int, patch_size: int, disp_range: int) -> np.ndarray:
    half_patch = patch_size // 2
    half_range = disp_range // 2
    path = util_root / f"{split}_{num_images}_{half_patch}_{half_range}.bin"
    locations = np.fromfile(path, dtype="<f4").reshape(-1, 5).astype(np.int32)
    locations[:, 2:5] -= 1
    return locations


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    std = float(image.std())
    if std < 1e-6:
        std = 1.0
    return (image - float(image.mean())) / std


def read_stereo_pair(data_root: Path, data_version: str, file_id: int) -> Tuple[np.ndarray, np.ndarray]:
    left_folder, right_folder, num_channels = _validate_data_version(data_version)
    left = np.asarray(Image.open(data_root / left_folder / f"{file_id:06d}_10.png"))
    right = np.asarray(Image.open(data_root / right_folder / f"{file_id:06d}_10.png"))

    left = normalize_image(left)
    right = normalize_image(right)

    if num_channels == 1:
        left = left[..., None]
        right = right[..., None]

    return left, right


def read_stereo_pair_by_name(root: Path, left_folder: str, right_folder: str, stem: str, num_channels: int) -> Tuple[np.ndarray, np.ndarray]:
    left = np.asarray(Image.open(root / left_folder / f"{stem}.png"))
    right = np.asarray(Image.open(root / right_folder / f"{stem}.png"))
    left = normalize_image(left)
    right = normalize_image(right)
    if num_channels == 1:
        left = left[..., None]
        right = right[..., None]
    return left, right


def read_disparity_png(path: Path) -> np.ndarray:
    disparity = np.asarray(Image.open(path)).astype(np.float32)
    return disparity / 256.0


def build_target_distribution(disp_range: int, values: Tuple[float, ...] = (0.05, 0.2, 0.5, 0.2, 0.05)) -> torch.Tensor:
    target = torch.zeros(disp_range, dtype=torch.float32)
    half = disp_range // 2
    radius = len(values) // 2
    start = half - radius
    end = half + radius + 1
    if start < 0 or end > disp_range:
        raise ValueError("disp_range is too small for the configured smooth target distribution")
    target[start:end] = torch.tensor(values, dtype=torch.float32)
    return target


@dataclass
class KittiStereoCache:
    data_version: str
    data_root: Path
    util_root: Path
    num_train_images: int
    num_val_images: int

    def __post_init__(self) -> None:
        _, _, self.num_channels = _validate_data_version(self.data_version)
        self.data_root = Path(self.data_root)
        self.util_root = Path(self.util_root)
        self.file_ids = load_file_ids(self.util_root)
        total = self.num_train_images + self.num_val_images
        self.left_images: Dict[int, np.ndarray] = {}
        self.right_images: Dict[int, np.ndarray] = {}
        for file_id in self.file_ids[:total]:
            left, right = read_stereo_pair(self.data_root, self.data_version, int(file_id))
            self.left_images[int(file_id)] = left
            self.right_images[int(file_id)] = right


class StereoPatchDataset(Dataset):
    def __init__(self, cache: KittiStereoCache, locations: np.ndarray, patch_size: int, disp_range: int) -> None:
        self.cache = cache
        self.locations = locations
        self.patch_size = patch_size
        self.disp_range = disp_range
        self.half_patch = patch_size // 2
        self.half_range = disp_range // 2
        self.target = build_target_distribution(disp_range)
        self.center_label = disp_range // 2

    def __len__(self) -> int:
        return int(self.locations.shape[0])

    def _extract(self, image: np.ndarray, center_x: int, center_y: int) -> np.ndarray:
        return image[
            center_y - self.half_patch : center_y + self.half_patch + 1,
            center_x - self.half_patch : center_x + self.half_patch + 1,
            :,
        ]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_id, loc_type, center_x, center_y, right_center_x = self.locations[index]
        right_center_y = center_y

        left_img = self.cache.left_images[int(img_id)]
        right_img = self.cache.right_images[int(img_id)]

        if loc_type == 1:
            left_patch = self._extract(left_img, int(center_x), int(center_y))
            right_patch = right_img[
                right_center_y - self.half_patch : right_center_y + self.half_patch + 1,
                right_center_x - self.half_patch - self.half_range : right_center_x + self.half_patch + self.half_range + 1,
                :,
            ]
        elif loc_type == 2:
            left_patch = np.transpose(self._extract(left_img, int(center_x), int(center_y)), (1, 0, 2))
            right_patch = np.transpose(
                right_img[
                    right_center_y - self.half_patch - self.half_range : right_center_y + self.half_patch + self.half_range + 1,
                    right_center_x - self.half_patch : right_center_x + self.half_patch + 1,
                    :,
                ],
                (1, 0, 2),
            )
        else:
            raise ValueError(f"Unsupported loc_type {loc_type}")

        left_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(left_patch, (2, 0, 1))))
        right_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(right_patch, (2, 0, 1))))
        return left_tensor.float(), right_tensor.float(), self.target.clone()


@dataclass
class RawKittiStereoCache:
    data_version: str
    data_root: Path
    split: str = "training"
    use_disp: str = "disp_noc"
    frame_suffix: str = "_10"

    def __post_init__(self) -> None:
        left_folder, right_folder, self.num_channels = _validate_data_version(self.data_version)
        self.left_folder = left_folder
        self.right_folder = right_folder
        self.data_root = Path(self.data_root)
        self.root = self.data_root / self.split
        if not self.root.exists():
            raise ValueError(
                f"Expected KITTI split directory at {self.root}. "
                "Please set --data-root to the directory that directly contains training/ and testing/."
            )
        self.left_images: Dict[str, np.ndarray] = {}
        self.right_images: Dict[str, np.ndarray] = {}
        self.disparities: Dict[str, np.ndarray] = {}

        if self.split == "training":
            stems = sorted(p.stem for p in (self.root / self.use_disp).glob(f"*{self.frame_suffix}.png"))
        else:
            stems = sorted(p.stem for p in (self.root / self.left_folder).glob(f"*{self.frame_suffix}.png"))
        self.sample_stems = stems

        for stem in stems:
            left, right = read_stereo_pair_by_name(self.root, self.left_folder, self.right_folder, stem, self.num_channels)
            self.left_images[stem] = left
            self.right_images[stem] = right
            if self.split == "training":
                self.disparities[stem] = read_disparity_png(self.root / self.use_disp / f"{stem}.png")


class RawKittiPatchDataset(Dataset):
    def __init__(
        self,
        cache: RawKittiStereoCache,
        patch_size: int,
        disp_range: int,
        sample_stems: List[str],
        samples_per_epoch: int,
        deterministic: bool,
        seed: int = 123,
    ) -> None:
        self.cache = cache
        self.patch_size = patch_size
        self.disp_range = disp_range
        self.half_patch = patch_size // 2
        self.target = build_target_distribution(disp_range)
        self.center_label = disp_range // 2
        self.sample_stems = sample_stems
        self.samples_per_epoch = samples_per_epoch
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)
        self.valid_points: List[Tuple[str, np.ndarray]] = []
        self.flat_points: Optional[List[Tuple[str, int, int, int]]] = [] if deterministic else None
        self._build_index()

    def _build_index(self) -> None:
        for stem in self.sample_stems:
            disparity = self.cache.disparities[stem]
            height, width = disparity.shape
            ys, xs = np.where(disparity > 0)
            valid = []
            for y, x in zip(ys.tolist(), xs.tolist()):
                d = int(round(float(disparity[y, x])))
                right_x = x - d
                if d < 0 or d >= self.disp_range:
                    continue
                if y - self.half_patch < 0 or y + self.half_patch >= height:
                    continue
                if x - self.half_patch < 0 or x + self.half_patch >= width:
                    continue
                if right_x - self.half_patch - self.center_label < 0:
                    continue
                if right_x + self.half_patch + (self.disp_range - self.center_label - 1) >= width:
                    continue
                valid.append((y, x, right_x))
            points = np.asarray(valid, dtype=np.int32)
            self.valid_points.append((stem, points))
            if self.flat_points is not None:
                for y, x, right_x in points.tolist():
                    self.flat_points.append((stem, y, x, right_x))

        if self.deterministic:
            assert self.flat_points is not None
            self.samples_per_epoch = min(self.samples_per_epoch, len(self.flat_points))
        else:
            total_valid = sum(points.shape[0] for _, points in self.valid_points)
            if total_valid == 0:
                raise ValueError("No valid stereo points found in the raw KITTI dataset")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _extract(self, image: np.ndarray, center_x: int, center_y: int) -> np.ndarray:
        return image[
            center_y - self.half_patch : center_y + self.half_patch + 1,
            center_x - self.half_patch : center_x + self.half_patch + 1,
            :,
        ]

    def _sample_point(self, index: int) -> Tuple[str, int, int, int]:
        if self.deterministic:
            assert self.flat_points is not None
            return self.flat_points[index]

        stem_idx = int(self.rng.integers(0, len(self.valid_points)))
        stem, points = self.valid_points[stem_idx]
        point_idx = int(self.rng.integers(0, points.shape[0]))
        y, x, right_x = points[point_idx].tolist()
        return stem, y, x, right_x

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stem, center_y, center_x, right_center_x = self._sample_point(index)
        left_img = self.cache.left_images[stem]
        right_img = self.cache.right_images[stem]

        left_patch = self._extract(left_img, center_x, center_y)
        right_patch = right_img[
            center_y - self.half_patch : center_y + self.half_patch + 1,
            right_center_x - self.half_patch - self.center_label : right_center_x + self.half_patch + (self.disp_range - self.center_label - 1) + 1,
            :,
        ]

        left_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(left_patch, (2, 0, 1))))
        right_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(right_patch, (2, 0, 1))))
        return left_tensor.float(), right_tensor.float(), self.target.clone()


def build_patch_datasets(
    data_version: str,
    data_root: str,
    util_root: str,
    num_train_images: int,
    num_val_images: int,
    num_val_locations: int,
    patch_size: int,
    disp_range: int,
    train_samples_per_epoch: Optional[int] = None,
) -> Tuple[StereoPatchDataset, StereoPatchDataset]:
    util_root_path = Path(util_root) if util_root else None
    if util_root_path and (util_root_path / "myPerm.bin").exists():
        cache = KittiStereoCache(
            data_version=data_version,
            data_root=Path(data_root),
            util_root=util_root_path,
            num_train_images=num_train_images,
            num_val_images=num_val_images,
        )
        train_locations = load_locations(cache.util_root, "tr", num_train_images, patch_size, disp_range)
        if num_val_images == 0:
            val_locations = train_locations[:num_val_locations]
        else:
            val_locations = load_locations(cache.util_root, "val", num_val_images, patch_size, disp_range)[:num_val_locations]
        return (
            StereoPatchDataset(cache, train_locations, patch_size, disp_range),
            StereoPatchDataset(cache, val_locations, patch_size, disp_range),
        )

    raw_cache = RawKittiStereoCache(data_version=data_version, data_root=Path(data_root), split="training")
    all_stems = raw_cache.sample_stems
    if not all_stems:
        raise ValueError("No training disparity files found under training/disp_noc")

    train_count = min(num_train_images, len(all_stems))
    train_stems = all_stems[:train_count]
    remaining = all_stems[train_count:]
    val_count = min(num_val_images, len(remaining))
    val_stems = remaining[:val_count] if val_count > 0 else train_stems[: max(1, min(3, len(train_stems)))]

    if train_samples_per_epoch is None:
        train_samples_per_epoch = 50000
    train_samples_per_epoch = max(1, train_samples_per_epoch)
    val_samples = max(1, num_val_locations)
    return (
        RawKittiPatchDataset(raw_cache, patch_size, disp_range, train_stems, train_samples_per_epoch, deterministic=False),
        RawKittiPatchDataset(raw_cache, patch_size, disp_range, val_stems, val_samples, deterministic=True),
    )
