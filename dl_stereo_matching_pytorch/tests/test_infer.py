from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dl_stereo_matching_pytorch.infer import save_disparity


def test_save_disparity_png(tmp_path: Path) -> None:
    disparity = np.array([[0, 1], [2, 3]], dtype=np.float32)
    out_path = tmp_path / "disp.png"
    save_disparity(out_path, disparity, disp_range=4)

    loaded = np.asarray(Image.open(out_path))
    assert loaded.shape == (2, 2)
    assert loaded.dtype == np.uint8
    assert loaded[0, 0] == 0
    assert loaded[1, 1] == 255
