# `dl_stereo_matching_pytorch`

Bản port PyTorch độc lập hoàn toàn của repo TensorFlow `dl_stereo_matching`, dựa trên paper:

- Wenjie Luo, Alexander G. Schwing, Raquel Urtasun, *Efficient Deep Learning for Stereo Matching*, CVPR 2016

Toàn bộ mã chạy trong thư mục này, không import gì từ thư mục `dl_stereo_matching`.

## Cấu trúc

- `data.py`: đọc KITTI + các file `.bin` preprocess giống repo gốc
- `models.py`: kiến trúc siamese CNN `win19_dep9` và `win37_dep9`
- `train.py`: train patch-level với smooth target distribution
- `evaluate.py`: đánh giá 3-pixel accuracy trên validation patches
- `infer.py`: sinh disparity map cho ảnh đầy đủ
- `tests/`: unit tests tự chứa bằng dữ liệu giả lập
- `PAPER.md`: ghi chú paper bằng tiếng Việt

## Quick Start

Chạy từ root của repo:

```bash
cd pytorch_dl_stereo_matching
```

Kích hoạt môi trường:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hqh
```

Tải và giải nén data:

```bash
pip install gdown
gdown 1cnMwEkRDP0vmu1L-u9YAZfo7UvpUL7qH -O dl_stereo_matching_pytorch/data_stereo_flow.zip
mkdir -p dl_stereo_matching_pytorch/data
unzip -o dl_stereo_matching_pytorch/data_stereo_flow.zip -d dl_stereo_matching_pytorch/data
```

Sau khi giải nén xong, dữ liệu phải nằm ở:

```text
dl_stereo_matching_pytorch/data/training
dl_stereo_matching_pytorch/data/testing
```

Train preset chất lượng model cuối:

```bash
python -m dl_stereo_matching_pytorch.train \
  --data-root dl_stereo_matching_pytorch/data \
  --util-root dl_stereo_matching_pytorch/data \
  --model-dir dl_stereo_matching_pytorch/model_win37_kitti2012_quality \
  --data-version kitti2012 \
  --net-type win37_dep9 \
  --patch-size 37 \
  --disp-range 256 \
  --optimizer adam \
  --learning-rate 1e-3 \
  --weight-decay 5e-4 \
  --num-tr-img 160 \
  --num-val-img 34 \
  --num-val-loc 5000 \
  --train-samples-per-epoch 50000 \
  --batch-size 128 \
  --num-iter 40000 \
  --eval-every 100
```

`--num-iter 40000` nghĩa là train đủ `40000` step thì dừng.

Vì sao dùng `train-samples-per-epoch=50000`:

- Bài này train theo `patch pair`, không phải theo số ảnh.
- Mỗi ảnh KITTI sinh ra rất nhiều pixel disparity hợp lệ.
- `50000` là mức đủ đa dạng để model học ổn định hơn, thay vì thấy quá ít patch ở mỗi vòng.

## Colab

Notebook dành cho Google Colab nằm tại:

- [colab_train_kitti2012_quality.ipynb](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/colab_train_kitti2012_quality.ipynb)

Notebook này được thiết kế để:

- mount Google Drive nếu cần
- clone repo nếu cần hoặc dùng repo đã có sẵn
- giải nén `data_stereo_flow.zip`
- chạy preset train chất lượng cao cho `kitti2012`
- chạy evaluate sau khi train xong

Trên Colab, chỉ cần:

1. Upload notebook này lên Colab hoặc mở nó từ repo `pytorch_dl_stereo_matching`.
2. Sửa cell cấu hình đầu tiên:
   - `REPO_DIR` hoặc `REPO_URL`
   - `DATA_ZIP`
   - `OUTPUT_ROOT`
3. Run all.

Nếu dùng GPU nhiều VRAM hơn, có thể tăng `BATCH_SIZE` trong notebook lên `256`, `384` hoặc `512` rồi theo dõi lại throughput và VRAM.

Evaluate:

```bash
python -m dl_stereo_matching_pytorch.evaluate \
  --data-root PATH_DATABASE \
  --util-root PATH_BINARY_OR_EMPTY \
  --model-dir MODEL_DIR \
  --net-type win37_dep9 \
  --patch-size 37 \
  --disp-range 256
```

Infer full image:

```bash
python -m dl_stereo_matching_pytorch.infer \
  --data-root PATH_DATABASE \
  --util-root PATH_BINARY_OR_EMPTY \
  --model-dir MODEL_DIR \
  --out-dir OUT_DIR \
  --net-type win37_dep9 \
  --disp-range 256 \
  --num-imgs 5
```

## Ghi chú port

- Kiến trúc dùng `Conv2d + BatchNorm2d + ReLU`, bỏ ReLU ở tầng cuối đúng theo paper.
- Loss là cross-entropy với nhãn mềm `[0.05, 0.2, 0.5, 0.2, 0.05]` quanh disparity đúng.
- Inference full-image tái sử dụng feature map trái/phải rồi dựng cost volume bằng inner product theo từng disparity.
- Với KITTI raw, train/eval dùng trực tiếp `disp_noc` để sinh patch hợp lệ, không cần bước preprocess ngoài repo.
- `train_samples_per_epoch` quyết định số patch ngẫu nhiên mỗi vòng train khi dùng KITTI raw.
- Với `data_stereo_flow.zip` hiện tại, disparity lớn nhất khoảng `228 px`, nên mặc định CLI đã đổi sang `disp_range=256`.
- Để ưu tiên chất lượng model cuối, preset khuyến nghị là `win37_dep9 + Adam + weight_decay=5e-4`.
- Chưa port phần smoothing/CRF hậu xử lý từ paper vì repo TensorFlow hiện tại cũng dừng ở matching network + raw disparity inference.
