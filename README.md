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
- `docs/DATA.md`: mô tả cấu trúc data, pipeline xử lý data và sample minh hoạ
- `docs/TRAINING.md`: mô tả pipeline train, sample train, loss và kiến trúc model
- `docs/TRAIN_ARGS.md`: giải thích ý nghĩa các tham số CLI khi train
- `docs/PAPER.md`: ghi chú paper bằng tiếng Việt

## Quick Start

Chạy từ root của repo:

```bash
cd .
```

Kích hoạt môi trường:

```bash
cd .
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hqh
```

Tải và giải nén data:

```bash
cd .
pip install gdown
gdown 1cnMwEkRDP0vmu1L-u9YAZfo7UvpUL7qH -O data_stereo_flow.zip
mkdir -p data
unzip -o data_stereo_flow.zip -d data
```

Sau khi giải nén xong, dữ liệu phải nằm ở:

```text
data/training
data/testing
```

Train preset chất lượng model cuối:

```bash
cd .
python -m dl_stereo_matching_pytorch.train \
  --data-root data \
  --util-root data \
  --model-dir model_win37_kitti2012_quality \
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

Trong lúc train, code sẽ tự ghi log vào:

```text
model_win37_kitti2012_quality/train.log
```

Evaluate checkpoint vừa train:

```bash
cd .
python -m dl_stereo_matching_pytorch.evaluate \
  --data-root data \
  --util-root data \
  --model-dir model_win37_kitti2012_quality \
  --data-version kitti2012 \
  --net-type win37_dep9 \
  --patch-size 37 \
  --disp-range 256 \
  --num-tr-img 160 \
  --num-val-img 34 \
  --num-val-loc 5000 \
  --batch-size 200
```

Trong lúc evaluate, code sẽ tự ghi log vào:

```text
model_win37_kitti2012_quality/eval.log
```

Infer một vài ảnh:

```bash
cd .
python -m dl_stereo_matching_pytorch.infer \
  --data-root data \
  --util-root data \
  --model-dir model_win37_kitti2012_quality \
  --out-dir preds_win37 \
  --data-version kitti2012 \
  --net-type win37_dep9 \
  --disp-range 256 \
  --num-imgs 5
```

Vì sao dùng `train-samples-per-epoch=50000`:

- Bài này train theo `patch pair`, không phải theo số ảnh.
- Mỗi ảnh KITTI sinh ra rất nhiều pixel disparity hợp lệ.
- `50000` là mức đủ đa dạng để model học ổn định hơn, thay vì thấy quá ít patch ở mỗi vòng.


## Ghi chú port

- Kiến trúc dùng `Conv2d + BatchNorm2d + ReLU`, bỏ ReLU ở tầng cuối đúng theo paper.
- Loss là cross-entropy với nhãn mềm `[0.05, 0.2, 0.5, 0.2, 0.05]` quanh disparity đúng.
- Inference full-image tái sử dụng feature map trái/phải rồi dựng cost volume bằng inner product theo từng disparity.
- Với KITTI raw, train/eval dùng trực tiếp `disp_noc` để sinh patch hợp lệ, không cần bước preprocess ngoài repo.
- `train_samples_per_epoch` quyết định số patch ngẫu nhiên mỗi vòng train khi dùng KITTI raw.
- Với `data_stereo_flow.zip` hiện tại, disparity lớn nhất khoảng `228 px`, nên mặc định CLI đã đổi sang `disp_range=256`.
- Để ưu tiên chất lượng model cuối, preset khuyến nghị là `win37_dep9 + Adam + weight_decay=5e-4`.
- Chưa port phần smoothing/CRF hậu xử lý từ paper vì repo TensorFlow hiện tại cũng dừng ở matching network + raw disparity inference.

## Colab

Repo hiện tại chưa chứa file notebook Colab dựng sẵn. Nếu muốn chạy trên Colab, bạn có thể tạo notebook mới rồi lần lượt:

- mount Google Drive nếu cần
- clone repo này
- giải nén `data_stereo_flow.zip` vào `data/`
- chạy lại đúng lệnh train hoặc evaluate ở các khối bash phía trên

## Tài liệu dữ liệu

- Xem chi tiết tại [DATA.md](docs/DATA.md)

## Tài liệu train

- Xem chi tiết tại [TRAINING.md](docs/TRAINING.md)
- Giải thích tham số train tại [TRAIN_ARGS.md](docs/TRAIN_ARGS.md)
