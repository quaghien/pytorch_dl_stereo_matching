# Code Flow: Train, Evaluate, Infer

Tài liệu này mô tả luồng chạy chính của folder `dl_stereo_matching_pytorch` theo đúng code hiện tại, gồm:

- train
- evaluate
- infer

Ảnh workflow:

- Train: [train_workflow.svg](../figures/train_workflow.svg)
- Infer: [infer_workflow.svg](../figures/infer_workflow.svg)

Mục tiêu là để đọc từ trên xuống và lần được:

- lệnh CLI đi vào file nào
- hàm nào gọi hàm nào
- model, data loader, checkpoint và output chạy qua đâu

## 1. Tổng quan file chính

- [train.py](../train.py): entrypoint train
- [evaluate.py](../evaluate.py): entrypoint evaluate
- [infer.py](../infer.py): entrypoint infer full image
- [data.py](../data.py): đọc KITTI, build patch dataset
- [models.py](../models.py): định nghĩa kiến trúc `win19_dep9`, `win37_dep9`, loss và cost volume

## 2. Flow train

Lệnh:

```bash
cd .
python -m dl_stereo_matching_pytorch.train ...
```

### 2.1 Entry

- Python gọi `main()` ở [train.py](../train.py)
- `main()` đọc CLI qua `parse_args()` ở [train.py](../train.py)
- `main()` kiểm tra `patch_size` có khớp với model qua `validate_patch_size()` ở [train.py](../train.py)

### 2.2 Logging và thư mục output

- `main()` tạo `model_dir` ở [train.py](../train.py)
- `main()` mở file log `train.log` ở [train.py](../train.py)
- `TeeStream` ở [train.py](../train.py) dùng để ghi đồng thời ra terminal và file log
- sau đó `main()` gọi `run_training()` ở [train.py](../train.py)

### 2.3 Build dataset

- `run_training()` gọi `build_patch_datasets()` ở [data.py](../data.py)
- `build_patch_datasets()` có 2 nhánh:
  - nếu có `myPerm.bin` thì dùng dữ liệu preprocess kiểu cũ:
    - `KittiStereoCache` ở [data.py](../data.py)
    - `load_locations()` ở [data.py](../data.py)
    - `StereoPatchDataset` ở [data.py](../data.py)
  - nếu không có `myPerm.bin` thì dùng KITTI raw:
    - `RawKittiStereoCache` ở [data.py](../data.py)
    - `RawKittiPatchDataset` ở [data.py](../data.py)

### 2.4 Cách dữ liệu KITTI raw được chuẩn bị

Trong nhánh raw:

- `RawKittiStereoCache.__post_init__()` đọc toàn bộ `training/image_0`, `training/image_1`, `training/disp_noc` ở [data.py](../data.py)
- ảnh được đọc bởi `read_stereo_pair_by_name()` ở [data.py](../data.py)
- disparity ground truth được đọc bởi `read_disparity_png()` ở [data.py](../data.py)
- ảnh được normalize bằng `normalize_image()` ở [data.py](../data.py)

### 2.5 Cách patch được sample

Trong `RawKittiPatchDataset`:

- `_build_index()` ở [data.py](../data.py) duyệt qua disparity valid pixel
- mỗi pixel được kiểm tra:
  - disparity hợp lệ
  - không vượt biên patch trái/phải
  - không vượt biên search range
- các điểm hợp lệ được lưu vào `valid_points`
- `__getitem__()` ở [data.py](../data.py) lấy một điểm, cắt:
  - patch trái
  - patch phải rộng hơn theo `disp_range`
  - nhãn mềm `target`

Nhãn mềm được tạo bởi `build_target_distribution()` ở [data.py](../data.py).

### 2.6 Build model

- `run_training()` gọi `create_model()` ở [models.py](../models.py)
- `create_model()` trả về `SiameseStereoMatching`
- `SiameseStereoMatching` dùng:
  - `get_network_spec()` ở [models.py](../models.py)
  - `SiameseFeatureExtractor` ở [models.py](../models.py)

### 2.7 Kiến trúc model

- `NETWORK_SPECS` ở [models.py](../models.py)
  - `win19_dep9`: 9 lớp kernel `3x3`
  - `win37_dep9`: 9 lớp kernel `5x5`
- `SiameseFeatureExtractor` build backbone:
  - `Conv2d`
  - `BatchNorm2d`
  - `ReLU`
  - lớp cuối không có `ReLU`
- `SiameseStereoMatching.forward()` ở [models.py](../models.py):
  - chạy backbone cho patch trái/phải
  - ép patch trái về vector `1 x 64`
  - patch phải thành chuỗi vector theo disparity
  - tính inner product bằng `torch.einsum`

### 2.8 Training loop

Trong [train.py](../train.py):

- `make_loader()` ở [train.py](../train.py) tạo `DataLoader`
- `infinite_loader()` ở [train.py](../train.py) cho phép lặp batch vô hạn
- optimizer được tạo bởi `build_optimizer()` ở [train.py](../train.py)
- learning rate được cập nhật bởi `update_learning_rate()` ở [train.py](../train.py)

Mỗi step:

1. lấy `left`, `right`, `targets`
2. đẩy lên device
3. `model(left, right)` ở [models.py](../models.py)
4. tính loss bằng `soft_target_cross_entropy()` ở [models.py](../models.py)
5. `loss.backward()`
6. `optimizer.step()`

### 2.9 Checkpoint

- cứ mỗi `eval_every` step:
  - in `step=... loss=... lr=...`
  - gọi `save_checkpoint()` ở [train.py](../train.py)
- `save_checkpoint()` lưu:
  - `checkpoint.pt`
  - `config.json`

## 3. Flow evaluate

Lệnh:

```bash
cd .
python -m dl_stereo_matching_pytorch.evaluate ...
```

### 3.1 Entry

- Python gọi `main()` ở [evaluate.py](../evaluate.py)
- `parse_args()` ở [evaluate.py](../evaluate.py)
- `get_network_spec()` được dùng để check `patch_size` ở [evaluate.py](../evaluate.py)

### 3.2 Build validation dataset

- `main()` gọi `build_patch_datasets()` ở [evaluate.py](../evaluate.py)
- ở đây chỉ lấy `val_dataset`, còn train dataset bị bỏ qua
- nếu là KITTI raw thì validation set được dựng bằng:
  - `RawKittiPatchDataset(... deterministic=True)` ở [data.py](../data.py)

Điểm này quan trọng:
- `deterministic=True` nghĩa là evaluation dùng danh sách patch cố định theo index, không random mỗi lần lấy mẫu trong một lượt evaluate

### 3.3 Load model

- checkpoint được load ở [evaluate.py](../evaluate.py)
- model được tạo lại bằng `create_model()` ở [evaluate.py](../evaluate.py)
- `model.load_state_dict()` ở [evaluate.py](../evaluate.py)
- `model.eval()` ở [evaluate.py](../evaluate.py)

### 3.4 Tính 3-pixel accuracy

- loop batch ở [evaluate.py](../evaluate.py)
- `logits.argmax(dim=1)` ở [evaluate.py](../evaluate.py)
- so với `val_dataset.center_label` ở [evaluate.py](../evaluate.py)

Ý nghĩa:

- label đúng nằm ở giữa search range
- prediction lệch tối đa `3` index thì được tính là đúng

Cuối cùng:

- in `3-pixel accuracy=...` ở [evaluate.py](../evaluate.py)

## 4. Flow infer

Lệnh:

```bash
cd .
python -m dl_stereo_matching_pytorch.infer ...
```

### 4.1 Entry

- Python gọi `main()` ở [infer.py](../infer.py)
- `parse_args()` ở [infer.py](../infer.py)
- tạo `out_dir` ở [infer.py](../infer.py)

### 4.2 Chọn danh sách ảnh test

`infer.py` có 2 nhánh:

- nếu có `myPerm.bin`:
  - dùng `load_file_ids()` ở [infer.py](../infer.py)
  - đọc ảnh bằng `read_stereo_pair()` ở [infer.py](../infer.py)
- nếu dùng KITTI raw:
  - duyệt `testing/image_0/*_10.png` ở [infer.py](../infer.py)
  - đọc ảnh bằng `read_stereo_pair_by_name()` ở [infer.py](../infer.py)

### 4.3 Load model

- load checkpoint ở [infer.py](../infer.py)
- tạo model ở [infer.py](../infer.py)
- `load_state_dict()` ở [infer.py](../infer.py)
- `model.eval()` ở [infer.py](../infer.py)

### 4.4 Full-image inference

Mỗi ảnh test:

1. convert ảnh trái/phải sang tensor ở [infer.py](../infer.py)
2. lấy feature map bằng `model.extract_feature_maps()` ở [infer.py](../infer.py)
3. dựng cost volume bằng `build_cost_volume()` ở [infer.py](../infer.py)
4. lấy `argmax` theo disparity ở [infer.py](../infer.py)
5. lưu ảnh disparity bằng `save_disparity()` ở [infer.py](../infer.py)

### 4.5 Cost volume được dựng như thế nào

- `build_cost_volume()` nằm ở [models.py](../models.py)
- hàm này:
  - duyệt từng `disparity`
  - shift feature map phải
  - nhân từng channel với feature map trái
  - cộng theo channel để ra matching score

Đây chính là bản full-image của ý tưởng inner product trong paper.

## 5. File output của từng flow

### Train

Trong `--model-dir`:

- `checkpoint.pt`
- `config.json`
- `train.log`

### Evaluate

- không ghi file mới
- chỉ in `3-pixel accuracy=...`

### Infer

Trong `--out-dir`:

- `disp_map_<sample>.png`

## 6. Những chỗ hay cần đọc khi debug

- lỗi config model / patch size:
  - [train.py](../train.py)
  - [evaluate.py](../evaluate.py)
- lỗi không tìm thấy data:
  - [data.py](../data.py)
  - [data.py](../data.py)
- lỗi không tìm thấy checkpoint:
  - [evaluate.py](../evaluate.py)
  - [infer.py](../infer.py)
- lỗi shape khi forward:
  - [models.py](../models.py)
  - [models.py](../models.py)

## 7. Kết luận

Flow hiện tại của folder `dl_stereo_matching_pytorch` là:

1. `train.py` build dataset -> build model -> loop train -> save checkpoint/log
2. `evaluate.py` build validation dataset -> load checkpoint -> tính `3-pixel accuracy`
3. `infer.py` load checkpoint -> chạy full-image feature extraction -> build cost volume -> lưu disparity map

Tức là folder này hiện đã là một flow PyTorch hoàn chỉnh, tách riêng với folder TensorFlow cũ.
