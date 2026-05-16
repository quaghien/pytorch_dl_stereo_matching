# Code Flow: Train, Evaluate, Infer

Tài liệu này mô tả luồng chạy chính của folder `dl_stereo_matching_pytorch` theo đúng code hiện tại, gồm:

- train
- evaluate
- infer

Ảnh workflow:

- Train: [train_workflow.svg](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/figures/train_workflow.svg)
- Infer: [infer_workflow.svg](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/figures/infer_workflow.svg)

Mục tiêu là để đọc từ trên xuống và lần được:

- lệnh CLI đi vào file nào
- hàm nào gọi hàm nào
- model, data loader, checkpoint và output chạy qua đâu

## 1. Tổng quan file chính

- [train.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/train.py:15): entrypoint train
- [evaluate.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/evaluate.py:13): entrypoint evaluate
- [infer.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/infer.py:14): entrypoint infer full image
- [data.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/data.py:19): đọc KITTI, build patch dataset
- [models.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/models.py:10): định nghĩa kiến trúc `win19_dep9`, `win37_dep9`, loss và cost volume

## 2. Flow train

Lệnh:

```bash
python -m dl_stereo_matching_pytorch.train ...
```

### 2.1 Entry

- Python gọi `main()` ở [train.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/train.py:104)
- `main()` đọc CLI qua `parse_args()` ở [train.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/train.py:15)
- `main()` kiểm tra `patch_size` có khớp với model qua `validate_patch_size()` ở [train.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/train.py:39)

### 2.2 Logging và thư mục output

- `main()` tạo `model_dir` ở [train.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/train.py:107)
- `main()` mở file log `train.log` ở [train.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/train.py:110)
- `TeeStream` ở [train.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/train.py:55) dùng để ghi đồng thời ra terminal và file log
- sau đó `main()` gọi `run_training()` ở [train.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/train.py:124)

### 2.3 Build dataset

- `run_training()` gọi `build_patch_datasets()` ở [data.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/data.py:299)
- `build_patch_datasets()` có 2 nhánh:
  - nếu có `myPerm.bin` thì dùng dữ liệu preprocess kiểu cũ:
    - `KittiStereoCache` ở [data.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/data.py:90)
    - `load_locations()` ở [data.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/data.py:30)
    - `StereoPatchDataset` ở [data.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/data.py:112)
  - nếu không có `myPerm.bin` thì dùng KITTI raw:
    - `RawKittiStereoCache` ở [data.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/data.py:165)
    - `RawKittiPatchDataset` ở [data.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/data.py:202)

### 2.4 Cách dữ liệu KITTI raw được chuẩn bị

Trong nhánh raw:

- `RawKittiStereoCache.__post_init__()` đọc toàn bộ `training/image_0`, `training/image_1`, `training/disp_noc` ở [data.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/data.py:173)
- ảnh được đọc bởi `read_stereo_pair_by_name()` ở [data.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/data.py:62)
- disparity ground truth được đọc bởi `read_disparity_png()` ở [data.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/data.py:73)
- ảnh được normalize bằng `normalize_image()` ở [data.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/data.py:39)

### 2.5 Cách patch được sample

Trong `RawKittiPatchDataset`:

- `_build_index()` ở [data.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/data.py:227) duyệt qua disparity valid pixel
- mỗi pixel được kiểm tra:
  - disparity hợp lệ
  - không vượt biên patch trái/phải
  - không vượt biên search range
- các điểm hợp lệ được lưu vào `valid_points`
- `__getitem__()` ở [data.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/data.py:282) lấy một điểm, cắt:
  - patch trái
  - patch phải rộng hơn theo `disp_range`
  - nhãn mềm `target`

Nhãn mềm được tạo bởi `build_target_distribution()` ở [data.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/data.py:78).

### 2.6 Build model

- `run_training()` gọi `create_model()` ở [models.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/models.py:80)
- `create_model()` trả về `SiameseStereoMatching`
- `SiameseStereoMatching` dùng:
  - `get_network_spec()` ở [models.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/models.py:26)
  - `SiameseFeatureExtractor` ở [models.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/models.py:33)

### 2.7 Kiến trúc model

- `NETWORK_SPECS` ở [models.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/models.py:20)
  - `win19_dep9`: 9 lớp kernel `3x3`
  - `win37_dep9`: 9 lớp kernel `5x5`
- `SiameseFeatureExtractor` build backbone:
  - `Conv2d`
  - `BatchNorm2d`
  - `ReLU`
  - lớp cuối không có `ReLU`
- `SiameseStereoMatching.forward()` ở [models.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/models.py:65):
  - chạy backbone cho patch trái/phải
  - ép patch trái về vector `1 x 64`
  - patch phải thành chuỗi vector theo disparity
  - tính inner product bằng `torch.einsum`

### 2.8 Training loop

Trong [train.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/train.py:146):

- `make_loader()` ở [train.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/train.py:45) tạo `DataLoader`
- `infinite_loader()` ở [train.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/train.py:49) cho phép lặp batch vô hạn
- optimizer được tạo bởi `build_optimizer()` ở [train.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/train.py:84)
- learning rate được cập nhật bởi `update_learning_rate()` ở [train.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/train.py:93)

Mỗi step:

1. lấy `left`, `right`, `targets`
2. đẩy lên device
3. `model(left, right)` ở [models.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/models.py:65)
4. tính loss bằng `soft_target_cross_entropy()` ở [models.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/models.py:84)
5. `loss.backward()`
6. `optimizer.step()`

### 2.9 Checkpoint

- cứ mỗi `eval_every` step:
  - in `step=... loss=... lr=...`
  - gọi `save_checkpoint()` ở [train.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/train.py:70)
- `save_checkpoint()` lưu:
  - `checkpoint.pt`
  - `config.json`

## 3. Flow evaluate

Lệnh:

```bash
python -m dl_stereo_matching_pytorch.evaluate ...
```

### 3.1 Entry

- Python gọi `main()` ở [evaluate.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/evaluate.py:30)
- `parse_args()` ở [evaluate.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/evaluate.py:13)
- `get_network_spec()` được dùng để check `patch_size` ở [evaluate.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/evaluate.py:32)

### 3.2 Build validation dataset

- `main()` gọi `build_patch_datasets()` ở [evaluate.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/evaluate.py:36)
- ở đây chỉ lấy `val_dataset`, còn train dataset bị bỏ qua
- nếu là KITTI raw thì validation set được dựng bằng:
  - `RawKittiPatchDataset(... deterministic=True)` ở [data.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/data.py:344)

Điểm này quan trọng:
- `deterministic=True` nghĩa là evaluation dùng danh sách patch cố định theo index, không random mỗi lần lấy mẫu trong một lượt evaluate

### 3.3 Load model

- checkpoint được load ở [evaluate.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/evaluate.py:48)
- model được tạo lại bằng `create_model()` ở [evaluate.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/evaluate.py:50)
- `model.load_state_dict()` ở [evaluate.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/evaluate.py:51)
- `model.eval()` ở [evaluate.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/evaluate.py:52)

### 3.4 Tính 3-pixel accuracy

- loop batch ở [evaluate.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/evaluate.py:57)
- `logits.argmax(dim=1)` ở [evaluate.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/evaluate.py:60)
- so với `val_dataset.center_label` ở [evaluate.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/evaluate.py:61)

Ý nghĩa:

- label đúng nằm ở giữa search range
- prediction lệch tối đa `3` index thì được tính là đúng

Cuối cùng:

- in `3-pixel accuracy=...` ở [evaluate.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/evaluate.py:64)

## 4. Flow infer

Lệnh:

```bash
python -m dl_stereo_matching_pytorch.infer ...
```

### 4.1 Entry

- Python gọi `main()` ở [infer.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/infer.py:35)
- `parse_args()` ở [infer.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/infer.py:14)
- tạo `out_dir` ở [infer.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/infer.py:37)

### 4.2 Chọn danh sách ảnh test

`infer.py` có 2 nhánh:

- nếu có `myPerm.bin`:
  - dùng `load_file_ids()` ở [infer.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/infer.py:42)
  - đọc ảnh bằng `read_stereo_pair()` ở [infer.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/infer.py:43)
- nếu dùng KITTI raw:
  - duyệt `testing/image_0/*_10.png` ở [infer.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/infer.py:48)
  - đọc ảnh bằng `read_stereo_pair_by_name()` ở [infer.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/infer.py:53)

### 4.3 Load model

- load checkpoint ở [infer.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/infer.py:57)
- tạo model ở [infer.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/infer.py:59)
- `load_state_dict()` ở [infer.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/infer.py:60)
- `model.eval()` ở [infer.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/infer.py:61)

### 4.4 Full-image inference

Mỗi ảnh test:

1. convert ảnh trái/phải sang tensor ở [infer.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/infer.py:74)
2. lấy feature map bằng `model.extract_feature_maps()` ở [infer.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/infer.py:77)
3. dựng cost volume bằng `build_cost_volume()` ở [infer.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/infer.py:78)
4. lấy `argmax` theo disparity ở [infer.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/infer.py:79)
5. lưu ảnh disparity bằng `save_disparity()` ở [infer.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/infer.py:81)

### 4.5 Cost volume được dựng như thế nào

- `build_cost_volume()` nằm ở [models.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/models.py:89)
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
  - [train.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/train.py:39)
  - [evaluate.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/evaluate.py:32)
- lỗi không tìm thấy data:
  - [data.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/data.py:179)
  - [data.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/data.py:331)
- lỗi không tìm thấy checkpoint:
  - [evaluate.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/evaluate.py:48)
  - [infer.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/infer.py:57)
- lỗi shape khi forward:
  - [models.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/models.py:67)
  - [models.py](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/models.py:72)

## 7. Kết luận

Flow hiện tại của folder `dl_stereo_matching_pytorch` là:

1. `train.py` build dataset -> build model -> loop train -> save checkpoint/log
2. `evaluate.py` build validation dataset -> load checkpoint -> tính `3-pixel accuracy`
3. `infer.py` load checkpoint -> chạy full-image feature extraction -> build cost volume -> lưu disparity map

Tức là folder này hiện đã là một flow PyTorch hoàn chỉnh, tách riêng với folder TensorFlow cũ.
