# Giải Thích Các Tham Số Train

File này giải thích các tham số trong `train.py`, tương ứng với phần `parse_args()`.

Mục tiêu:

- biết mỗi tham số dùng để làm gì
- biết tham số nào ảnh hưởng dữ liệu
- biết tham số nào ảnh hưởng model
- biết tham số nào ảnh hưởng tốc độ train, VRAM và chất lượng

## 1. Nhóm tham số điều khiển train

### `--batch-size`

Mặc định:

```text
128
```

Ý nghĩa:

- số sample train trong một batch
- ở repo này, `1 sample` là `1 cặp patch trái/phải + target`

Tăng lên thì:

- train thường nhanh hơn về throughput
- dùng nhiều VRAM hơn
- gradient ổn định hơn một chút

Giảm xuống thì:

- đỡ tốn VRAM
- có thể train chậm hơn
- gradient nhiễu hơn

### `--num-iter`

Mặc định:

```text
40000
```

Ý nghĩa:

- số bước train tổng cộng
- không phải số epoch

Ví dụ:

- `num-iter = 40000` nghĩa là optimizer update `40000` lần rồi dừng

### `--eval-every`

Mặc định:

```text
100
```

Ý nghĩa:

- cứ mỗi bao nhiêu step thì in log và lưu checkpoint

Ví dụ:

- `eval-every = 100`
- sau mỗi `100` step sẽ in loss trung bình và lưu `checkpoint.pt`

## 2. Nhóm tham số đường dẫn

### `--model-dir`

Mặc định:

```text
model
```

Ý nghĩa:

- thư mục lưu output train

Trong đó thường có:

- `checkpoint.pt`
- `config.json`
- `train.log`

### `--data-root`

Bắt buộc truyền.

Ý nghĩa:

- thư mục gốc chứa `training/` và `testing/`

Ví dụ:

```text
dl_stereo_matching_pytorch/data
```

### `--util-root`

Bắt buộc truyền.

Ý nghĩa:

- thư mục chứa dữ liệu hỗ trợ kiểu preprocess cũ như `myPerm.bin`
- nếu không có `myPerm.bin`, code sẽ fallback sang KITTI raw

Trong repo hiện tại, thường cũng trỏ cùng chỗ với `data-root`.

## 3. Nhóm tham số chọn dữ liệu

### `--data-version`

Mặc định:

```text
kitti2012
```

Chọn một trong:

- `kitti2012`
- `kitti2015`

Ý nghĩa:

- chọn mapping thư mục ảnh trái/phải
- đồng thời quyết định số channel đầu vào

Trong code:

- `kitti2012` dùng `image_0` và `image_1`
- `kitti2015` dùng `image_2` và `image_3`

### `--num-tr-img`

Mặc định:

```text
160
```

Ý nghĩa:

- số sample ảnh dùng cho train

Với KITTI raw trong repo hiện tại:

- code lấy danh sách sample từ `training/disp_noc`
- sau đó lấy `160` sample đầu cho train

### `--num-val-img`

Mặc định:

```text
34
```

Ý nghĩa:

- số sample ảnh dùng cho validation

Trong preset hiện tại:

- `160` train
- `34` val
- tổng là `194` sample có ground truth

### `--num-val-loc`

Mặc định:

```text
50000
```

Ý nghĩa:

- số patch validation tối đa dùng để evaluate trong mỗi lần validate

Hiểu đơn giản:

- validation không nhất thiết phải duyệt mọi điểm hợp lệ của mọi ảnh
- có thể chỉ lấy tối đa `50000` điểm

## 4. Nhóm tham số liên quan patch và disparity

### `--patch-size`

Mặc định:

```text
37
```

Ý nghĩa:

- kích thước patch trái
- đồng thời là chiều cao patch phải

Lưu ý:

- giá trị này phải khớp receptive field của model

Ví dụ:

- `win37_dep9` cần `patch-size = 37`
- `win19_dep9` cần `patch-size = 19`

### `--disp-range`

Mặc định:

```text
256
```

Ý nghĩa:

- số disparity candidate model sẽ xét

Ảnh hưởng trực tiếp đến:

- bề rộng vùng tìm kiếm bên phải
- số score đầu ra của model

Ví dụ:

- nếu `disp-range = 256`
- output của model có `256` score
- patch phải sẽ rộng `patch_size + 256 - 1`

### `--train-samples-per-epoch`

Mặc định:

```text
50000
```

Ý nghĩa:

- số patch train random được lấy trong một “epoch logic”

Lưu ý:

- ở repo này train theo patch, không train theo full ảnh
- nên tham số này quan trọng hơn khái niệm epoch theo số ảnh

Hiểu đơn giản:

- càng lớn thì mỗi vòng train model thấy nhiều patch hơn
- nhưng một vòng cũng lâu hơn

## 5. Nhóm tham số model

### `--net-type`

Mặc định:

```text
win37_dep9
```

Chọn một trong:

- `win19_dep9`
- `win37_dep9`

Ý nghĩa:

- chọn kiến trúc CNN

`win19_dep9`:

- receptive field `19`
- patch nhỏ hơn

`win37_dep9`:

- receptive field `37`
- patch lớn hơn
- thường ưu tiên chất lượng hơn trong preset hiện tại

## 6. Nhóm tham số optimizer

### `--optimizer`

Mặc định:

```text
adam
```

Chọn một trong:

- `adam`
- `adagrad`
- `sgd`

Ý nghĩa:

- chọn thuật toán cập nhật trọng số

Thực tế:

- `adam`: thường dễ train ổn định
- `adagrad`: gần với một số thiết lập cổ điển
- `sgd`: cần tuning kỹ hơn

### `--learning-rate`

Mặc định:

```text
1e-3
```

Ý nghĩa:

- tốc độ học của optimizer

Nếu quá lớn:

- loss có thể dao động mạnh hoặc diverge

Nếu quá nhỏ:

- train chậm, khó tiến bộ

### `--weight-decay`

Mặc định:

```text
5e-4
```

Ý nghĩa:

- regularization lên trọng số
- giúp giảm overfitting phần nào

Hiểu đơn giản:

- phạt trọng số quá lớn

### `--momentum`

Mặc định:

```text
0.9
```

Ý nghĩa:

- chỉ dùng khi `optimizer = sgd`
- giúp cập nhật mượt hơn và tích luỹ hướng gradient

Nếu dùng `adam` hoặc `adagrad` thì tham số này không phải yếu tố chính.

## 7. Nhóm tham số device

### `--device`

Mặc định:

```text
cuda nếu có GPU, ngược lại là cpu
```

Ý nghĩa:

- chọn thiết bị train

Ví dụ:

- `--device cuda`
- `--device cpu`

Thực tế:

- train trên `cuda` nhanh hơn rất nhiều
- `cpu` thường chỉ phù hợp để test nhỏ hoặc debug

## 8. Ví dụ preset trong repo có nghĩa là gì

Preset hiện tại:

```text
--net-type win37_dep9
--patch-size 37
--disp-range 256
--optimizer adam
--learning-rate 1e-3
--weight-decay 5e-4
--num-tr-img 160
--num-val-img 34
--num-val-loc 5000
--train-samples-per-epoch 50000
--batch-size 128
--num-iter 40000
--eval-every 100
```

Diễn giải dễ hiểu:

- dùng model patch lớn hơn là `win37_dep9`
- mỗi sample train là patch trái `37x37`
- vùng tìm disparity gồm `256` ứng viên
- train bằng `Adam`
- train trên `160` ảnh, validate trên `34` ảnh
- mỗi vòng logic lấy `50000` patch train
- mỗi batch có `128` patch
- train tổng cộng `40000` bước
- cứ `100` bước thì lưu checkpoint và ghi log

## 9. Nên nhớ nhanh thế nào

Nếu chỉ cần nhớ nhanh:

- `data-root`, `util-root`, `data-version`: chọn dữ liệu
- `num-tr-img`, `num-val-img`, `num-val-loc`: chọn lượng train/val
- `patch-size`, `disp-range`: chọn cách tạo sample stereo
- `net-type`: chọn model
- `batch-size`, `num-iter`, `eval-every`: điều khiển quá trình train
- `optimizer`, `learning-rate`, `weight-decay`, `momentum`: điều khiển tối ưu
- `device`: chọn CPU hay GPU
