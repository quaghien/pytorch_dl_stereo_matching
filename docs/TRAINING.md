# Train Trong Repo Này Diễn Ra Như Thế Nào

Tài liệu này giải thích theo góc nhìn thực hành:

- dữ liệu train được lấy từ đâu
- có resize hay augment gì không
- một sample train gồm những gì
- input, output, target là gì
- loss tính thế nào và có ý nghĩa gì
- model CNN trong repo có kiến trúc ra sao

Ảnh kiến trúc model:

- [model_architecture.svg](../figures/model_architecture.svg)

Hình này có ghi rõ:

- shape tensor ở từng bước
- vì sao `matching head` cho ra `disp_range` score
- công thức inner product ở bước matching
- công thức `soft target cross entropy` dùng để train

## 1. Hiểu nhanh trong 1 phút

Repo này train theo kiểu `patch-based stereo matching`.

Nói đơn giản:

- không đưa cả ảnh lớn vào train trực tiếp
- mà cắt ra từng ô nhỏ bên ảnh trái
- lấy một dải ô tương ứng bên ảnh phải
- model học xem vị trí nào trong dải bên phải là khớp đúng

Train dùng:

- ảnh trái từ `training/image_0`
- ảnh phải từ `training/image_1`
- disparity ground truth từ `training/disp_noc`

Code hiện tại:

- không resize ảnh trước khi train
- không flip
- không rotate
- không color jitter
- không random crop toàn ảnh

Biến đổi dữ liệu chính chỉ là:

- chuẩn hoá ảnh
- lọc các pixel hợp lệ
- cắt patch trái và vùng tìm kiếm bên phải

## 2. Dữ liệu train đi từ đâu vào đâu

Với dữ liệu hiện có trong repo, train đi theo nhánh KITTI raw trong [data.py](../data.py).

Luồng chính:

1. Đọc danh sách sample từ `training/disp_noc/*.png`
2. Đọc ảnh trái từ `training/image_0`
3. Đọc ảnh phải từ `training/image_1`
4. Đọc disparity ground truth từ `training/disp_noc`
5. Duyệt các pixel hợp lệ để tạo danh sách điểm train
6. Khi lấy batch thì cắt patch trái và patch phải từ các điểm đó

Điểm quan trọng:

- repo train theo `pixel hợp lệ` rồi sinh patch
- không train theo kiểu “1 sample là 1 cặp ảnh full-size”

## 3. Dữ liệu có resize hay augment không?

### Không có resize

Code hiện tại không resize ảnh trước khi train.

Ảnh được đọc nguyên kích thước từ file PNG.

### Không có data augmentation

Code hiện tại không thấy các bước như:

- lật ảnh ngang/dọc
- xoay ảnh
- đổi sáng/tối/ngẫu nhiên màu
- blur
- noise
- random affine

### Có normalize

Ảnh được chuẩn hoá bằng [normalize_image()](../data.py):

```python
(image - mean) / std
```

Ý nghĩa:

- đưa giá trị ảnh về thang ổn định hơn
- giúp model học dễ hơn
- giảm việc ảnh quá sáng hoặc quá tối làm lệch phân phối đầu vào

Nếu `std` quá nhỏ thì code thay bằng `1.0` để tránh chia cho `0`.

## 4. Một sample train gồm những gì

Nếu nói theo dữ liệu gốc của một cảnh, một sample như `000000_10` gồm:

- 1 ảnh trái: `image_0/000000_10.png`
- 1 ảnh phải: `image_1/000000_10.png`
- 1 ảnh disparity ground truth: `disp_noc/000000_10.png`

Nhưng nếu nói theo đúng thứ DataLoader trả ra cho model ở lúc train, thì một sample train gồm:

- `left_patch`
- `right_patch`
- `target`

### `left_patch`

Là một ô nhỏ cắt từ ảnh trái.

Với model `win37_dep9`:

- `patch_size = 37`
- nên `left_patch` có kích thước `37 x 37`

Với KITTI 2012 là ảnh xám:

- tensor đầu vào có shape `1 x 37 x 37`

### `right_patch`

Không phải chỉ là một ô nhỏ đúng bằng patch trái.

Nó là một dải rộng hơn, để chứa nhiều vị trí disparity ứng viên.

Kích thước:

- cao bằng `patch_size`
- rộng bằng `patch_size + disp_range - 1`

Ví dụ với:

- `patch_size = 37`
- `disp_range = 256`

thì `right_patch` sẽ có shape:

- `1 x 37 x 292`

### `target`

`target` là nhãn đúng cho sample đó.

Repo này không dùng one-hot cứng, mà dùng nhãn mềm:

```text
[0.05, 0.2, 0.5, 0.2, 0.05]
```

Nhãn mềm này được đặt quanh vị trí disparity đúng ở giữa search range.

## 5. Một sample train được tạo như thế nào

Giả sử code chọn một pixel hợp lệ ở ảnh trái tại:

- `(x, y)`

và disparity đúng tại điểm đó là:

- `d`

Khi đó:

1. Code cắt một ô `left_patch` quanh `(x, y)` bên ảnh trái
2. Tính điểm tương ứng bên phải là:
   - `right_x = x - d`
3. Cắt một dải `right_patch` quanh `right_x`
4. Đặt nhãn đúng ở giữa dải tìm kiếm
5. Đưa cặp patch này cho model học

Hiểu đơn giản:

- patch trái là “mảnh ảnh cần đi tìm”
- patch phải là “vùng để tìm xem nó khớp ở đâu”

## 6. Input và output của model là gì

### Input

Model nhận 2 tensor:

- `left`: batch các patch trái
- `right`: batch các dải patch phải

Ví dụ với `win37_dep9`, KITTI 2012, batch size `B`:

- `left.shape = (B, 1, 37, 37)`
- `right.shape = (B, 1, 37, 292)` nếu `disp_range=256`

### Output

Model trả ra `logits` cho từng disparity ứng viên:

- `logits.shape = (B, disp_range)`

Nếu `disp_range = 256`:

- mỗi sample có `256` điểm số
- điểm số lớn hơn nghĩa là model tin disparity đó khớp hơn

## 7. Model kiến trúc như thế nào

Code model nằm ở [models.py](../models.py).

Repo có 2 cấu hình chính:

- `win19_dep9`
- `win37_dep9`

### `win19_dep9`

- 9 lớp convolution
- tất cả kernel `3x3`
- số kênh mỗi lớp là `64`
- receptive field là `19`

Nên:

- patch trái phải có kích thước `19 x 19`

### `win37_dep9`

- 9 lớp convolution
- tất cả kernel `5x5`
- số kênh là:
  - `32, 32, 64, 64, 64, 64, 64, 64, 64`
- receptive field là `37`

Nên:

- patch trái phải có kích thước `37 x 37`

### Backbone là Siamese CNN

`Siamese` nghĩa là:

- cùng một mạng CNN
- dùng chung trọng số
- chạy cho cả patch trái và patch phải

Mỗi block gồm:

- `Conv2d`
- `BatchNorm2d`
- `ReLU`

Riêng lớp cuối:

- không có `ReLU`

Ý nghĩa:

- patch trái và patch phải được mã hoá về cùng không gian đặc trưng
- sau đó mới đem so khớp

## 8. Cách model tạo điểm số matching

Sau khi đi qua backbone:

- `left_patch` được nén về feature map `1 x 1`
- `right_patch` được biến thành một dãy feature vector theo chiều ngang

Sau đó model tính inner product:

```text
score(k) = f_left . f_right(k)
```

Trong đó:

- `f_left` là vector đặc trưng của patch trái
- `f_right(k)` là vector đặc trưng tại vị trí disparity ứng viên thứ `k`

Ý nghĩa:

- nếu 2 vector giống nhau hơn thì tích vô hướng lớn hơn
- điểm số lớn hơn nghĩa là vị trí đó có vẻ khớp hơn

## 9. Loss là gì và công thức ra sao

Loss được định nghĩa ở [soft_target_cross_entropy()](../models.py).

Code:

```python
log_probs = torch.log_softmax(logits, dim=1)
loss = -(targets * log_probs).sum(dim=1).mean()
```

Viết theo công thức:

```text
L = - (1 / B) * sum_i sum_d t_{i,d} log p_{i,d}
```

Trong đó:

- `B` là batch size
- `t_{i,d}` là target mềm của sample `i` tại disparity `d`
- `p_{i,d}` là xác suất model dự đoán sau `softmax`

### Ý nghĩa của loss này

Nếu model đặt xác suất cao vào disparity đúng và các vị trí rất gần nó:

- loss sẽ nhỏ

Nếu model đặt xác suất cao vào disparity sai:

- loss sẽ lớn

### Vì sao dùng nhãn mềm thay vì one-hot cứng?

Vì trong stereo:

- disparity đúng và disparity lệch 1 pixel thường không khác nhau quá thảm hoạ
- dùng nhãn mềm giúp model học “mềm” hơn quanh vị trí đúng
- ổn định hơn so với bắt model chỉ được đúng đúng 1 vị trí duy nhất

## 10. Target mềm trông như thế nào

Repo dùng mặc định:

```text
[0.05, 0.2, 0.5, 0.2, 0.05]
```

Ý nghĩa:

- vị trí đúng nhất ở giữa có trọng số `0.5`
- lệch 1 ô vẫn còn được `0.2`
- lệch 2 ô vẫn còn được `0.05`

Hiểu dễ:

- đúng tâm là tốt nhất
- gần đúng vẫn được chấp nhận một phần
- càng lệch xa càng bị phạt

## 11. Train loop chạy như thế nào

Ở [train.py](../train.py), mỗi step train làm như sau:

1. Lấy một batch patch trái, patch phải, target
2. Chạy model để ra `logits`
3. Tính `soft target cross entropy`
4. `backward()`
5. `optimizer.step()`

Optimizer hỗ trợ:

- `adam`
- `adagrad`
- `sgd`

Preset khuyến nghị trong repo hiện tại là:

- `win37_dep9`
- `Adam`
- `weight_decay = 5e-4`

## 12. Validate đo cái gì

Ở [evaluate.py](../evaluate.py), code:

- lấy `argmax` của output
- so với vị trí center label
- nếu lệch không quá `3` thì tính là đúng

Nên metric đang in ra là:

- `3-pixel accuracy`

Ý nghĩa:

- model đoán vị trí disparity gần đúng trong phạm vi 3 pixel thì vẫn được tính đúng

## 13. Không có những gì trong pipeline hiện tại

Để tránh hiểu nhầm, pipeline train hiện tại chưa có các bước sau:

- không resize ảnh
- không augment ảnh
- không fine-tune theo màu vì train mặc định dùng ảnh xám
- không dùng calibration trong train patch
- không có hậu xử lý kiểu CRF/smoothing trong pipeline train

## 14. Kết luận rất ngắn

Nếu chỉ nhớ 5 ý:

1. Train dùng `image_0`, `image_1`, `disp_noc` trong `training`.
2. Không có resize và không có augmentation.
3. Mỗi sample train thực tế là `left_patch + right_patch + soft target`.
4. Model là Siamese CNN 9 lớp, so khớp bằng inner product.
5. Loss là soft-target cross entropy để học disparity đúng theo cách mềm hơn one-hot.
