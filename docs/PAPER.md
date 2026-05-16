# Ghi chú paper: Efficient Deep Learning for Stereo Matching

Nguồn paper: https://www.cs.toronto.edu/~urtasun/publications/luo_etal_cvpr16.pdf

## 1. Bài toán

Paper giải bài toán stereo matching trên cặp ảnh đã rectified. Mục tiêu là dự đoán disparity cho từng pixel ảnh trái bằng cách so khớp với ảnh phải theo trục ngang.

## 2. Ý tưởng chính

Khác với các mô hình trước đó dùng siamese network rồi nối đặc trưng hai nhánh bằng phép `concatenate + MLP`, paper này thay bằng:

- hai nhánh CNN chia sẻ trọng số để trích đặc trưng patch trái/phải
- một `product layer` tính inner product trực tiếp giữa hai biểu diễn
- huấn luyện theo phân phối xác suất trên toàn bộ disparity thay vì phân loại nhị phân match / non-match

Điểm quan trọng là inner product giúp suy luận nhanh hơn rất nhiều vì có thể tái sử dụng feature map cho nhiều disparity.

## 3. Kiến trúc

Paper mô tả các nhánh CNN gồm nhiều lớp tích chập nhỏ, mỗi lớp đi kèm batch normalization và ReLU, trừ lớp cuối không dùng ReLU để giữ cả giá trị âm.

Các biến thể xuất hiện trong repo gốc:

- `win19_dep9`: 9 lớp `3x3`, receptive field `19x19`
- `win37_dep9`: 9 lớp `5x5`, receptive field `37x37`

Trong huấn luyện patch-level:

- patch trái có kích thước đúng bằng receptive field
- patch phải rộng hơn để chứa mọi disparity ứng viên
- nhánh trái cho ra vector `64-D`
- nhánh phải cho ra `disp_range` vector `64-D`
- inner product tạo logits cho toàn bộ disparity

## 4. Loss và nhãn mềm

Paper không dùng nhãn one-hot tuyệt đối. Thay vào đó, họ dùng phân phối mềm quanh ground-truth disparity:

- đúng disparity: `0.5`
- lệch `1` pixel: `0.2`
- lệch `2` pixel: `0.05`
- còn lại: `0`

Loss là cross-entropy trên phân phối disparity. Lựa chọn này phù hợp với thước đo `3-pixel error`.

## 4.1 Hiểu trực giác: dot product ra vector score, vậy ground truth cho vector đó là gì?

Đây là chỗ dễ bị mơ hồ nhất nếu chỉ nhìn công thức.

### Model trả ra gì?

Với một sample train, model nhận:

- một `left_patch`
- một `right_patch_strip`

Sau backbone:

- nhánh trái ra đúng `1` vector đặc trưng
- nhánh phải ra một dãy vector đặc trưng theo chiều ngang

Sau đó matching head tính:

```text
score(k) = f_left . f_right(k)
```

với:

- `f_left`: vector đặc trưng của patch trái
- `f_right(k)`: vector đặc trưng ở vị trí disparity ứng viên thứ `k`

Nên đầu ra cuối là:

- một vector `score`
- độ dài bằng `disp_range`

Ví dụ nếu `disp_range = 256` thì output là:

```text
[score(0), score(1), ..., score(255)]
```

Mỗi phần tử trả lời câu hỏi:

- "nếu disparity là `k` thì mức khớp tốt đến đâu?"

### Ground truth cho vector score được lấy từ đâu?

Ground truth không phải do model tự suy ra.

Nó được lấy trực tiếp từ ảnh disparity ground truth:

- `training/disp_noc/*.png`

Trong code:

- chọn một pixel hợp lệ ở ảnh trái tại tọa độ `(x, y)`
- đọc disparity thật tại điểm đó:

```text
d = disparity[y, x]
```

Ở nhánh KITTI raw, chỗ này nằm ở [data.py](../data.py).

Sau đó code tính:

```text
right_x = x - d
```

tức là:

- biết pixel trái nằm ở đâu
- biết độ lệch ngang thật là bao nhiêu
- thì suy ra điểm tương ứng bên ảnh phải nằm ở đâu

### Vậy “điểm đúng” trong vector score là điểm nào?

Trong repo này, `right_patch_strip` được cắt sao cho:

- vị trí đúng rơi vào giữa dải tìm kiếm

Nên ground truth label không phải là một chỉ số ngẫu nhiên chạy lung tung.

Nó được chuẩn hoá về giữa vector output.

Trong code:

- `center_label = disp_range // 2`

nghĩa là:

- nếu `disp_range = 256`
- thì vị trí "đúng tâm" là index `128`

Tức là model luôn được huấn luyện theo kiểu:

- vị trí đúng nhất nằm ở giữa vector output
- các vị trí lệch sang trái/phải quanh nó là các disparity gần đúng

### Vì sao làm như vậy?

Vì sample train không cắt một patch phải đúng khít tại đúng điểm match.

Thay vào đó, nó cắt một dải rộng hơn quanh `right_x`.

Nên:

- điểm đúng nằm trong dải đó
- và được đặt vào giữa để mọi sample có cùng quy ước nhãn

Nhờ vậy, output luôn có cùng ý nghĩa:

- index giữa là disparity đúng
- index bên trái/phải là lệch ít nhiều so với disparity đúng

### Ví dụ số rất cụ thể

Giả sử:

- lấy một pixel trên thân cây ở ảnh trái
- tọa độ là `(x=500, y=200)`
- từ `disp_noc`, đọc được disparity thật là `d = 8`

Khi đó:

```text
right_x = 500 - 8 = 492
```

Nghĩa là:

- điểm tương ứng của pixel thân cây đó ở ảnh phải nằm tại `x = 492`

Code sẽ làm:

1. Cắt `left_patch` quanh `(500, 200)`
2. Cắt `right_patch_strip` quanh vùng chứa `x = 492`
3. Sắp vùng đó sao cho vị trí đúng nằm ở giữa output
4. Model sinh ra vector `256` score
5. Ground truth bảo rằng:
   - score ở giữa phải lớn nhất
   - score lệch `1` pixel vẫn được điểm một phần
   - score lệch `2` pixel vẫn được điểm ít hơn

Nếu dùng nhãn mềm mặc định thì target trông như:

```text
..., 0, 0.05, 0.2, 0.5, 0.2, 0.05, 0, ...
```

trong đó:

- `0.5` nằm ở index giữa
- hai bên là các vị trí gần đúng

### Nói cực ngắn theo trực giác

Model làm việc như sau:

1. Patch trái được mã hoá thành 1 vector
2. Dải patch phải được mã hoá thành nhiều vector
3. Model chấm điểm từng vector phải với vector trái
4. Ground truth nói:
   - vị trí giữa mới là đúng nhất
   - quanh nó cũng được chấp nhận một phần

Nên:

- vector score là "mức khớp theo từng disparity ứng viên"
- ground truth là "phân phối xác suất mong muốn trên vector đó"

## 5. Suy luận

Khi test full image:

- chạy CNN một lần trên toàn ảnh trái và ảnh phải để lấy feature map
- với mỗi disparity, dịch đặc trưng ảnh phải
- tính inner product với đặc trưng ảnh trái
- ghép thành cost volume và lấy `argmax`

Theo paper, cách này nhanh hơn rõ rệt so với kiến trúc concatenate-based trước đó.

## 6. Hậu xử lý

Paper có thêm phần smoothing trên output của deep net bằng các mô hình regularization hậu xử lý. Tuy nhiên repo TensorFlow hiện tại trong `dl_stereo_matching` chủ yếu cover phần matching network và suy luận raw disparity map, chưa bao gồm toàn bộ pipeline smoothing trong bài báo.

## 7. Kết nối với repo hiện có

Repo TensorFlow ở `dl_stereo_matching` giữ lại đúng phần cốt lõi sau:

- data loader lấy patch từ KITTI và các file `.bin` preprocess
- hai mạng `win19_dep9` và `win37_dep9`
- train bằng AdaGrad
- evaluate patch-level theo 3-pixel criterion
- infer full image bằng feature reuse + inner product

Bản port PyTorch mới trong `dl_stereo_matching_pytorch` tái tạo đúng những phần này, nhưng được tổ chức lại thành code độc lập hơn, dễ test hơn và không phụ thuộc TensorFlow 1.x.
