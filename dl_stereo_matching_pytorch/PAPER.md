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
