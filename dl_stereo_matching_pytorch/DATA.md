# Dữ liệu Trong Repo Và Cách Code Dùng Dữ Liệu

File này giải thích theo kiểu dễ hiểu:

- mỗi thư mục ảnh là gì
- ảnh nào dùng để train
- ảnh nào chỉ để test/infer
- vì sao nhìn thấy nhiều file ảnh nhưng code chỉ train trên `194` sample

Nếu bạn mới nhìn repo lần đầu, chỉ cần đọc mục `1`, `2`, `3`, `4` là sẽ nắm được ý chính.

## 1. Hiểu nhanh trong 1 phút

Repo này có 2 phần dữ liệu chính:

- `data/training`: dữ liệu để train và validate
- `data/testing`: dữ liệu để chạy infer

Điểm quan trọng nhất:

- code train chỉ dùng `training`
- code không dùng `testing` để train
- với KITTI 2012 trong repo này, chỉ có `194` sample train được vì chỉ có `194` file ground truth disparity trong `training/disp_noc`

Nói ngắn gọn:

- `388` ảnh trong `training/image_0` không có nghĩa là có `388` sample train độc lập
- đó là vì mỗi cảnh có 2 frame: `*_10` và `*_11`
- nhưng ground truth disparity để train chỉ có cho frame `*_10`
- nên số sample train được là `194`

## 2. Một vài từ hay gặp, giải thích thật đơn giản

### Sample là gì?

Một `sample` có thể hiểu là:

- một cảnh chụp stereo hoàn chỉnh
- gồm ảnh trái, ảnh phải, và nếu là tập train thì có thêm ảnh disparity ground truth

Ví dụ sample `000000_10` gồm:

- ảnh trái xám
- ảnh phải xám
- ảnh trái màu
- ảnh phải màu
- ảnh disparity đúng
- file calib

### Stereo là gì?

`Stereo` nghĩa là có 2 ảnh chụp cùng một cảnh:

- 1 ảnh từ camera trái
- 1 ảnh từ camera phải

Model nhìn sự lệch nhau giữa 2 ảnh để đo độ sâu.

### Disparity là gì?

`Disparity` là độ lệch ngang giữa điểm ở ảnh trái và điểm tương ứng ở ảnh phải.

Hiểu đơn giản:

- vật càng gần camera thì thường lệch nhiều hơn
- vật càng xa thì lệch ít hơn

Model của repo này học để dự đoán độ lệch đó.

### Ground truth là gì?

`Ground truth` là đáp án đúng của dữ liệu.

Trong repo này:

- `disp_noc/*.png` là ảnh chứa disparity đúng
- code dùng nó để biết patch nào là đúng khi train

### Infer là gì?

`Infer` nghĩa là chạy model đã train xong lên ảnh mới để dự đoán kết quả.

Ở repo này:

- train dùng `training`
- infer thường chạy trên `testing`

### Patch là gì?

`Patch` là một ô ảnh nhỏ được cắt ra từ ảnh lớn.

Ví dụ:

- ảnh gốc rộng hơn 1000 px
- model không train trực tiếp trên cả ảnh
- model cắt từng ô nhỏ cỡ `37 x 37` để học việc so khớp trái/phải

## 3. Các thư mục ảnh là gì và để làm gì

Phần này giải thích chậm hơn một chút, theo kiểu:

- thư mục đó chứa ảnh gì
- từ tiếng Anh đó nghĩa là gì
- người ta dùng nó để làm gì trong bài toán stereo

### Hiểu bài toán trước bằng ví dụ rất đời thường

Hãy tưởng tượng bạn nhìn một cái hộp bằng:

- mắt trái
- mắt phải

Bạn sẽ thấy:

- vị trí cái hộp trong mắt trái hơi khác vị trí trong mắt phải

Độ lệch này chính là thứ bài toán stereo quan tâm.

Mục tiêu của stereo matching là:

- nhìn 2 ảnh trái và phải
- tìm xem mỗi điểm bên ảnh trái nằm ở đâu bên ảnh phải
- từ độ lệch đó suy ra vật gần hay xa

Nói dễ hiểu hơn:

- lệch nhiều thường là vật gần
- lệch ít thường là vật xa

### `image_0` và `image_1` là gì?

Đây là 2 ảnh gốc để model học so khớp.

- `image_0`: ảnh chụp từ camera trái
- `image_1`: ảnh chụp từ camera phải

Trong repo này, với `kitti2012`:

- model train mặc định bằng ảnh xám từ `image_0` và `image_1`
- nên số channel đầu vào là `1`

`channel = 1` nghĩa là:

- mỗi pixel chỉ có 1 giá trị độ sáng
- không phải 3 giá trị màu như ảnh RGB

Ví dụ dễ hiểu:

- ảnh màu thường có 3 kênh: đỏ, lục, lam
- ảnh xám chỉ có 1 kênh: sáng hay tối

### `colored_0` và `colored_1` là gì?

Đây cũng là cùng cảnh đó, nhưng là bản màu:

- `colored_0`: ảnh trái màu
- `colored_1`: ảnh phải màu

Trong repo hiện tại:

- chúng giúp người đọc nhìn dữ liệu dễ hơn
- nhưng pipeline train mặc định không dùng chúng làm đầu vào

Nói đơn giản:

- `image_*` là ảnh code đang dùng để train
- `colored_*` là ảnh để con người nhìn cho trực quan hơn

### `disparity` là gì?

Đây là từ rất hay gặp trong stereo.

`Disparity` nghĩa là:

- độ lệch ngang giữa cùng một điểm trong ảnh trái và ảnh phải

Ví dụ rất đơn giản:

- trong ảnh trái, mép chiếc xe ở cột `x = 500`
- trong ảnh phải, đúng mép chiếc xe đó ở cột `x = 492`
- vậy disparity gần đúng là `500 - 492 = 8` pixel

Ý nghĩa:

- disparity lớn hơn thường là vật gần hơn
- disparity nhỏ hơn thường là vật xa hơn

Nên ảnh `disparity` có thể hiểu là:

- một ảnh mà mỗi pixel ghi lại “độ lệch ngang” của điểm đó

Nó không phải ảnh bình thường để ngắm.

Nó là:

- ảnh đáp án
- để model học xem độ lệch đúng là bao nhiêu

### `ground truth` là gì?

`Ground truth` nghĩa là:

- đáp án đúng có sẵn trong dữ liệu

Trong repo này:

- `disp_noc` và `disp_occ` là ảnh chứa đáp án đúng về disparity

Model train cần ground truth để biết:

- dự đoán của mình đúng hay sai

### `occluded` và `non-occluded` là gì?

Đây cũng là cặp từ rất đặc trưng của stereo.

`Occluded` nghĩa là:

- bị che khuất

`Non-occluded` nghĩa là:

- không bị che khuất

Ví dụ dễ hiểu:

Bạn đứng nhìn một cái cột điện.

- mắt trái có thể vẫn thấy mép sau của cái cột
- nhưng mắt phải có thể không thấy mép đó vì bị thân cột che mất

Khi đó:

- điểm đó là `occluded`

Trong bài toán stereo, các điểm bị che khuất thường khó ghép đúng hơn vì:

- một bên có nhìn thấy
- bên còn lại không có điểm tương ứng rõ ràng

Nên:

- `disp_noc` là disparity ground truth ở vùng không bị che khuất
- `disp_occ` là disparity ground truth có cả vùng bị che khuất

Repo này mặc định dùng `disp_noc` để train vì:

- dữ liệu sạch hơn
- dễ học hơn
- ít gây nhiễu hơn

### `calibration` là gì?

`Calibration` là thông tin hiệu chỉnh camera.

Hiểu đơn giản:

- camera ngoài đời không phải lúc nào cũng hoàn hảo
- cần biết tiêu cự, vị trí tương đối 2 camera, tâm ảnh, các tham số hình học khác

File `calib` chứa các thông tin đó.

Ví dụ đời thường:

- nếu bạn muốn đo khoảng cách thật ngoài đời từ 2 ảnh
- bạn không chỉ cần biết ảnh lệch bao nhiêu
- bạn còn cần biết 2 camera đặt cách nhau bao xa, tiêu cự là bao nhiêu

Những thông tin kiểu đó nằm trong calibration.

Trong repo này:

- file `calib` có trong data
- nhưng pipeline train patch hiện tại không dùng trực tiếp

Nói đơn giản:

- `calib` quan trọng cho bài toán hình học camera
- nhưng code hiện tại đang tập trung học so khớp patch bằng CNN, nên chưa dùng nó trong train mặc định

### Trong `data/training`

- `image_0`: ảnh trái dạng xám
- `image_1`: ảnh phải dạng xám
- `colored_0`: ảnh trái dạng màu
- `colored_1`: ảnh phải dạng màu
- `disp_noc`: disparity đúng ở vùng không bị che khuất
- `disp_occ`: disparity đúng gồm cả vùng bị che khuất
- `calib`: thông tin camera/calibration

Ý nghĩa thực tế:

- code train mặc định đang dùng `image_0`, `image_1`, `disp_noc`
- `colored_0`, `colored_1` có trong data nhưng không phải đầu vào train mặc định của repo này

Ví dụ dễ hiểu một sample train:

- lấy `image_0/000000_10.png` làm ảnh trái
- lấy `image_1/000000_10.png` làm ảnh phải
- lấy `disp_noc/000000_10.png` làm đáp án đúng

Rồi code sẽ làm việc kiểu:

1. chọn một điểm trên ảnh trái
2. nhìn vào ảnh disparity để biết điểm đó lệch bao nhiêu pixel
3. cắt patch trái và vùng tìm kiếm bên phải
4. bắt model học vị trí khớp đúng

Nói ngắn gọn:

- `image_0`, `image_1` là đề bài
- `disp_noc` là đáp án

### Trong `data/testing`

- `image_0`: ảnh trái xám
- `image_1`: ảnh phải xám
- `colored_0`: ảnh trái màu
- `colored_1`: ảnh phải màu
- `calib`: calibration

Điểm rất quan trọng:

- `testing` không có `disp_noc`
- nghĩa là không có đáp án đúng disparity để train
- vì vậy `testing` được dùng để infer, không dùng để train

## 4. Vì sao có 388 ảnh training nhưng chỉ train trên 194 sample?

Đây là chỗ dễ gây nhầm nhất.

Trong repo hiện tại:

- `data/training/image_0` có `388` file
- `data/training/image_1` có `388` file
- `data/training/disp_noc` có `194` file

Lý do:

- mỗi cảnh có 2 frame là `*_10` và `*_11`
- ví dụ `000000_10.png` và `000000_11.png`
- nhưng disparity ground truth để train chỉ có cho `*_10`

Ví dụ:

```text
image_0/000000_10.png
image_0/000000_11.png
image_1/000000_10.png
image_1/000000_11.png
disp_noc/000000_10.png
```

Bạn thấy:

- có 4 file ảnh trái/phải cho 2 frame
- nhưng chỉ có 1 file disparity để làm đáp án train

Nên khi code build dữ liệu train, nó quét từ:

- `training/disp_noc/*.png`

chứ không quét từ toàn bộ `training/image_0/*.png`.

Vì vậy số sample train được là:

- `194` sample

chứ không phải `388`.

## 5. Ví dụ dễ hiểu với 1 sample cụ thể

Lấy sample `000000_10` làm ví dụ.

Các file liên quan:

- `data/training/image_0/000000_10.png`: ảnh trái xám
- `data/training/image_1/000000_10.png`: ảnh phải xám
- `data/training/colored_0/000000_10.png`: ảnh trái màu
- `data/training/colored_1/000000_10.png`: ảnh phải màu
- `data/training/disp_noc/000000_10.png`: đáp án disparity đúng
- `data/training/calib/000000.txt`: thông số camera

Ý nghĩa từng file:

- `image_0`: ảnh model nhìn từ camera trái
- `image_1`: ảnh model nhìn từ camera phải
- `disp_noc`: đáp án đúng để model học
- `colored_*`: ảnh màu để quan sát cho dễ, nhưng train mặc định không dùng
- `calib`: thông tin camera, repo hiện tại không dùng nó trong train patch mặc định

Bạn có thể xem ảnh ghép minh hoạ:

- [sample_000000_10.png](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/figures/data_samples/sample_000000_10.png)
- [sample_000001_10.png](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/figures/data_samples/sample_000001_10.png)
- [sample_000002_10.png](/home/quanghien/aivn/stereo/dl_stereo_matching_pytorch/figures/data_samples/sample_000002_10.png)

## 6. Code train thật sự lấy dữ liệu từ đâu?

Nếu repo không có bộ file preprocess cũ như `myPerm.bin`, code sẽ đi theo nhánh KITTI raw.

Khi đó code:

1. vào `data/training/disp_noc`
2. lấy danh sách các file `*_10.png`
3. coi mỗi file disparity là một sample có thể train
4. đọc ảnh trái/phải tương ứng từ `image_0` và `image_1`
5. từ disparity đúng, cắt ra nhiều patch nhỏ để train

Nói đơn giản:

- code không train theo kiểu “mỗi ảnh lớn là một batch”
- code train theo kiểu “từ mỗi ảnh lớn, cắt ra rất nhiều ô nhỏ”

Nên:

- `194` là số cảnh có thể dùng để train
- còn số patch train thực tế thì lớn hơn rất nhiều

## 7. Patch train là gì, vì sao phải cắt patch?

Repo này làm stereo matching theo kiểu patch-based.

Nghĩa là:

- không đưa cả ảnh lớn vào để train trực tiếp
- mà cắt một ô nhỏ bên trái
- rồi lấy một dải ô bên phải
- model học xem ô trái khớp với vị trí nào bên phải

Ví dụ dễ hiểu:

1. chọn một điểm ở ảnh trái
2. cắt một ô `37 x 37` quanh điểm đó
3. sang ảnh phải, lấy một vùng rộng hơn ở cùng hàng
4. model đo xem vị trí nào trong vùng phải là khớp nhất

Làm như vậy vì:

- bài toán stereo chủ yếu là tìm độ lệch ngang
- patch nhỏ giúp model học trực tiếp việc so khớp
- tiết kiệm hơn so với train full ảnh theo cách của paper này

## 8. `image_0`, `image_1`, `colored_0`, `colored_1` khác nhau thế nào?

### `image_0` và `image_1`

Đây là ảnh xám:

- `image_0`: camera trái
- `image_1`: camera phải

Repo đang train mặc định bằng 2 thư mục này.

### `colored_0` và `colored_1`

Đây là bản màu của cùng cảnh:

- `colored_0`: trái màu
- `colored_1`: phải màu

Trong repo hiện tại:

- chúng chủ yếu hữu ích để nhìn dữ liệu dễ hơn
- chưa phải đầu vào train mặc định của `kitti2012`

## 9. `disp_noc` và `disp_occ` khác nhau thế nào?

### `disp_noc`

`noc` là `non-occluded`, hiểu đơn giản là:

- các vùng nhìn thấy rõ ở cả 2 camera
- đây là ground truth “sạch” hơn

Repo hiện tại dùng `disp_noc` để train.

### `disp_occ`

`occ` là `occluded`, tức là:

- có cả các vùng bị che khuất

Repo này hiện không dùng `disp_occ` trong pipeline train mặc định.

## 10. Train, validate, infer khác nhau thế nào theo góc nhìn dữ liệu?

### Train

Train dùng:

- ảnh từ `training/image_0`
- ảnh từ `training/image_1`
- disparity đúng từ `training/disp_noc`

Mục đích:

- cho model học đáp án đúng

### Validate

Validate cũng lấy từ `training`, không lấy từ `testing`.

Chỉ khác là:

- code tách một phần sample trong `training` ra để đo chất lượng

Mặc định trong lệnh train:

- `160` sample để train
- `34` sample để validate

Tổng cộng:

- `160 + 34 = 194`

### Infer

Infer thường chạy trên:

- `testing/image_0`
- `testing/image_1`

Mục đích:

- model tự dự đoán disparity
- không có đáp án đúng kèm theo trong pipeline này

## 11. Một số từ kỹ thuật trong code, nói đơn giản

### `normalize_image`

Chuẩn hoá ảnh trước khi đưa vào model.

Hiểu đơn giản:

- ảnh được đổi về thang dễ học hơn
- giúp model train ổn định hơn

### `cache`

`Cache` nghĩa là:

- đọc ảnh lên bộ nhớ trước
- để khi train không phải mở file lại quá nhiều lần

### `disp_range`

Là khoảng disparity tối đa model sẽ đi tìm.

Hiểu đơn giản:

- model không tìm vô hạn
- nó chỉ tìm trong một khoảng ngang nhất định

### `patch_size`

Là kích thước ô ảnh nhỏ cắt ra để train.

Ví dụ:

- `patch_size=37` nghĩa là patch trái cỡ `37 x 37`

## 12. Kết luận thật ngắn

Nếu chỉ cần nhớ 4 ý:

1. Repo train bằng `training`, không train bằng `testing`.
2. Số `194` đến từ `training/disp_noc`, vì chỉ có `194` file disparity ground truth.
3. `388` ảnh trong `image_0/image_1` là do có cả frame `*_10` và `*_11`.
4. Code train theo patch nhỏ cắt ra từ ảnh lớn, không train trực tiếp trên toàn ảnh lớn.
