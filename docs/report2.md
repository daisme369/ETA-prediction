# Report 2: Comparison of New ETA Correction Formulas

## 1. Mục tiêu

Mục tiêu của phần này là đánh giá các công thức correction mới cho bài toán ETA, dựa trên API ETA ban đầu:

```text
api_eta_secs = estimate_time
actual_eta_secs = delta_time
```

Các phương pháp cũ chủ yếu dùng công thức additive residual:

```text
residual = actual_eta_secs - api_eta_secs
corrected_eta = estimate_time + predicted_residual
```

Trong lần thử mới, bổ sung ba family công thức:

```text
1. Multiplicative / ratio:
   corrected_eta = estimate_time * ratio

2. Affine:
   corrected_eta = a * estimate_time + b

3. Log-ratio:
   corrected_eta = estimate_time * exp(log_residual)
```

Dữ liệu vẫn được chia theo thời gian:

```text
Train: 215 rows
Validation: 46 rows
Test: 47 rows
Split: chronological 70% / 15% / 15%
```

Các metrics chính:

```text
MAE: lỗi tuyệt đối trung bình, dùng để đánh giá sai số trung bình.
RMSE: nhạy với lỗi lớn.
MAPE: lỗi phần trăm, nhạy với các chuyến ngắn.
P95: tail error, đo nhóm 5% case lỗi lớn nhất.
```

---

## 2. Giải thích các công thức residual mới

### 2.1 Multiplicative / Ratio Correction

Công thức:

```text
ratio = actual_eta_secs / estimate_time
corrected_eta = estimate_time * ratio
```

Ý nghĩa:

Nếu API thường underestimate theo tỷ lệ, ví dụ thực tế thường cao hơn ETA khoảng 10%, thì ratio correction phù hợp hơn additive correction. Thay vì cộng cố định `+15s`, phương pháp này scale ETA theo độ dài chuyến đi.

Ví dụ:

```text
estimate_time = 100s, ratio = 1.10 -> corrected_eta = 110s
estimate_time = 300s, ratio = 1.10 -> corrected_eta = 330s
```

Điểm khác biệt so với additive:

```text
additive: cộng cùng một offset theo giây
ratio: nhân theo tỷ lệ, correction lớn hơn với ETA dài hơn
```

Các biến thể đã thử:

```text
ratio_global:
corrected_eta = estimate_time * median(train.actual_eta_secs / train.estimate_time)

ratio_time_bin:
corrected_eta = estimate_time * median(train.actual_eta_secs / train.estimate_time | time_bin)

ratio_smoothed_time_bin:
corrected_eta = estimate_time * smoothed_ratio_by_time_bin
```

Global ratio học được:

```text
global_ratio = 1.1009
```

Nghĩa là ở mức median trên train set, actual time cao hơn API ETA khoảng `10.09%`.

---

### 2.2 Affine Correction

Công thức:

```text
corrected_eta = a * estimate_time + b
```

Ý nghĩa:

Affine correction tổng quát hơn cả additive và ratio:

```text
Nếu a = 1 và b != 0 -> gần với additive correction
Nếu a != 1 và b = 0 -> gần với ratio correction
Nếu a != 1 và b != 0 -> vừa scale vừa dịch offset
```

Do dữ liệu residual có outlier và distribution lệch phải, affine được fit bằng Huber Regression thay vì ordinary least squares:

```text
estimator = sklearn.linear_model.HuberRegressor
epsilon   = 1.35
alpha     = 0.0001
max_iter  = 1000
```

Huber Regression phù hợp hơn OLS vì:

* Với lỗi nhỏ, nó vẫn tối ưu gần giống squared loss.
* Với lỗi lớn/outlier, nó giảm ảnh hưởng của điểm bất thường.
* Phù hợp khi actual travel time có một số case bất thường do traffic, dừng đón khách, đèn đỏ hoặc incident.

Global affine học được:

```text
a = 0.1595
b = 150.7553
```

Hệ số này cho thấy model đang kéo prediction về một intercept khá lớn. Đây là dấu hiệu affine có thể đang fit theo median/central tendency của data nhiều hơn là giữ được scale tự nhiên của `estimate_time`.

Các biến thể đã thử:

```text
affine_global:
corrected_eta = a_global * estimate_time + b_global

affine_time_bin:
corrected_eta = a_time_bin * estimate_time + b_time_bin

affine_smoothed_time_bin:
corrected_eta =
    (w * a_time_bin + (1 - w) * a_global) * estimate_time
  + (w * b_time_bin + (1 - w) * b_global)
```

---

### 2.3 Log-ratio Correction

Công thức:

```text
log_residual = log(actual_eta_secs / estimate_time)
corrected_eta = estimate_time * exp(log_residual)
```

Ý nghĩa:

Log-ratio vẫn là correction dạng tỷ lệ, nhưng học trong không gian log. Cách này thường hữu ích khi ratio bị lệch phải hoặc có outlier lớn, vì log transform nén các giá trị ratio quá lớn.

Ví dụ:

```text
actual / estimate = 1.10 -> log_residual = log(1.10)
corrected_eta = estimate_time * exp(log_residual)
```

Trong setup hiện tại, vì ta dùng median và ratio luôn dương, `log_ratio_global` gần như trùng với `ratio_global`. Time-bin log-ratio cũng gần như trùng với time-bin ratio.

Các biến thể đã thử:

```text
log_ratio_global
log_ratio_time_bin
log_ratio_smoothed_time_bin
```

---

## 3. Kết quả các phương pháp dùng công thức mới

Kết quả dưới đây là trên test set gồm 47 samples.

```text
method                         MAE      RMSE     MAPE     P50      P90      P95      R2
api_eta                        37.956   56.887   18.34%   23.079   84.593   123.750  0.034
ratio_global                   37.237   52.529   19.60%   30.309   68.764   108.039  0.176
ratio_time_bin                 38.018   51.202   20.69%   32.327   66.063   97.279   0.217
ratio_smoothed_time_bin        38.018   51.202   20.69%   32.327   66.063   97.279   0.217
affine_global                  39.812   56.781   21.31%   32.574   74.337   111.769  0.038
affine_time_bin                41.758   56.609   23.00%   31.129   77.325   100.859  0.043
affine_smoothed_time_bin       40.403   56.443   21.97%   34.651   75.111   107.764  0.049
log_ratio_global               37.237   52.529   19.60%   30.309   68.764   108.039  0.176
log_ratio_time_bin             38.018   51.202   20.69%   32.327   66.063   97.279   0.217
log_ratio_smoothed_time_bin    38.018   51.202   20.69%   32.327   66.063   97.279   0.217
```

So với API baseline:

```text
method                         MAE improvement   P95 improvement
ratio_global                   +1.89%            +12.70%
ratio_time_bin                 -0.16%            +21.39%
ratio_smoothed_time_bin        -0.16%            +21.39%
affine_global                  -4.89%            +9.68%
affine_time_bin                -10.02%           +18.50%
affine_smoothed_time_bin       -6.45%            +12.92%
log_ratio_global               +1.89%            +12.70%
log_ratio_time_bin             -0.16%            +21.39%
log_ratio_smoothed_time_bin    -0.16%            +21.39%
```

Nhận xét nhanh:

* `ratio_global` và `log_ratio_global` có MAE tốt nhất trong toàn bộ nhóm mới: `37.237s`.
* `ratio_time_bin` và `log_ratio_time_bin` có RMSE/P95 tốt nhất: `RMSE = 51.202s`, `P95 = 97.279s`.
* `affine` cải thiện P95 so với API nhưng làm MAE tệ hơn đáng kể.
* `ratio_smoothed_time_bin` và `log_ratio_smoothed_time_bin` trùng với raw time-bin vì validation chọn smoothing `k = 0`.

---

## 4. So sánh với các phương pháp cũ

Các phương pháp cũ gồm:

```text
api_eta
additive_global
additive_time_bin
additive_smoothed_time_bin
MLP residual ETA
Hour-bin MLP residual ETA
DeeprETA-like
XGBoost-based methods
```

### 4.1 So sánh với additive residual baselines

```text
method                         MAE      RMSE     MAPE     P95      MAE vs API   P95 vs API
api_eta                        37.956   56.887   18.34%   123.750  0.00%        0.00%
additive_global                37.249   52.804   19.58%   107.954  +1.86%       +12.76%
additive_time_bin              37.965   51.794   20.45%   98.936   -0.02%       +20.05%
additive_smoothed_time_bin     38.073   52.159   20.40%   103.111  -0.31%       +16.68%
ratio_global                   37.237   52.529   19.60%   108.039  +1.89%       +12.70%
ratio_time_bin                 38.018   51.202   20.69%   97.279   -0.16%       +21.39%
log_ratio_global               37.237   52.529   19.60%   108.039  +1.89%       +12.70%
log_ratio_time_bin             38.018   51.202   20.69%   97.279   -0.16%       +21.39%
affine_global                  39.812   56.781   21.31%   111.769  -4.89%       +9.68%
affine_time_bin                41.758   56.609   23.00%   100.859  -10.02%      +18.50%
affine_smoothed_time_bin       40.403   56.443   21.97%   107.764  -6.45%       +12.92%
```

Kết luận so với additive:

* `ratio_global` mạnh hơn `additive_global` theo MAE, RMSE và R2, nhưng mức chênh rất nhỏ.
* `additive_global` có P95 nhỉnh hơn `ratio_global` một chút: `107.954s` so với `108.039s`.
* `ratio_time_bin` mạnh hơn `additive_time_bin` theo RMSE và P95.
* `additive_time_bin` nhỉnh hơn `ratio_time_bin` theo MAE, nhưng gần như ngang API baseline.
* `affine` không mạnh hơn additive trên test set hiện tại.

Nói ngắn gọn:

```text
Global correction:
ratio/log-ratio hơi mạnh hơn additive về MAE.

Time-bin correction:
ratio/log-ratio mạnh hơn additive về RMSE và P95.

Affine correction:
không mạnh hơn additive, dù đã dùng Huber Regression.
```

---

### 4.2 So sánh với các model ML/deep cũ

```text
method                         MAE      RMSE     MAPE     P95
ratio_global                   37.237   52.529   19.60%   108.039
ratio_time_bin                 38.018   51.202   20.69%   97.279
log_ratio_global               37.237   52.529   19.60%   108.039
log_ratio_time_bin             38.018   51.202   20.69%   97.279
MLP residual ETA               37.612   52.661   19.43%   101.684
Hour-bin MLP residual ETA      38.297   54.500   19.41%   107.584
DeeprETA-like                  38.453   52.213   20.95%   99.546
xgb_residual_eta               42.639   n/a      n/a      106.104
hour_bin_xgb_residual_eta      43.383   n/a      n/a      113.415
xgb_direct_eta                 44.914   n/a      n/a      99.304
```

Kết luận so với ML/deep:

* `ratio_global` / `log_ratio_global` tốt hơn MLP residual theo MAE.
* `ratio_time_bin` / `log_ratio_time_bin` tốt hơn MLP residual, Hour-bin MLP, DeeprETA-like và XGB theo P95.
* DeeprETA-like có RMSE khá tốt (`52.213s`) nhưng vẫn kém `ratio_time_bin` (`51.202s`) và MAE kém hơn.
* Các XGBoost method kém rõ theo MAE; chỉ có `xgb_direct_eta` có P95 tương đối gần nhóm tốt nhất nhưng MAE quá cao.

Do đó, các công thức mới dạng ratio/log-ratio đang mạnh hơn các model phức tạp trong điều kiện dữ liệu hiện tại. Lý do chính là dataset nhỏ, feature hạn chế, và route tương đối cố định, nên correction thống kê đơn giản có bias-variance tradeoff tốt hơn.

---

## 5. So sánh các phương pháp mới với nhau

### 5.1 Ratio correction

Điểm mạnh:

* `ratio_global` có MAE tốt nhất: `37.237s`.
* `ratio_time_bin` có P95 tốt nhất: `97.279s`.
* Công thức đơn giản, dễ triển khai.
* Phù hợp khi API ETA sai theo tỷ lệ, không chỉ sai theo offset cố định.
* Time-bin ratio học được bias khác nhau theo khung giờ.

Điểm yếu:

* `ratio_time_bin` làm MAE kém hơn API baseline một chút: `38.018s` so với `37.956s`.
* MAPE cao hơn API baseline vì correction làm tăng lỗi tương đối trên một số chuyến ngắn.
* Time-bin ratio có thể over-correct nếu một bin có ít sample, đặc biệt late evening.

Giải thích điểm yếu:

Ratio correction scale theo ETA. Với chuyến ngắn, chỉ cần correction sai một lượng nhỏ theo giây thì MAPE đã tăng mạnh. Ngoài ra, nếu một time-bin có ít dữ liệu, median ratio của bin đó dễ bị ảnh hưởng bởi vài case bất thường.

---

### 5.2 Log-ratio correction

Điểm mạnh:

* Kết quả gần như giống ratio correction.
* Có cùng MAE tốt nhất ở global: `37.237s`.
* Có cùng P95 tốt nhất ở time-bin: `97.279s`.
* Về mặt lý thuyết robust hơn khi ratio lệch phải vì log transform nén outlier.

Điểm yếu:

* Trong setup hiện tại, không tạo ra lợi ích thực nghiệm rõ so với ratio.
* Vì đang dùng median, log-ratio và ratio gần như tương đương.
* Phức tạp hơn ratio một chút khi giải thích cho production.

Giải thích điểm yếu:

Với dữ liệu positive ratio và estimator là median, `median(log(ratio))` sau khi `exp` gần như bằng `median(ratio)`. Vì vậy log-ratio không khác biệt nhiều. Lợi ích của log-ratio có thể chỉ rõ hơn nếu sau này dùng model học `log_residual` hoặc dùng mean/regularized regression trong log-space.

---

### 5.3 Affine correction with Huber Regression

Điểm mạnh:

* Công thức tổng quát nhất: vừa scale vừa cộng offset.
* Dùng Huber Regression nên hợp lý hơn OLS trong bối cảnh có outlier và distribution lệch phải.
* Vẫn cải thiện P95 so với API baseline:
  * `affine_global`: P95 improvement `+9.68%`
  * `affine_time_bin`: P95 improvement `+18.50%`
  * `affine_smoothed_time_bin`: P95 improvement `+12.92%`

Điểm yếu:

* Test MAE kém API baseline:
  * `affine_global`: `39.812s`
  * `affine_time_bin`: `41.758s`
  * `affine_smoothed_time_bin`: `40.403s`
* Time-bin affine có dấu hiệu overfit/mismatch giữa validation và test.
* Global affine học được `a = 0.1595`, `b = 150.7553`, khiến prediction bị kéo mạnh về intercept.

Giải thích điểm yếu:

Affine Huber cố fit quan hệ tuyến tính giữa `estimate_time` và `actual_time`. Nhưng với dataset nhỏ, route cố định, nhiều noise không quan sát được, và test set chỉ có 47 samples, hệ số affine có thể không generalize tốt. Intercept lớn làm model tốt hơn ở một số case tail-error nhưng làm sai trung bình tăng lên ở nhiều case bình thường.

---

## 6. Kết luận: công thức và phương pháp nên chọn

### 6.1 Nếu mục tiêu chính là MAE

Chọn:

```text
ratio_global hoặc log_ratio_global
```

Lý do:

```text
MAE = 37.237s
RMSE = 52.529s
P95 = 108.039s
MAE improvement vs API = +1.89%
P95 improvement vs API = +12.70%
```

So với additive global:

```text
ratio_global MAE    = 37.237s
additive_global MAE = 37.249s
```

Chênh lệch rất nhỏ, nhưng ratio/log-ratio vẫn là tốt nhất theo MAE.

---

### 6.2 Nếu mục tiêu chính là giảm lỗi lớn / tail-error

Chọn:

```text
ratio_time_bin hoặc log_ratio_time_bin
```

Lý do:

```text
RMSE = 51.202s
P95  = 97.279s
P95 improvement vs API = +21.39%
```

Đây là nhóm tốt nhất theo RMSE và P95 trong toàn bộ các phương pháp đã thử.

Trade-off:

```text
MAE = 38.018s
MAE improvement vs API = -0.16%
```

Nghĩa là phương pháp này giảm lỗi lớn tốt nhất, nhưng đánh đổi một chút MAE trung bình.

---

### 6.3 Recommendation cuối cùng

Primary method nên chọn:

```text
API + Time-bin Ratio Correction
```

Công thức:

```text
corrected_eta = estimate_time * median_ratio_by_time_bin
```

Lý do chọn:

* Có P95 tốt nhất: `97.279s`.
* Có RMSE tốt nhất: `51.202s`.
* Cải thiện P95 mạnh nhất so với API baseline: `+21.39%`.
* Công thức đơn giản, dễ triển khai, không cần model phức tạp.
* Phù hợp với quan sát rằng API bias thay đổi theo khung giờ và có tính scale theo ETA.

Secondary method nên giữ:

```text
API + Global Ratio Correction
```

Lý do:

* Có MAE tốt nhất: `37.237s`.
* Dễ giải thích và ổn định hơn time-bin khi dữ liệu mỗi bin còn ít.
* Là fallback tốt nếu muốn ưu tiên MAE hoặc muốn production method đơn giản hơn.

Không nên chọn làm primary ở giai đoạn hiện tại:

```text
Affine Huber Regression
```

Lý do:

* Dù hợp lý về mặt robust regression, test MAE kém API baseline.
* Hệ số global affine cho thấy model bị kéo mạnh về intercept.
* Cần thêm rolling/expanding time split để xác nhận khả năng generalize trước khi dùng.

Kết luận ngắn:

```text
Best overall for tail-error:
API + Time-bin Ratio Correction

Best overall for average error:
API + Global Ratio Correction / Global Log-ratio Correction

Most production-safe fallback:
API + Global Ratio Correction hoặc API + Global Median Residual
```
