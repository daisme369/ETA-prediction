# Final Report: Bus ETA Correction for Map Application

## Mục lục
- [Final Report: Bus ETA Correction for Map Application](#final-report-bus-eta-correction-for-map-application)
  - [Mục lục](#mục-lục)
  - [1. Problem Context](#1-problem-context)
  - [2. Goals](#2-goals)
  - [3. Dataset](#3-dataset)
  - [4. Evaluation Metrics](#4-evaluation-metrics)
  - [5. Methods, Metrics, and Comments](#5-methods-metrics-and-comments)
    - [5.1 API / Vietmap Baseline](#51-api--vietmap-baseline)
    - [5.2 Additive Residual Correction](#52-additive-residual-correction)
    - [5.3 Ratio / Multiplicative Correction](#53-ratio--multiplicative-correction)
    - [5.4 Log-ratio Correction](#54-log-ratio-correction)
    - [5.5 Affine Correction with Huber Regression](#55-affine-correction-with-huber-regression)
    - [5.6 ML and Deep Learning Methods](#56-ml-and-deep-learning-methods)
  - [6. Overall Comparison](#6-overall-comparison)
  - [7. Final Solution](#7-final-solution)
    - [Primary Solution](#primary-solution)
    - [Secondary Solution](#secondary-solution)
    - [Production Recommendation](#production-recommendation)
  - [8. Limitations and Next Steps](#8-limitations-and-next-steps)

---

## 1. Problem Context

Trong hệ thống map hiện tại, một tính năng quan trọng là **bus ETA**: ước lượng thời gian xe buýt di chuyển hoặc đến điểm đón/điểm đến. Hệ thống đã có sẵn ETA ban đầu từ map/routing API:

```text
api_eta_secs = estimate_time
actual_eta_secs = delta_time
```

`estimate_time` là ETA do routing engine trả về, còn `delta_time` là thời gian thực tế quan sát được. Routing API đã là một baseline mạnh, nhưng trong thực tế bus ETA vẫn có thể sai do:

- Traffic pattern thay đổi theo khung giờ.
- Xe buýt dừng đón/trả khách.
- Đèn đỏ, congestion, incident hoặc thời tiết.
- Bias của routing API trên một route hoặc một số time-bin cụ thể.

Vì API ETA đã có chất lượng tương đối tốt, hướng tiếp cận hợp lý không phải là thay thế hoàn toàn API bằng model mới, mà là **ETA correction / residual correction**:

```text
corrected_eta = api_eta + correction
```

hoặc ở dạng tỷ lệ:

```text
corrected_eta = api_eta * correction_ratio
```

Mục tiêu là tận dụng estimate tốt từ routing API, sau đó học phần sai lệch có tính hệ thống từ dữ liệu lịch sử.

---

## 2. Goals

Mục tiêu sản phẩm và mô hình:

1. Cải thiện độ chính xác của bus ETA hiển thị trên map.
2. Giảm các lỗi lớn, vì ETA sai nặng ảnh hưởng trực tiếp đến trải nghiệm người dùng khi chờ xe buýt.
3. Giữ phương pháp đủ đơn giản để triển khai production, dễ giải thích và dễ monitor.
4. Ưu tiên mô hình có bias-variance tradeoff tốt với dataset hiện tại còn nhỏ.
5. Xác định phương pháp chính và phương pháp fallback cho hệ thống bus ETA.

Với use case bus ETA trên map, **P95/tail-error quan trọng hơn chỉ tối ưu MAE**, vì một số case ETA sai rất lớn có thể gây trải nghiệm tệ hơn nhiều so với nhiều sai số nhỏ.

---

## 3. Dataset

Dữ liệu chính gồm:

```text
estimate_time: ETA từ map/routing API
delta_time: thời gian di chuyển thực tế
hour / time_bin: thông tin thời điểm khởi hành
```

Dữ liệu được chia theo thời gian:

```text
Train: 215 rows
Validation: 46 rows
Test: 47 rows
Split: chronological 70% / 15% / 15%
```

Việc chia theo thời gian phù hợp hơn random split vì bài toán ETA có yếu tố temporal. Cách chia này tránh tình huống model học từ dữ liệu tương lai để dự đoán quá khứ.

Các time-bin được sử dụng trong một số phương pháp:

```text
early_morning: 4-6
morning_peak: 7-9
off_peak_day: 10-14
evening_peak: 15-18
late_evening: 19-21
```

Nhận xét về dữ liệu:

- Dataset hiện tại còn nhỏ, đặc biệt test set chỉ có 47 samples.
- Một số time-bin có thể sparse, ví dụ late evening.
- Feature hiện tại còn hạn chế, chủ yếu xoay quanh ETA API và thông tin thời gian.
- Vì route tương đối cố định, các correction thống kê đơn giản có lợi thế lớn so với model phức tạp.

---

## 4. Evaluation Metrics

Các metrics được sử dụng:

```text
MAE: lỗi tuyệt đối trung bình, phản ánh sai số trung bình.
RMSE: nhạy với lỗi lớn, phạt mạnh các case sai nặng.
MAPE: lỗi phần trăm, nhạy với các chuyến ngắn.
P50: median absolute error.
P95: lỗi ở nhóm 5% case tệ nhất, đại diện cho tail-error.
R2: mức giải thích variance so với baseline trung bình.
```

Trong bus ETA, cách đọc metric nên ưu tiên:

- **MAE** để đánh giá độ chính xác trung bình.
- **P95 và RMSE** để đánh giá khả năng giảm lỗi lớn.
- **MAPE** để kiểm tra ảnh hưởng trên các chuyến ngắn.

---

## 5. Methods, Metrics, and Comments

### 5.1 API / Vietmap Baseline

Công thức:

```text
prediction = estimate_time
```

Kết quả test set:

```text
MAE  = 37.956s
RMSE = 56.887s
MAPE = 18.34%
P50  = 23.079s
P95  = 123.750s
R2   = 0.034
```

Nhận xét:

- Đây là baseline thực tế, có sẵn từ routing engine.
- MAPE tốt nhất trong các phương pháp hiện tại.
- Không cần training, dễ triển khai.
- Điểm yếu chính là P95 cao, nghĩa là vẫn có các case ETA sai lớn.
- API có xu hướng underestimate trong một số ngữ cảnh, nên correction vẫn có giá trị.

---

### 5.2 Additive Residual Correction

Công thức chung:

```text
residual = actual_eta_secs - api_eta_secs
corrected_eta = estimate_time + predicted_residual
```

Các biến thể:

```text
additive_global:
corrected_eta = estimate_time + median(train.residual)

additive_time_bin:
corrected_eta = estimate_time + median(train.residual | time_bin)

additive_smoothed_time_bin:
corrected_eta = estimate_time + smoothed_residual_by_time_bin
```

Kết quả:

```text
method                         MAE      RMSE     MAPE     P95      MAE vs API   P95 vs API
api_eta                        37.956   56.887   18.34%   123.750  0.00%        0.00%
additive_global                37.249   52.804   19.58%   107.954  +1.86%       +12.76%
additive_time_bin              37.965   51.794   20.45%   98.936   -0.02%       +20.05%
additive_smoothed_time_bin     38.073   52.159   20.40%   103.111  -0.31%       +16.68%
```

Nhận xét:

- `additive_global` là baseline correction rất mạnh: đơn giản, ổn định, MAE tốt.
- `additive_time_bin` không cải thiện MAE nhưng giảm P95 mạnh, từ `123.750s` xuống `98.936s`.
- `additive_smoothed_time_bin` robust hơn về mặt phương pháp, vì giảm rủi ro overfit ở bin ít sample.
- MAPE của nhóm additive tệ hơn API baseline, vì cộng offset cố định có thể làm tăng lỗi tương đối trên chuyến ngắn.

Kết luận:

- Nếu cần fallback cực kỳ dễ giải thích, `additive_global` là lựa chọn tốt.
- Nếu ưu tiên giảm tail-error, additive time-bin đã cho thấy hướng đi đúng: correction nên phụ thuộc vào khung giờ.

---

### 5.3 Ratio / Multiplicative Correction

Công thức:

```text
ratio = actual_eta_secs / estimate_time
corrected_eta = estimate_time * ratio
```

Các biến thể:

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

Điều này nghĩa là ở mức median trên train set, actual bus travel time cao hơn API ETA khoảng `10.09%`.

Kết quả:

```text
method                         MAE      RMSE     MAPE     P50      P90      P95      R2
ratio_global                   37.237   52.529   19.60%   30.309   68.764   108.039  0.176
ratio_time_bin                 38.018   51.202   20.69%   32.327   66.063   97.279   0.217
ratio_smoothed_time_bin        38.018   51.202   20.69%   32.327   66.063   97.279   0.217
```

So với API baseline:

```text
method                         MAE improvement   P95 improvement
ratio_global                   +1.89%            +12.70%
ratio_time_bin                 -0.16%            +21.39%
ratio_smoothed_time_bin        -0.16%            +21.39%
```

Nhận xét:

- `ratio_global` có MAE tốt nhất trong toàn bộ các phương pháp đã thử: `37.237s`.
- `ratio_time_bin` có RMSE và P95 tốt nhất: `RMSE = 51.202s`, `P95 = 97.279s`.
- Ratio correction phù hợp khi API sai theo tỷ lệ, không chỉ sai theo offset cộng thêm.
- `ratio_time_bin` đánh đổi MAE rất nhỏ để giảm lỗi lớn mạnh nhất.
- Với bus ETA, trade-off này hợp lý nếu sản phẩm ưu tiên tránh các case ETA sai nặng.

---

### 5.4 Log-ratio Correction

Công thức:

```text
log_residual = log(actual_eta_secs / estimate_time)
corrected_eta = estimate_time * exp(log_residual)
```

Kết quả:

```text
method                         MAE      RMSE     MAPE     P50      P90      P95      R2
log_ratio_global               37.237   52.529   19.60%   30.309   68.764   108.039  0.176
log_ratio_time_bin             38.018   51.202   20.69%   32.327   66.063   97.279   0.217
log_ratio_smoothed_time_bin    38.018   51.202   20.69%   32.327   66.063   97.279   0.217
```

Nhận xét:

- Kết quả gần như trùng với ratio correction.
- Về lý thuyết, log-ratio robust hơn khi ratio lệch phải hoặc có outlier lớn.
- Trong setup hiện tại, vì dùng median và ratio luôn dương, `exp(median(log(ratio)))` gần như tương đương `median(ratio)`.
- Log-ratio có thể hữu ích hơn nếu sau này dùng regression/model học trong log-space.

Kết luận:

- `log_ratio_global` có thể xem là tương đương `ratio_global`.
- `log_ratio_time_bin` có thể xem là tương đương `ratio_time_bin`.
- Nếu ưu tiên giải thích đơn giản cho production, ratio thường dễ truyền đạt hơn log-ratio.

---

### 5.5 Affine Correction with Huber Regression

Công thức:

```text
corrected_eta = a * estimate_time + b
```

Affine tổng quát hơn additive và ratio:

```text
Nếu a = 1 và b != 0 -> gần với additive correction
Nếu a != 1 và b = 0 -> gần với ratio correction
Nếu a != 1 và b != 0 -> vừa scale vừa dịch offset
```

Do residual có outlier và distribution lệch phải, affine được fit bằng Huber Regression:

```text
estimator = sklearn.linear_model.HuberRegressor
epsilon   = 1.35
alpha     = 0.0001
max_iter  = 1000
```

Global affine học được:

```text
a = 0.1595
b = 150.7553
```

Kết quả:

```text
method                         MAE      RMSE     MAPE     P95      MAE vs API   P95 vs API
affine_global                  39.812   56.781   21.31%   111.769  -4.89%       +9.68%
affine_time_bin                41.758   56.609   23.00%   100.859  -10.02%      +18.50%
affine_smoothed_time_bin       40.403   56.443   21.97%   107.764  -6.45%       +12.92%
```

Nhận xét:

- Affine vẫn cải thiện P95 so với API baseline.
- Tuy nhiên MAE tệ hơn API baseline khá rõ.
- Global affine có intercept lớn, khiến prediction bị kéo mạnh về một mức trung tâm.
- Với dataset nhỏ và test set chỉ 47 samples, affine có dấu hiệu nhạy với split hoặc mismatch giữa validation và test.

Kết luận:

- Chưa nên chọn affine làm primary method.
- Có thể giữ affine như diagnostic method để kiểm tra quan hệ tuyến tính giữa API ETA và actual ETA.

---

### 5.6 ML and Deep Learning Methods

Các phương pháp đã thử:

```text
MLP residual ETA
Hour-bin MLP residual ETA
DeeprETA-like
XGBoost residual/direct ETA
```

Kết quả:

```text
method                         MAE      RMSE     MAPE     P95
MLP residual ETA               37.612   52.661   19.43%   101.684
Hour-bin MLP residual ETA      38.297   54.500   19.41%   107.584
DeeprETA-like                  38.453   52.213   20.95%   99.546
xgb_residual_eta               42.639   n/a      n/a      106.104
hour_bin_xgb_residual_eta      43.383   n/a      n/a      113.415
xgb_direct_eta                 44.914   n/a      n/a      99.304
```

Nhận xét:

- `MLP residual ETA` là ML residual model tốt nhất hiện tại, nhưng vẫn không thắng `ratio_global` về MAE và không thắng `ratio_time_bin` về P95.
- `DeeprETA-like` có P95 tương đối tốt, nhưng MAE và MAPE kém hơn các correction đơn giản.
- XGBoost kém rõ theo MAE; một số biến thể có P95 khá nhưng đánh đổi quá lớn ở sai số trung bình.
- Với dataset nhỏ và feature hạn chế, model phức tạp dễ học noise hơn là pattern thật.

Lý do model phức tạp chưa vượt baseline:

- Dữ liệu còn nhỏ.
- Feature chưa đủ phong phú: thiếu weather, traffic condition, holiday, bus stop dwell time, road segment, route polyline.
- Bài toán hiện tại tập trung vào một route cố định, nên statistical correction đã rất mạnh.
- Residual có nhiều noise không quan sát được.
- Một số time-bin ít sample, dễ gây overfit.

---

## 6. Overall Comparison

Bảng tổng hợp các phương pháp quan trọng:

```text
method                         MAE      RMSE     MAPE     P95      Main comment
api_eta                        37.956   56.887   18.34%   123.750  Best MAPE, strong no-training baseline
additive_global                37.249   52.804   19.58%   107.954  Simple, stable, strong fallback
additive_time_bin              37.965   51.794   20.45%   98.936   Strong tail-error reduction
additive_smoothed_time_bin     38.073   52.159   20.40%   103.111  More robust for sparse bins
ratio_global                   37.237   52.529   19.60%   108.039  Best MAE
ratio_time_bin                 38.018   51.202   20.69%   97.279   Best RMSE and P95
log_ratio_global               37.237   52.529   19.60%   108.039  Equivalent to ratio_global here
log_ratio_time_bin             38.018   51.202   20.69%   97.279   Equivalent to ratio_time_bin here
affine_global                  39.812   56.781   21.31%   111.769  Worse MAE, improves P95
affine_time_bin                41.758   56.609   23.00%   100.859  Stronger P95, poor MAE
MLP residual ETA               37.612   52.661   19.43%   101.684  Best ML model, not best overall
DeeprETA-like                  38.453   52.213   20.95%   99.546   Good P95, too complex for current gain
xgb_direct_eta                 44.914   n/a      n/a      99.304   Good P95, poor MAE
```

Best by objective:

```text
Best MAE:
API + Global Ratio / Global Log-ratio Correction

Best RMSE:
API + Time-bin Ratio / Time-bin Log-ratio Correction

Best P95:
API + Time-bin Ratio / Time-bin Log-ratio Correction

Best MAPE:
API baseline

Best additive fallback:
API + Global Median Residual

Best ML residual model:
MLP Residual ETA

Most production-friendly:
API + Global Ratio Correction or API + Global Median Residual

Best for bus ETA tail-error:
API + Time-bin Ratio Correction
```

---

## 7. Final Solution

### Primary Solution

Chọn:

```text
API + Time-bin Ratio Correction
```

Công thức:

```text
ratio_by_time_bin = median(train.actual_eta_secs / train.estimate_time | time_bin)
corrected_eta = estimate_time * ratio_by_time_bin
```

Kết quả:

```text
MAE  = 38.018s
RMSE = 51.202s
MAPE = 20.69%
P95  = 97.279s
R2   = 0.217
```

So với API baseline:

```text
MAE improvement = -0.16%
P95 improvement = +21.39%
```

Lý do chọn:

- Có P95 tốt nhất trong toàn bộ các phương pháp: `97.279s`.
- Có RMSE tốt nhất: `51.202s`.
- Giảm lỗi lớn mạnh nhất, phù hợp với bus ETA trên map.
- Khai thác được bias khác nhau theo khung giờ.
- Correction dạng tỷ lệ hợp lý hơn khi sai số API scale theo độ dài ETA.
- Công thức đơn giản, không cần model phức tạp, dễ triển khai và monitor.

Trade-off:

- MAE kém API baseline rất nhẹ: `38.018s` so với `37.956s`.
- MAPE cao hơn API baseline.
- Time-bin sparse có thể gây over-correction nếu triển khai trên dữ liệu ít.

Với mục tiêu sản phẩm là bus ETA đáng tin hơn, đặc biệt giảm các case sai nặng, trade-off này chấp nhận được.

### Secondary Solution

Chọn:

```text
API + Global Ratio Correction
```

Công thức:

```text
global_ratio = median(train.actual_eta_secs / train.estimate_time)
corrected_eta = estimate_time * global_ratio
```

Kết quả:

```text
MAE  = 37.237s
RMSE = 52.529s
MAPE = 19.60%
P95  = 108.039s
```

Lý do giữ làm secondary/fallback:

- Có MAE tốt nhất trong toàn bộ các phương pháp.
- Cải thiện MAE `+1.89%` và P95 `+12.70%` so với API baseline.
- Đơn giản hơn time-bin ratio.
- Ổn định hơn khi dữ liệu trong từng time-bin còn ít.
- Phù hợp nếu production muốn ưu tiên sai số trung bình và giảm rủi ro sparse-bin.

### Production Recommendation

Khuyến nghị triển khai theo hai tầng:

```text
Nếu time_bin có đủ dữ liệu lịch sử:
    corrected_eta = estimate_time * ratio_by_time_bin

Nếu time_bin thiếu dữ liệu hoặc ratio bất thường:
    corrected_eta = estimate_time * global_ratio
```

Có thể thêm guardrail:

```text
ratio_min <= applied_ratio <= ratio_max
```

và monitor định kỳ:

```text
MAE, RMSE, MAPE, P95 theo ngày / tuần / time_bin
```

Kết luận cuối cùng:

```text
Primary method:
API + Time-bin Ratio Correction

Fallback method:
API + Global Ratio Correction

Simple sanity-check baseline:
API + Global Median Residual
```

---

## 8. Limitations and Next Steps

Limitations:

- Dataset hiện tại còn nhỏ, test set chỉ có 47 samples.
- Một số time-bin có thể thiếu dữ liệu.

Next steps:

1. Đánh giá bằng rolling hoặc expanding-window validation.
2. Tune smoothing cho time-bin ratio để giảm rủi ro sparse-bin.
3. Thêm guardrail cho ratio khi triển khai production.
4. Log prediction, actual ETA và error theo time-bin để monitor drift.
5. Khi dữ liệu lớn hơn, thử lại MLP/XGBoost/DeepETA-like với feature phong phú hơn.

