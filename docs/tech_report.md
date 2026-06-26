# Technical Report: ETA Prediction and Correction System

## Table of Contents

- [1. Executive Summary](#1-executive-summary)
- [2. Problem Definition](#2-problem-definition)
- [3. System Architecture](#3-system-architecture)
- [4. Dataset and Evaluation Setup](#4-dataset-and-evaluation-setup)
- [5. Methods Evaluated](#5-methods-evaluated)
- [6. Overall Benchmark](#6-overall-benchmark)
- [7. Technical Interpretation](#7-technical-interpretation)
- [8. Selected Production Approach](#8-selected-production-approach)
- [9. Production Design](#9-production-design)
- [10. Monitoring and Rollout Plan](#10-monitoring-and-rollout-plan)
- [11. Limitations](#11-limitations)
- [12. Future Work](#12-future-work)
- [13. Final Recommendation](#13-final-recommendation)

## 1. Executive Summary

Dự án xây dựng hệ thống dự đoán và hiệu chỉnh thời gian di chuyển thực tế (ETA - Estimated Time of Arrival) dựa trên ETA ban đầu từ Vietmap Route API. Thay vì thay thế hoàn toàn routing engine bằng một mô hình học máy phức tạp, hệ thống sử dụng Vietmap API như baseline mạnh, sau đó học hoặc ước lượng phần sai lệch giữa thời gian thực tế và thời gian API dự đoán.

Hướng tiếp cận chính là **ETA correction / residual correction**:

```text
api_eta = estimate_time
actual_eta = delta_time
error = actual_eta - api_eta
corrected_eta = f(api_eta, time_context)
```

Kết quả thực nghiệm cho thấy các phương pháp hiệu chỉnh thống kê đơn giản đang hiệu quả hơn các mô hình phức tạp như MLP, XGBoost và DeeprETA-like trong điều kiện dữ liệu hiện tại. Phương pháp được khuyến nghị làm hướng chính là:

```text
API + Time-bin Ratio Correction
corrected_eta = estimate_time * median_ratio_by_time_bin
```

Lý do chọn:

- Giảm mạnh tail-error: P95 từ `123.750s` xuống `97.279s`, tương đương cải thiện `21.39%`.
- Có RMSE tốt nhất trong các phương pháp đã thử: `51.202s`.
- Không cần model serving phức tạp, không cần GPU, latency gần như không đáng kể.
- Dễ giải thích và phù hợp với quan sát rằng sai số của API thay đổi theo khung giờ.

Phương pháp dự phòng nên giữ là **API + Global Ratio Correction**, vì có MAE tốt nhất:

```text
MAE = 37.237s
MAE improvement vs API = +1.89%
```

---

## 2. Problem Definition

### 2.1 Bối cảnh

Hệ thống hiện tại có sẵn ETA từ Vietmap API trong trường `estimate_time`, đồng thời có thời gian di chuyển thực tế trong trường `delta_time`. Vietmap API đã cung cấp một baseline tương đối tốt vì routing engine đã biết thông tin đường đi, khoảng cách và logic định tuyến.

Tuy nhiên, dữ liệu vận hành thực tế cho thấy API ETA vẫn có sai lệch:

- Một số khung giờ có xu hướng bị underestimate.
- Giờ cao điểm có thể tạo ra các lỗi lớn.
- Tail-error cao gây ảnh hưởng đến trải nghiệm người dùng và độ tin cậy vận hành.

Vì vậy, bài toán phù hợp không phải là học trực tiếp:

```text
features -> actual_eta
```

mà là hiệu chỉnh dự báo có sẵn:

```text
api_eta + correction -> corrected_eta
```

### 2.2 Mục tiêu kỹ thuật

Hệ thống cần đạt các mục tiêu sau:

- Giảm sai số trung bình, đo bằng MAE.
- Giảm lỗi lớn ở nhóm xấu nhất, đo bằng P95.
- Không làm tăng đáng kể latency suy luận.
- Dễ triển khai trong production.
- Có khả năng giải thích và kiểm soát khi dữ liệu còn nhỏ.

---

## 3. System Architecture

Hệ thống được thiết kế theo kiến trúc phân lớp:

```text
Frontend public/
    |
    | route/search/map requests
    v
Node.js Backend Proxy server.js
    |
    | protected API calls
    v
Vietmap Route / Geocoding API

Frontend public/
    |
    | ETA prediction request
    v
Node.js Backend Proxy server.js
    |
    | model inference API
    v
FastAPI Model Server api/app.py
    |
    v
Model Artifacts / Statistical Correction Tables
```

### 3.1 Frontend

Frontend nằm trong `public/`, sử dụng HTML/CSS/JavaScript thuần và Leaflet để hiển thị bản đồ. Các chức năng chính:

- Nhập điểm đi và điểm đến.
- Gọi API tìm kiếm, định tuyến và hiển thị route.
- Hiển thị ETA từ Vietmap API và ETA sau hiệu chỉnh.
- Hỗ trợ so sánh kết quả giữa các mô hình hoặc phương pháp correction.

### 3.2 Node.js Backend Proxy

`server.js` đóng vai trò API Gateway giữa trình duyệt và các dịch vụ bên ngoài:

- Proxy Vietmap Route API, Search API và Place API.
- Ẩn `VIETMAP_API_KEY` khỏi frontend.
- Có thể proxy map tiles nếu cấu hình `VIETMAP_TILE_API_KEY` và `VIETMAP_TILE_URL_TEMPLATE`.
- Là lớp trung gian để frontend gọi FastAPI model server.

Thiết kế này giúp giảm rủi ro lộ API key và giữ frontend đơn giản.

### 3.3 FastAPI Model Server

`api/app.py` phục vụ inference cho các mô hình ETA:

- Endpoint danh sách mô hình: `/api/eta/models`.
- Endpoint dự đoán ETA: `/api/eta/predict`.
- Nạp artifact `.joblib` cho scikit-learn/XGBoost và `.pt` cho PyTorch.

Trong giai đoạn hiện tại, do phương pháp thống kê đang outperform các mô hình phức tạp, FastAPI có thể phục vụ cả model artifacts và correction tables.

### 3.4 Training and Experiment Pipeline

`eta_modeling/` chứa pipeline huấn luyện, đánh giá và tracking:

- Cấu hình huấn luyện trong `eta_modeling/configs/`.
- Mã nguồn xử lý dữ liệu và training trong `eta_modeling/src/`.
- Artifacts, metrics và plots trong `eta_modeling/artifacts/`.
- MLflow dùng để tracking các thử nghiệm.

Lệnh so sánh mô hình:

```bash
cd eta_modeling
python -m src.training.compare_models --config configs/config.yaml
```

Kết quả được ghi vào:

```text
eta_modeling/artifacts/metrics/model_comparison.csv
eta_modeling/artifacts/plots/model_comparison_mae_p95.png
```

---

## 4. Dataset and Evaluation Setup

### 4.1 Dữ liệu

Các trường dữ liệu chính:

```text
estimate_time: ETA từ Vietmap / routing API
delta_time: thời gian di chuyển thực tế
hour: giờ khởi hành
time_bin: nhóm khung giờ
```

Một số thử nghiệm deep-like sử dụng thêm các trường:

```text
lat
lng
destination_lat
destination_lng
is_raining
rain_level
day_of_week
rush_hour
is_weekend
```

Tuy nhiên, signal chính hiện tại vẫn xoay quanh `estimate_time` và thông tin thời gian. Đây là lý do các phương pháp thống kê có lợi thế hơn mô hình phức tạp.

### 4.2 Data Split

Dữ liệu được chia theo thời gian thay vì random split:

```text
Train: 70%  = 215 rows
Validation: 15% = 46 rows
Test: 15% = 47 rows
```

Chronological split phù hợp với bài toán ETA vì dữ liệu có yếu tố thời gian. Cách chia này tránh leakage, tức model học từ dữ liệu tương lai để dự đoán dữ liệu quá khứ.

### 4.3 Evaluation Metrics

Các chỉ số chính:

| Metric | Ý nghĩa |
|---|---|
| MAE | Sai số tuyệt đối trung bình, đơn vị giây |
| RMSE | Phạt nặng lỗi lớn, phản ánh độ ổn định |
| MAPE | Sai số phần trăm, nhạy với chuyến ngắn |
| P50 | Median absolute error |
| P90 | Lỗi ở phân vị 90 |
| P95 | Tail-error ở nhóm 5% tệ nhất |
| R2 | Mức giải thích phương sai so với actual ETA |

Trong production, MAE và P95 là hai chỉ số quan trọng nhất:

- MAE phản ánh trải nghiệm trung bình.
- P95 phản ánh các tình huống sai lệch lớn, thường gây tác động vận hành mạnh hơn.

---

## 5. Methods Evaluated

### 5.1 API Baseline

Phương pháp baseline dùng trực tiếp ETA từ Vietmap:

```text
prediction = estimate_time
```

Kết quả trên test set:

| Metric | Value |
|---|---:|
| MAE | 37.956s |
| MAPE | 18.34% |
| RMSE | 56.887s |
| P50 | 23.079s |
| P95 | 123.750s |
| R2 | 0.034 |

Baseline này mạnh vì routing API đã có thông tin định tuyến tốt. Điểm yếu chính là P95 còn cao, cho thấy vẫn tồn tại các case dự đoán sai lớn.

### 5.2 Additive Global Median Residual

Phương pháp này học median residual toàn cục trên train set:

```text
residual = actual_eta - api_eta
global_median_residual = 15.795s
corrected_eta = estimate_time + global_median_residual
```

Kết quả:

| Metric | Value |
|---|---:|
| MAE | 37.249s |
| MAPE | 19.58% |
| RMSE | 52.804s |
| P50 | 30.098s |
| P95 | 107.954s |

So với API baseline:

| Metric | Improvement |
|---|---:|
| MAE | +1.86% |
| P95 | +12.76% |

Ưu điểm là rất đơn giản, dễ giải thích và ổn định. Nhược điểm là cộng cùng một offset cho mọi chuyến đi, không xét đến độ dài chuyến hoặc khung giờ.

### 5.3 Additive Time-bin Median Residual

Phương pháp này chia dữ liệu theo khung giờ, sau đó cộng median residual tương ứng:

```text
corrected_eta = estimate_time + median_residual_by_time_bin
```

Các time-bin:

| Time-bin | Khoảng giờ | Ý nghĩa |
|---|---|---|
| early_morning | 04:00-06:00 | Giao thông thông thoáng |
| morning_peak | 07:00-09:00 | Cao điểm sáng |
| off_peak_day | 10:00-14:00 | Thấp điểm ban ngày |
| evening_peak | 15:00-18:00 | Cao điểm chiều |
| late_evening | 19:00-21:00 | Tối muộn |
| other | Ngoài các khung trên | Khác |

Kết quả:

| Metric | Value |
|---|---:|
| MAE | 37.965s |
| MAPE | 20.45% |
| RMSE | 51.794s |
| P50 | 32.085s |
| P95 | 98.936s |

Phương pháp này không cải thiện MAE, nhưng giảm P95 mạnh. Điều này cho thấy correction theo thời gian có khả năng giảm các lỗi lớn.

### 5.4 Smoothed Time-bin Residual

Để giảm overfit ở các bin ít dữ liệu, hệ thống thử smoothing:

```text
smoothed_residual_bin =
    w * bin_residual + (1 - w) * global_residual

w = count_bin / (count_bin + prior_strength)
```

Kết quả:

| Metric | Value |
|---|---:|
| MAE | 38.073s |
| MAPE | 20.40% |
| RMSE | 52.159s |
| P50 | 31.447s |
| P95 | 103.111s |

Smoothing robust hơn về mặt phương pháp, nhưng trên test set hiện tại chưa thắng raw time-bin.

### 5.5 Global Ratio Correction

Thay vì cộng offset cố định, ratio correction nhân ETA theo tỷ lệ:

```text
ratio = actual_eta / estimate_time
corrected_eta = estimate_time * median_ratio
```

Global ratio học được:

```text
global_ratio = 1.1009
```

Điều này cho thấy ở mức median trên train set, actual time cao hơn API ETA khoảng `10.09%`.

Kết quả:

| Metric | Value |
|---|---:|
| MAE | 37.237s |
| RMSE | 52.529s |
| MAPE | 19.60% |
| P50 | 30.309s |
| P90 | 68.764s |
| P95 | 108.039s |
| R2 | 0.176 |

So với API baseline:

| Metric | Improvement |
|---|---:|
| MAE | +1.89% |
| P95 | +12.70% |

Đây là phương pháp tốt nhất theo MAE trong toàn bộ nhóm thống kê đã thử.

### 5.6 Time-bin Ratio Correction

Phương pháp này tính ratio riêng cho từng time-bin:

```text
corrected_eta = estimate_time * median_ratio_by_time_bin
```

Kết quả:

| Metric | Value |
|---|---:|
| MAE | 38.018s |
| RMSE | 51.202s |
| MAPE | 20.69% |
| P50 | 32.327s |
| P90 | 66.063s |
| P95 | 97.279s |
| R2 | 0.217 |

So với API baseline:

| Metric | Improvement |
|---|---:|
| MAE | -0.16% |
| P95 | +21.39% |

Đây là phương pháp tốt nhất theo RMSE và P95. Trade-off là MAE trung bình tăng nhẹ so với API baseline.

### 5.7 Log-ratio Correction

Log-ratio chuyển tỷ lệ sang không gian log:

```text
log_residual = log(actual_eta / estimate_time)
corrected_eta = estimate_time * exp(log_residual)
```

Kết quả gần như trùng với ratio correction:

| Method | MAE | RMSE | P95 |
|---|---:|---:|---:|
| log_ratio_global | 37.237s | 52.529s | 108.039s |
| log_ratio_time_bin | 38.018s | 51.202s | 97.279s |

Vì estimator hiện tại là median và ratio luôn dương, `exp(median(log(ratio)))` gần như tương đương `median(ratio)`. Log-ratio có thể hữu ích hơn nếu sau này dùng mean, regression hoặc model học trong log-space.

### 5.8 Affine Huber Regression

Affine correction học đồng thời scale và offset:

```text
corrected_eta = a * estimate_time + b
```

Mô hình được fit bằng Huber Regression để giảm ảnh hưởng của outlier:

```text
epsilon = 1.35
alpha = 0.0001
max_iter = 1000
```

Global affine học được:

```text
a = 0.1595
b = 150.7553
```

Kết quả:

| Method | MAE | RMSE | MAPE | P95 |
|---|---:|---:|---:|---:|
| affine_global | 39.812s | 56.781s | 21.31% | 111.769s |
| affine_time_bin | 41.758s | 56.609s | 23.00% | 100.859s |
| affine_smoothed_time_bin | 40.403s | 56.443s | 21.97% | 107.764s |

Affine cải thiện P95 so với API baseline, nhưng làm MAE xấu hơn đáng kể. Hệ số `a` nhỏ và `b` lớn cho thấy model bị kéo mạnh về intercept, không giữ tốt scale tự nhiên của API ETA.

### 5.9 Machine Learning and Deep Models

Các mô hình ML/deep đã thử:

- MLP Residual ETA.
- Hour-bin MLP Residual ETA.
- DeeprETA-like.
- XGBoost residual ETA.
- Hour-bin XGBoost residual ETA.
- XGBoost direct ETA.

Kết quả chính:

| Method | MAE | RMSE | MAPE | P95 |
|---|---:|---:|---:|---:|
| MLP Residual ETA | 37.612s | 52.661s | 19.43% | 101.684s |
| Hour-bin MLP Residual ETA | 38.297s | 54.500s | 19.41% | 107.584s |
| DeeprETA-like | 38.453s | 52.213s | 20.95% | 99.546s |
| xgb_residual_eta | 42.639s | n/a | n/a | 106.104s |
| hour_bin_xgb_residual_eta | 43.383s | n/a | n/a | 113.415s |
| xgb_direct_eta | 44.914s | n/a | n/a | 99.304s |

MLP là mô hình ML residual tốt nhất hiện tại, nhưng vẫn không thắng `ratio_global` theo MAE và không thắng `ratio_time_bin` theo P95. DeeprETA-like có P95 khá tốt, nhưng MAE và MAPE kém hơn. XGBoost có MAE xấu rõ rệt.

---

## 6. Overall Benchmark

### 6.1 Main Comparison

| Method | MAE | RMSE | MAPE | P95 | MAE vs API | P95 vs API |
|---|---:|---:|---:|---:|---:|---:|
| API baseline | 37.956s | 56.887s | 18.34% | 123.750s | 0.00% | 0.00% |
| additive_global | 37.249s | 52.804s | 19.58% | 107.954s | +1.86% | +12.76% |
| additive_time_bin | 37.965s | 51.794s | 20.45% | 98.936s | -0.02% | +20.05% |
| additive_smoothed_time_bin | 38.073s | 52.159s | 20.40% | 103.111s | -0.31% | +16.68% |
| ratio_global | 37.237s | 52.529s | 19.60% | 108.039s | +1.89% | +12.70% |
| ratio_time_bin | 38.018s | 51.202s | 20.69% | 97.279s | -0.16% | +21.39% |
| log_ratio_global | 37.237s | 52.529s | 19.60% | 108.039s | +1.89% | +12.70% |
| log_ratio_time_bin | 38.018s | 51.202s | 20.69% | 97.279s | -0.16% | +21.39% |
| affine_global | 39.812s | 56.781s | 21.31% | 111.769s | -4.89% | +9.68% |
| affine_time_bin | 41.758s | 56.609s | 23.00% | 100.859s | -10.02% | +18.50% |
| MLP Residual ETA | 37.612s | 52.661s | 19.43% | 101.684s | +0.91% | +17.83% |
| DeeprETA-like | 38.453s | 52.213s | 20.95% | 99.546s | -1.31% | +19.56% |

### 6.2 Best by Metric

| Objective | Best Method | Result |
|---|---|---:|
| Best MAE | ratio_global / log_ratio_global | 37.237s |
| Best RMSE | ratio_time_bin / log_ratio_time_bin | 51.202s |
| Best P95 | ratio_time_bin / log_ratio_time_bin | 97.279s |
| Best MAPE | API baseline | 18.34% |
| Best ML residual model | MLP Residual ETA | MAE 37.612s, P95 101.684s |
| Most interpretable | additive_global / ratio_global | Simple global correction |
| Best production tail-error reduction | ratio_time_bin | P95 improvement 21.39% |

---

## 7. Technical Interpretation

### 7.1 Vì sao correction thống kê thắng model phức tạp?

Các mô hình phức tạp chưa vượt trội vì các lý do sau:

1. Dataset còn nhỏ. Test set hiện tại chỉ có 47 samples, trong khi MLP, XGBoost và deep models cần nhiều dữ liệu hơn để học pattern ổn định.
2. Feature space còn hạn chế. Signal chính vẫn là `estimate_time` và thông tin thời gian; thiếu traffic condition, holiday, bus frequency, road segment, incident, weather chất lượng cao hoặc route sequence.
3. Bài toán tập trung vào một route tương đối cố định. Với một route cố định, bias của API có thể được mô tả tốt bằng global correction hoặc time-bin correction.
4. Residual có nhiều noise không quan sát được. Các yếu tố như chờ đèn đỏ, dừng đón khách, ùn tắc bất thường hoặc sự cố giao thông chưa được encode đầy đủ.
5. Một số time-bin có ít sample. Model phức tạp dễ học noise từ các bin sparse.

Nói ngắn gọn, statistical correction có bias-variance tradeoff tốt hơn trong giai đoạn dữ liệu hiện tại.

### 7.2 Additive vs Ratio

Additive correction phù hợp khi API sai theo offset cố định:

```text
actual_eta ~= api_eta + constant
```

Ratio correction phù hợp khi API sai theo tỷ lệ:

```text
actual_eta ~= api_eta * ratio
```

Kết quả cho thấy ratio tốt hơn nhẹ ở global MAE và tốt hơn ở time-bin P95/RMSE. Điều này gợi ý sai số của API không chỉ là offset cố định, mà có thành phần scale theo ETA.

### 7.3 Global vs Time-bin

Global correction ổn định hơn vì dùng toàn bộ train set:

```text
corrected_eta = estimate_time * global_ratio
```

Time-bin correction nhạy hơn với dữ liệu từng khung giờ:

```text
corrected_eta = estimate_time * ratio_by_time_bin
```

Trade-off:

- Global ratio có MAE tốt nhất và ổn định hơn.
- Time-bin ratio giảm P95 tốt nhất nhưng làm MAE tăng nhẹ.

Vì mục tiêu production thường ưu tiên giảm lỗi lớn gây ảnh hưởng vận hành, time-bin ratio là lựa chọn primary hợp lý.

---

## 8. Selected Production Approach

### 8.1 Primary Method: API + Time-bin Ratio Correction

Công thức:

```text
corrected_eta = estimate_time * median_ratio_by_time_bin
```

Trong đó:

```text
median_ratio_by_time_bin =
    median(actual_eta / estimate_time | time_bin)
```

Lý do chọn:

- P95 tốt nhất: `97.279s`.
- RMSE tốt nhất: `51.202s`.
- Cải thiện P95 so với API baseline: `+21.39%`.
- Không cần GPU hoặc model serving phức tạp.
- Có thể triển khai như một bảng lookup nhỏ theo time-bin.
- Dễ debug và rollback.

Trade-off:

- MAE là `38.018s`, kém API baseline nhẹ `0.16%`.
- MAPE tăng lên `20.69%`, chủ yếu do correction có thể làm lỗi tương đối lớn hơn ở chuyến ngắn.
- Cần giám sát kỹ các bin ít dữ liệu.

### 8.2 Secondary Method: API + Global Ratio Correction

Công thức:

```text
corrected_eta = estimate_time * global_ratio
```

Lý do giữ làm fallback:

- MAE tốt nhất: `37.237s`.
- Đơn giản và ổn định hơn khi dữ liệu time-bin ít.
- Cải thiện P95 đáng kể: `+12.70%`.
- Phù hợp nếu cần ưu tiên sai số trung bình thay vì tail-error.

### 8.3 Methods Not Recommended as Primary

Không nên chọn Affine Huber Regression làm primary ở giai đoạn này:

- MAE kém API baseline.
- Hệ số global affine bị kéo mạnh về intercept.
- Cần nhiều split theo thời gian hơn để xác nhận generalization.

Không nên chọn XGBoost làm primary:

- MAE cao hơn baseline rõ rệt.
- Feature hiện tại chưa đủ giàu để boosting học pattern ổn định.

Chưa nên chọn DeeprETA-like làm primary:

- Kiến trúc phức tạp hơn nhiều so với lợi ích đạt được.
- Deep ETA cần dữ liệu route sequence, road segment, traffic và multi-route phong phú hơn.

---

## 9. Production Design

### 9.1 Inference Flow

Luồng inference đề xuất:

```text
1. Frontend yêu cầu route từ Node.js proxy
2. Node.js proxy gọi Vietmap Route API
3. Vietmap trả về estimate_time
4. Hệ thống xác định time_bin từ thời điểm khởi hành
5. Lookup median_ratio_by_time_bin
6. corrected_eta = estimate_time * ratio
7. Trả về API ETA và corrected ETA cho frontend/product
```

### 9.2 Correction Table

Correction table có thể lưu dưới dạng JSON hoặc artifact versioned:

```json
{
  "global_ratio": 1.1009,
  "time_bin_ratio": {
    "early_morning": 1.0,
    "morning_peak": 1.0,
    "off_peak_day": 1.0,
    "evening_peak": 1.0,
    "late_evening": 1.0,
    "other": 1.0
  },
  "metadata": {
    "train_rows": 215,
    "validation_rows": 46,
    "test_rows": 47,
    "created_at": "YYYY-MM-DD"
  }
}
```

Các giá trị `1.0` trong ví dụ cần được thay bằng median ratio thực tế từ artifact huấn luyện.

### 9.3 Fallback Logic

Đề xuất fallback:

```text
if time_bin ratio exists and count_bin >= min_samples:
    use time_bin_ratio
else:
    use global_ratio
```

Nếu muốn production-safe hơn, có thể dùng smoothing:

```text
ratio = w * ratio_bin + (1 - w) * global_ratio
w = count_bin / (count_bin + prior_strength)
```

Dù validation hiện tại chọn `k = 0`, smoothing vẫn nên được giữ như cơ chế bảo vệ khi mở rộng dữ liệu hoặc route.

### 9.4 Latency

Time-bin ratio correction chỉ cần:

- Xác định time-bin.
- Lookup một số thực.
- Nhân với `estimate_time`.

Latency kỳ vọng nhỏ hơn nhiều so với network call tới routing API. Mục tiêu production hợp lý:

```text
P99 correction latency < 10 ms
```

---

## 10. Monitoring and Rollout Plan

### 10.1 Offline Success Metrics

Mục tiêu offline:

```text
P95 improvement > 15%
MAE improvement >= 0% hoặc giảm không đáng kể nếu P95 cải thiện mạnh
```

Các metric cần theo dõi:

- MAE.
- RMSE.
- MAPE.
- P50/P90/P95.
- Error by time-bin.
- Coverage theo time-bin.

### 10.2 Online Monitoring

Sau khi có actual arrival time, log:

```text
api_eta
corrected_eta
actual_eta
time_bin
route_id
request_timestamp
error_api = actual_eta - api_eta
error_corrected = actual_eta - corrected_eta
```

Dashboard production nên có:

- MAE daily.
- P95 daily.
- MAPE daily.
- Error distribution by time-bin.
- Drift của ratio theo tuần.
- Số sample theo time-bin.
- Tỷ lệ fallback sang global ratio.

### 10.3 Rollout Strategy

Phase 1: Shadow mode

```text
Tính corrected ETA và log lại, nhưng chưa hiển thị cho user.
```

Mục tiêu:

- Kiểm tra metric online.
- So sánh API ETA và corrected ETA trên dữ liệu thật.
- Xác nhận không có lỗi integration.

Phase 2: A/B test

```text
Áp dụng corrected ETA cho 10%-20% traffic.
```

Theo dõi:

- Online MAE/P95.
- User-facing ETA deviation.
- Latency.
- Error theo time-bin.

Phase 3: Full rollout

```text
Triển khai toàn bộ traffic nếu P95 cải thiện ổn định và MAE không suy giảm đáng kể.
```

Phase 4: Weekly recalibration

```text
Cập nhật ratio table định kỳ theo dữ liệu mới.
```

---

## 11. Limitations

Các giới hạn hiện tại:

- Dataset nhỏ: train 215 rows, validation 46 rows, test 47 rows.
- Test set nhỏ nên kết quả có thể nhạy với một vài sample bất thường.
- Feature còn hạn chế, chưa đủ để deep/boosting models phát huy lợi thế.
- Một số time-bin có thể sparse.
- Kết quả mới được đánh giá trên một split theo thời gian; cần thêm rolling hoặc expanding window validation.
- Chưa có đánh giá online thực tế sau khi triển khai shadow mode.

---

## 12. Future Work

Các hướng cải thiện tiếp theo:

1. Thu thập thêm dữ liệu theo nhiều ngày, nhiều tuần và nhiều điều kiện giao thông.
2. Đánh giá bằng rolling time split hoặc expanding window split để kiểm tra generalization.
3. Thêm feature:
   - day of week
   - weekend / holiday
   - weather
   - traffic condition
   - bus dwell time
   - stop density
   - route segment
   - incident signal
4. Tách correction theo route nếu hệ thống mở rộng nhiều tuyến.
5. Tune smoothing strength cho time-bin ratio với nhiều split hơn.
6. Thử hierarchical correction:

```text
global_ratio -> route_ratio -> time_bin_ratio -> route_time_bin_ratio
```

7. Chỉ quay lại MLP/XGBoost/Deep ETA khi dữ liệu và feature đủ lớn.

---

## 13. Final Recommendation

Với dữ liệu và kết quả hiện tại, lựa chọn hợp lý nhất là:

```text
Primary:
API + Time-bin Ratio Correction

Fallback:
API + Global Ratio Correction
```

Thông điệp kỹ thuật chính:

> Time-bin Ratio Correction là phương pháp phù hợp nhất để đưa vào production ở giai đoạn hiện tại vì giảm P95 mạnh nhất, có RMSE tốt nhất, triển khai rất nhẹ và outperform các mô hình ML/deep phức tạp trên dataset hiện có.

Thông điệp vận hành:

> Hệ thống không thay thế Vietmap API mà thêm một lớp calibration mỏng phía sau API. Cách này tận dụng routing engine hiện có, giảm rủi ro triển khai, giữ latency thấp và vẫn cải thiện đáng kể các case ETA sai lớn.
