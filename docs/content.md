Với đối tượng là **CTO**, slide không nên đi sâu vào công thức toán học. CTO quan tâm:

1. Business problem là gì?
2. Tại sao solution hiện tại chưa đủ tốt?
3. Đã thử những hướng nào?
4. Vì sao chọn solution cuối?
5. Impact khi đưa lên production?

Tôi đề xuất deck **5 slides** theo flow sau.

---

# Slide 1. Feature Overview

## ETA Calibration & Prediction System

### Problem

Hệ thống hiện tại đang sử dụng ETA từ Vietmap API làm nguồn dự báo thời gian di chuyển.

Tuy nhiên qua phân tích dữ liệu vận hành thực tế:

* API ETA tồn tại bias theo thời gian trong ngày
* Một số khung giờ cao điểm có sai số lớn
* Tail-error (các trường hợp sai số cực lớn) ảnh hưởng đến trải nghiệm người dùng

### Goal

Xây dựng lớp hiệu chỉnh ETA (ETA Correction Layer) nhằm:

* Giảm sai số dự báo trung bình
* Giảm các trường hợp dự báo lệch lớn
* Dễ triển khai production
* Không làm tăng latency đáng kể

### Solution Architecture

```text
Historical Trip Data
        ↓
Bias Analysis
        ↓
ETA Correction Layer
        ↓
Corrected ETA
        ↓
API / Product
```

**Điểm quan trọng cho CTO**

> Thay vì thay thế hoàn toàn Vietmap API bằng một model ML phức tạp, hệ thống tận dụng API hiện có và chỉ học phần sai số (correction), giúp giảm rủi ro triển khai và tiết kiệm chi phí vận hành.

---

# Slide 2. Methods Evaluated

## Candidate Solutions Evaluated

### Statistical Methods

#### Additive Correction

```text
ETA = API ETA + residual
```

* Global residual
* Time-bin residual

---

#### Ratio Correction

```text
ETA = API ETA × ratio
```

* Global Ratio
* Time-bin Ratio

---

#### Log-Ratio Correction

```text
ETA = API ETA × exp(log_ratio)
```

* Robust với outlier
* Phù hợp dữ liệu skewed

---

#### Affine Correction

```text
ETA = a × API ETA + b
```

* Huber Regression
* Học scale + offset

---

### Machine Learning Methods

* MLP Residual ETA
* Hour-bin MLP
* DeeprETA-like
* XGBoost Residual ETA
* XGBoost Direct ETA



---

# Slide 3. Benchmark Results

## Offline Evaluation Results

| Method         | MAE (s) | RMSE (s) | P95 (s) |
| -------------- | ------- | -------- | ------- |
| API Baseline   | 37.96   | 56.89    | 123.75  |
| Ratio Global   | 37.24   | 52.53    | 108.04  |
| Ratio Time-bin | 38.02   | 51.20    | 97.28   |
| MLP Residual   | 37.61   | 52.66    | 101.68  |
| DeeprETA-like  | 38.45   | 52.21    | 99.55   |



### Key Findings

#### Average Error

Best:

```text
Ratio Global
MAE = 37.24s
(+1.89%)
```

#### Tail Error

Best:

```text
Ratio Time-bin
P95 = 97.28s
(+21.39%)
```

#### Surprising Observation

Các phương pháp Statistical Correction vượt qua:

* MLP
* XGBoost
* DeeprETA-like

trên dữ liệu hiện tại. 

### Reason

* Dataset còn nhỏ
* Route tương đối cố định
* Feature space hạn chế

=> Statistical correction có bias-variance tradeoff tốt hơn. 

---

# Slide 4. Selected Solution

## Recommended Production Solution

### Primary

### Time-bin Ratio Correction

```text
Corrected ETA
=
API ETA × Median Ratio(Time Bin)
```

### Why?

#### Best Tail Error

```text
P95
123.75s
→
97.28s

Improvement = 21.39%
```

#### Best RMSE

```text
56.89s
→
51.20s
```

#### Production Friendly

* Không cần model serving
* Không cần GPU
* Không cần online training
* Triển khai đơn giản



---

### Secondary Fallback

Global Ratio Correction

Ưu điểm:

* MAE tốt nhất
* Ổn định hơn khi dữ liệu từng time-bin còn ít



---

# Slide 5. Production Metrics & Rollout Plan

## Production Success Metrics

### Solution Metrics (Offline)

Theo dõi:

* MAE
* RMSE
* P95 Error
* MAPE

Mục tiêu:

```text
P95 Improvement > 15%
MAE Improvement > 1%
```

---

### Production Metrics (Online)

#### Accuracy Metrics

```text
ETA Error
Actual Arrival Time
-
Predicted ETA
```

Monitor:

* MAE Daily
* P95 Daily
* Error by Time Bin

---

#### System Metrics

Latency:

```text
P99 < 10 ms
```

Availability:

```text
99.9%
```

Correction coverage:

```text
100% requests
```

---

### Rollout Strategy

Phase 1

```text
Shadow Mode
```

* Log prediction
* Không ảnh hưởng user

Phase 2

```text
A/B Test
```

* 10%-20% traffic

Phase 3

```text
Full Production
```

* Continuous monitoring
* Weekly recalibration

---

# Thông điệp cuối cùng cho CTO

Nếu chỉ có **1 slide kết luận**, tôi sẽ dùng:

> **Time-bin Ratio Correction được chọn làm production solution vì đạt mức giảm P95 tốt nhất (-21.39%), triển khai cực nhẹ, không yêu cầu model serving phức tạp và hiện đang outperform toàn bộ các mô hình ML/Deep Learning đã thử nghiệm trên dataset hiện tại.** 

Đây là thông điệp CTO thường quan tâm nhất: **impact + simplicity + maintainability**.
