# Report: ETA Correction Using Residual Learning

## 1. Problem Context

Trong dữ liệu hiện tại đã có sẵn `estimate_time` từ map/routing API và `delta_time` là thời gian di chuyển thực tế,  Vì vậy, hướng tiếp cận bài toán hợp lý là **ETA correction / residual correction**.

Thay vì học trực tiếp:

```text
features → actual_eta
```

ta xem ETA từ API là baseline ban đầu, sau đó học phần sai lệch giữa thực tế và API:

```text
residual = actual_time - estimate_time
```

Final prediction:

```text
corrected_eta = estimate_time + predicted_residual
```

Cách tiếp cận này phù hợp vì routing API đã cung cấp một estimate tương đối tốt, nhưng có thể tồn tại bias, ví dụ thường dự đoán thấp hơn thực tế ở một số khung giờ.

---

## 2. Dataset and Evaluation Setup

Dữ liệu gồm các cột chính:

```text
estimate_time: ETA từ map/routing API
delta_time: thời gian di chuyển thực tế
hour / time_bin: thông tin thời điểm khởi hành
```

Dữ liệu được chia theo thời gian:

```text
Train: 70%
Validation: 15%
Test: 15%
```

Cách chia theo thời gian phù hợp hơn random split vì bài toán ETA có yếu tố temporal, tránh việc model học từ dữ liệu tương lai để dự đoán quá khứ.

Các metrics được sử dụng:

```text
MAE: sai số tuyệt đối trung bình
RMSE: nhạy với các lỗi lớn
MAPE: sai số phần trăm
P95: lỗi lớn ở nhóm 5% tệ nhất
```

---

## 3. Methods Applied

### 3.1 Vietmap / API Baseline

Phương pháp đầu tiên là sử dụng trực tiếp ETA từ routing API:

```text
prediction = estimate_time
```

Kết quả trên test set:

```text
MAE  = 37.956s
MAPE = 18.34%
RMSE = 56.887s
P50  = 23.079s
P95  = 123.750s
```

Điểm mạnh:

* Đây là baseline thực tế và có sẵn từ routing engine.
* MAPE tốt nhất trong các phương pháp hiện tại.
* Không cần training, dễ triển khai.

Điểm yếu:

* P95 khá cao, nghĩa là vẫn có những case API sai lớn.
* API có xu hướng underestimate trong một số khung giờ.
* Không học được bias thực tế từ dữ liệu lịch sử của route cụ thể.

Nhận xét:

API baseline là một baseline mạnh. Tuy nhiên, vì route thực tế có thể chịu ảnh hưởng bởi traffic pattern, bus operation, thời điểm trong ngày, nên API estimate có thể bị lệch so với actual travel time.

---

### 3.2 API + Global Median Residual

Phương pháp này tính residual trên train set:

```text
residual = delta_time - estimate_time
```

Sau đó lấy median residual toàn cục:

```text
global_median_residual = 15.795s
```

Prediction:

```text
corrected_eta = estimate_time + global_median_residual
```

Kết quả trên test set:

```text
MAE  = 37.249s
MAPE = 19.58%
RMSE = 52.804s
P50  = 30.098s
P95  = 107.954s
```

So với API baseline:

```text
MAE improvement  = +1.86%
P95 improvement  = +12.76%
```

Điểm mạnh:

* Có MAE tốt nhất trong các phương pháp hiện tại.
* Cải thiện RMSE và P95 so với API baseline.
* Rất đơn giản, dễ giải thích, ít rủi ro overfit.
* Phù hợp khi API có bias chung, ví dụ thường dự đoán thấp hơn thực tế.

Điểm yếu:

* MAPE xấu hơn API baseline.
* Chỉ cộng một correction cố định cho mọi khung giờ, không phân biệt morning peak, evening peak hay off-peak.
* Không tận dụng được pattern residual khác nhau theo thời điểm.


--> Phương pháp này cộng cùng một residual cho toàn bộ sample, nó có thể cải thiện các case API underestimate, nhưng cũng có thể làm overestimate ở các chuyến ngắn hoặc các khung giờ mà API đã dự đoán khá đúng. Do đó MAPE tăng lên.

Nhận xét:

Đây là phương pháp mạnh nhất nếu ưu tiên MAE và tính đơn giản. Hiện tại, đây là candidate tốt nhất để làm primary baseline.

---

### 3.3 API + Raw Time-bin Median Residual

Phương pháp này chia thời gian thành các bin:

```text
early_morning: 4–6
morning_peak: 7–9
off_peak_day: 10–14
evening_peak: 15–18
late_evening: 19–21
```

Sau đó tính median residual theo từng bin:

```text
corrected_eta = estimate_time + median_residual_by_time_bin
```

Kết quả trên test set:

```text
MAE  = 37.965s
MAPE = 20.45%
RMSE = 51.794s
P50  = 32.085s
P95  = 98.936s
```

So với API baseline:

```text
MAE improvement  = -0.024%
P95 improvement  = +20.05%
```

Điểm mạnh:

* Có P95 tốt nhất trong các phương pháp hiện tại.
* Học được bias khác nhau theo từng khung giờ.
* Phù hợp với quan sát rằng API có thể underestimate nhiều hơn trong morning/evening peak.

Điểm yếu:

* Không cải thiện MAE so với API baseline.
* MAPE tệ hơn API baseline.
* Dễ bị ảnh hưởng bởi các bin có ít sample, ví dụ late evening.
* Có nguy cơ over-correction nếu median residual của một bin không đại diện tốt cho thực tế.


--> Raw time-bin residual tin hoàn toàn vào residual của từng bin. Nếu một bin có nhiều dữ liệu, ví dụ evening peak, thống kê sẽ tương đối ổn định. Tuy nhiên nếu một bin có rất ít sample, ví dụ late evening, median residual có thể bị lệch bởi vài chuyến bất thường. Điều này làm model dễ overfit vào dữ liệu ít.

Nhận xét:

Phương pháp này không phải tốt nhất theo MAE, nhưng rất quan trọng vì nó giảm P95 mạnh nhất. Điều này cho thấy correction theo time-bin có khả năng giảm các case ETA sai nặng.

---

### 3.4 API + Smoothed Time-bin Median Residual

Phương pháp này là biến thể robust hơn của raw time-bin residual. Thay vì tin hoàn toàn vào residual của từng bin, nó trộn residual theo bin với global residual:

```text
smoothed_residual_bin =
w * bin_residual + (1 - w) * global_residual
```

```text
w = count_bin / (count_bin + prior_strength)
```

Nếu bin có ít sample, trọng số `w` nhỏ, residual sẽ bị kéo về global residual. Nếu bin có nhiều sample, residual theo bin được tin tưởng nhiều hơn.

Kết quả trên test set:

```text
MAE  = 38.073s
MAPE = 20.40%
RMSE = 52.159s
P50  = 31.447s
P95  = 103.111s
```

Điểm mạnh:

* Robust hơn raw time-bin về mặt phương pháp.
* Giảm rủi ro overfit ở các bin ít sample.
* Vẫn cải thiện P95 đáng kể so với API baseline.
* Phù hợp hơn nếu muốn triển khai production-safe.

Điểm yếu:

* Trên test set hiện tại, smoothed time-bin không thắng raw time-bin về P95.
* Không thắng global residual về MAE.
* MAPE vẫn xấu hơn API baseline.


--> Smoothing làm giảm độ cực đoan của residual theo bin. Điều này giúp ổn định hơn ở bin ít sample, nhưng cũng có thể làm mất một phần lợi ích của raw time-bin ở những bin mà residual thực sự có pattern rõ. Vì test set hiện tại nhỏ, raw time-bin có thể đang tận dụng tốt một số pattern cụ thể.

Nhận xét:

Smoothed time-bin chưa phải phương pháp tốt nhất trên test hiện tại, nhưng là hướng đáng giữ lại nếu mục tiêu là robustness và tránh overfitting khi dữ liệu sparse.

---

### 3.5 MLP Residual ETA

Phương pháp này sử dụng MLP để học residual:

```text
residual = actual_time - estimate_time
corrected_eta = estimate_time + predicted_residual
```

Kết quả trên test set:

```text
MAE  = 37.612s
MAPE = 19.43%
RMSE = 52.661s
P50  = 27.882s
P95  = 101.684s
```

So với API baseline:

```text
MAE improvement = +0.91%
P95 improvement = +17.83%
```

Điểm mạnh:

* Là ML residual model tốt nhất hiện tại.
* Cải thiện MAE so với API baseline.
* Giảm P95 đáng kể.
* Cân bằng tương đối tốt giữa MAE và tail-error.

Điểm yếu:

* Không thắng API + Global Median Residual về MAE.
* Không thắng raw time-bin / hour-bin median về P95.
* Phức tạp hơn các baseline thống kê.
* Với dữ liệu nhỏ và feature hạn chế, lợi thế của MLP chưa rõ.

--> MLP cần đủ dữ liệu và feature để học non-linear pattern. Trong dataset hiện tại, feature chính vẫn xoay quanh estimate time và time information. Khi signal còn hạn chế, MLP dễ chỉ học được pattern tương tự simple residual correction, nhưng lại có nhiều tham số hơn, dễ overfit hoặc không ổn định.

Nhận xét:

MLP residual là hướng promising, nhưng hiện tại chưa đủ tốt để thay thế các correction baseline đơn giản.

---

### 3.7 Hour-bin MLP Residual ETA

Kết quả trên test set:

```text
MAE  = 38.297s
MAPE = 19.41%
RMSE = 54.500s
P50  = 29.316s
P95  = 107.584s
```

Điểm mạnh:

* MAPE tương đối tốt trong nhóm ML correction.
* Vẫn cải thiện P95 so với API baseline.

Điểm yếu:

* MAE kém hơn API baseline.
* RMSE và P95 kém hơn MLP residual không dùng hour-bin.
* Không có lợi thế rõ so với simple correction.

--> Việc thêm hour-bin vào MLP không đảm bảo cải thiện nếu dữ liệu trong một số bin còn sparse. Model có thể học các pattern không ổn định từ các bin ít sample.

Nhận xét:

Hiện tại chưa nên chọn phương pháp này.

---

### 3.8 DeeprETA-like

```text
Data schema:

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

Kết quả trên test set:

```text
MAE  = 38.453s
MAPE = 20.95%
RMSE = 52.213s
P50  = 32.310s
P95  = 99.546s
```

Điểm mạnh:

* P95 tốt, gần với raw time-bin residual.
* RMSE tương đối thấp.

Điểm yếu:

* MAE kém hơn API baseline.
* MAPE cao.
* Inference time cao hơn các model đơn giản.
* Kiến trúc phức tạp hơn nhiều so với lợi ích đạt được.

--> Các mô hình deep ETA thường cần nhiều dữ liệu hơn và feature phong phú hơn như route sequence, road segment, polyline, traffic condition, temporal context, weather hoặc multi-route data. Với dữ liệu hiện tại chỉ tập trung vào một route cố định và feature còn hạn chế, deep-like architecture chưa có đủ thông tin để phát huy lợi thế.

Nhận xét:

DeeprETA-like có thể là hướng nghiên cứu tương lai, nhưng chưa phù hợp làm model chính trong giai đoạn hiện tại.

---

### 3.9 XGBoost-based Methods

Các kết quả XGBoost hiện tại:

```text
xgb_residual_eta:
MAE = 42.639s
P95 = 106.104s

hour_bin_xgb_residual_eta:
MAE = 43.383s
P95 = 113.415s

xgb_direct_eta:
MAE = 44.914s
P95 = 99.304s
```

Điểm mạnh:

* Một số biến thể XGB vẫn giảm P95 so với API baseline.
* XGB direct có P95 khá tốt.

Điểm yếu:

* MAE tệ hơn API baseline khá rõ.
* MAPE cao nhất trong các phương pháp.
* Không phù hợp với dữ liệu nhỏ và feature hạn chế hiện tại.
* Có khả năng overfit hoặc học noise trong residual.

--> XGBoost thường mạnh khi có feature đa dạng và đủ dữ liệu. Trong bài toán hiện tại, feature chưa đủ phong phú để tree-based boosting học được pattern ổn định. Residual có thể chứa nhiều noise không giải thích được bởi feature hiện có, khiến XGB học sai pattern.

Nhận xét:

Không nên ưu tiên XGBoost trong giai đoạn hiện tại.

---

## 4. Overall Comparison

```text
Best MAE:
API + Global Median Residual

Best RMSE:
API + Raw Time-bin Median Residual

Best P95:
API + Raw Time-bin Median Residual / Hour-bin Median ETA

Best MAPE:
API baseline

Best ML residual model:
MLP Residual ETA

Most interpretable:
API + Global Median Residual

Most useful for tail-error reduction:
API + Raw Time-bin Median Residual
```

Kết quả này cho thấy các phương pháp correction đơn giản đang hiệu quả hơn các model phức tạp trong điều kiện dữ liệu hiện tại.

---

Các model phức tạp như XGBoost, MLP, DeeprETA-like chưa vượt trội vì một số lý do:

1. Dữ liệu còn nhỏ
   Test set chỉ có khoảng vài chục sample, trong khi deep model hoặc boosting model cần nhiều dữ liệu hơn để học pattern ổn định.

2. Feature còn hạn chế
   Hiện tại chưa có nhiều auxiliary features như weather, traffic condition, holiday, day of week, bus frequency, road segment hoặc route polyline.

3. Bài toán chỉ tập trung vào một route cố định
   Với một route cố định, các statistical baselines như global residual hoặc time-bin residual đã rất mạnh. Model phức tạp không có nhiều không gian để học thêm.

4. Residual có nhiều noise
   Residual giữa actual time và API estimate có thể bị ảnh hưởng bởi nhiều yếu tố chưa quan sát được như chờ đèn đỏ, dừng đón khách, congestion bất thường, thời tiết hoặc incident.

5. Một số time-bin có ít sample
   Các khung giờ ít xe buýt chạy như late evening có rất ít sample. Nếu model học quá mạnh theo các bin này, có thể overfit.

---

## 6. Selected Method

Primary method:
API + Raw Time-bin Median Residual

Lý do:

Có RMSE tốt nhất và P95 tốt nhất trong các phương pháp đã thử.
Giảm mạnh các lỗi lớn so với API baseline.
Khai thác được sự khác biệt về residual giữa các khung giờ, phù hợp với đặc tính vận hành thực tế của tuyến xe buýt.
Cho thấy ETA từ API không có cùng mức độ sai lệch ở mọi thời điểm trong ngày, do đó correction theo time-bin mang lại nhiều thông tin hơn so với một residual cố định.
Với dữ liệu hiện tại, phương pháp này đạt được sự cân bằng tốt giữa hiệu quả và tính đơn giản.

Tuy nhiên, nên giữ lại:

Secondary method:
API + Global Median Residual


* Có MAE tốt nhất trên test set.
* Rất đơn giản, dễ giải thích và dễ triển khai.
* Hoạt động ổn định ngay cả khi dữ liệu trong từng time-bin còn hạn chế.
* Là baseline correction mạnh và đáng tin cậy để so sánh với các phương pháp khác.

Với smoothed time-bin residual -> Robust future candidate:

* Giữ được ý tưởng correction theo thời gian trong khi giảm độ nhạy với các bin có ít dữ liệu.
* Có tiềm năng tổng quát hóa tốt hơn khi mở rộng sang các giai đoạn dữ liệu khác.
Cần tune thêm smoothing strength và đánh giá trên nhiều time split để xác nhận lợi ích thực tế.

---


Kết quả hiện tại cho thấy trong các phương pháp đã thử, **API + Raw-time bin Residual** là phương pháp tuy không cải thiện MAE nhưng giảm P95 mạnh nhất, cho thấy correction theo khung giờ có tiềm năng giảm lỗi lớn. Các model phức tạp như MLP, XGBoost và DeeprETA-like chưa vượt được các baseline đơn giản do dữ liệu nhỏ và feature còn hạn chế.
