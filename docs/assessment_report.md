# Metrics Collecting Report Table

Dưới đây là bảng tổng hợp metrics đánh giá của tất cả các mô hình và phương pháp hiệu chỉnh trên tập Test, được sắp xếp theo MAE (từ tốt nhất đến kém nhất).

| Model / Method | MAE (s) ↓ | RMSE (s) | MAPE (%) | P50 (s) | P95 (s) | MAE Imprv (%) | P95 Imprv (%) |
|---|---|---|---|---|---|---|---|
| **ratio_global** | 37.24 | 52.53 | 19.60 | 30.31 | 108.04 | 1.89% | 12.70% |
| **log_ratio_global** | 37.24 | 52.53 | 19.60 | 30.31 | 108.04 | 1.89% | 12.70% |
| **additive_global** | 37.25 | 52.80 | 19.58 | 30.10 | 107.95 | 1.86% | 12.76% |
| **mlp_residual_eta** | 37.61 | 52.66 | 19.43 | 27.88 | 101.68 | 0.91% | 17.83% |
| **Vietmap Baseline** | 37.96 | 56.89 | 18.34 | 23.08 | 123.75 | 0.00% | 0.00% |
| **additive_time_bin** | 37.97 | 51.79 | 20.45 | 32.08 | 98.94 | -0.02% | 20.05% |
| **ratio_time_bin** | 38.02 | 51.20 | 20.69 | 32.33 | 97.28 | -0.16% | 21.39% |
| **log_ratio_time_bin** | 38.02 | 51.20 | 20.69 | 32.33 | 97.28 | -0.16% | 21.39% |
| **ratio_smoothed_time_bin** | 38.02 | 51.20 | 20.69 | 32.33 | 97.28 | -0.16% | 21.39% |
| **log_ratio_smoothed_time_bin**| 38.02 | 51.20 | 20.69 | 32.33 | 97.28 | -0.16% | 21.39% |
| **hour_bin_median_eta** | 38.04 | 52.01 | 20.45 | 30.41 | 98.94 | -0.21% | 20.05% |
| **additive_smoothed_time_bin** | 38.07 | 52.16 | 20.40 | 30.87 | 103.11 | -0.31% | 16.68% |
| **hour_bin_mlp_residual_eta** | 38.30 | 54.50 | 19.41 | 29.32 | 107.58 | -0.90% | 13.06% |
| **deepr_eta_like** | 38.45 | 52.21 | 20.95 | 32.31 | 99.55 | -1.31% | 19.56% |
| **affine_global** | 39.81 | 56.78 | 21.31 | 32.57 | 111.77 | -4.89% | 9.68% |
| **affine_smoothed_time_bin** | 40.40 | 56.44 | 21.97 | 34.65 | 107.76 | -6.45% | 12.92% |
| **affine_time_bin** | 41.76 | 56.61 | 23.00 | 31.13 | 100.86 | -10.02% | 18.50% |
| **hour_bin_deepr_eta_like** | 41.96 | 57.56 | 23.72 | 32.44 | 101.36 | -10.54% | 18.09% |
| **xgb_residual_eta** | 42.64 | 56.55 | 24.66 | 35.29 | 106.10 | -12.34% | 14.26% |
| **hour_bin_xgb_residual_eta** | 43.38 | 56.46 | 25.25 | 37.58 | 113.42 | -14.30% | 8.35% |
| **xgb_direct_eta** | 44.91 | 58.54 | 25.92 | 41.99 | 99.30 | -18.33% | 19.75% |


# Metrics Explanation 

- **MAE (Mean Absolute Error)**: Sai số tuyệt đối trung bình (đơn vị: giây). Là trung bình của giá trị tuyệt đối hiệu số giữa ETA thực tế và ETA dự đoán. MAE càng nhỏ nghĩa là dự đoán tổng thể càng sát với thực tế.
- **RMSE (Root Mean Squared Error)**: Căn bậc hai của sai số bình phương trung bình. Khác với MAE, RMSE bình phương sai số trước khi lấy trung bình, nên nó phạt rất nặng các trường hợp dự đoán sai lệch lớn (outliers).
- **MAPE (Mean Absolute Percentage Error)**: Sai số phần trăm tuyệt đối trung bình. Cho biết sai lệch trung bình chiếm bao nhiêu % so với thời gian ETA thực tế của cuốc xe.
- **P50 (Median Absolute Error)**: Phân vị thứ 50 (trung vị) của sai số tuyệt đối. Có nghĩa là 50% số lượng chuyến đi có sai số nhỏ hơn hoặc bằng mức này. Ít bị ảnh hưởng bởi outliers hơn so với MAE.
- **P95 (95th Percentile Absolute Error)**: Phân vị thứ 95 của sai số tuyệt đối. Có nghĩa là 95% số chuyến đi có sai số nhỏ hơn hoặc bằng mức này. Đây là metric cực kỳ quan trọng để đánh giá worst-case performance (đuôi của phân phối sai số), ảnh hưởng trực tiếp đến trải nghiệm của những khách hàng phải chờ lâu nhất.
- **MAE Imprv (%) (MAE Improvement vs API)**: Tỷ lệ phần trăm cải thiện MAE của mô hình so với Baseline (Vietmap API). Giá trị dương thể hiện mô hình đã làm giảm sai số MAE tốt hơn so với Vietmap, trong khi giá trị âm thể hiện dự đoán bị kém đi xét trên trung bình.
- **P95 Imprv (%) (P95 Improvement vs API)**: Tỷ lệ phần trăm cải thiện P95 của mô hình so với Baseline. Đặc biệt quan trọng vì dù một số mô hình có MAE không cải thiện (âm) nhưng P95 Imprv lại dương lớn (như các biến thể time_bin), nghĩa là chúng giúp giảm bớt được sai số của những ca tệ nhất so với Vietmap.
