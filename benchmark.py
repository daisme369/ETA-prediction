import time
import requests
import concurrent.futures
import statistics
import argparse

def send_request(url, payload):
    start = time.perf_counter()
    error_msg = None
    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        success = True
    except Exception as e:
        success = False
        error_msg = type(e).__name__
        if isinstance(e, requests.exceptions.HTTPError):
            error_msg += f" (Status {e.response.status_code})"
        elif hasattr(e, "args") and e.args:
            error_msg += f" ({e.args[0]})"
    end = time.perf_counter()
    return (end - start) * 1000, success, error_msg  # return ms, success, error_msg

def main():
    parser = argparse.ArgumentParser(description="Load Test ETA Prediction API")
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/eta/predict", help="Target API URL")
    parser.add_argument("--requests", type=int, default=100, help="Total number of requests to send")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent workers")
    parser.add_argument("--model", default="ratio_time_bin", help="Model ID to test")
    args = parser.parse_args()

    payload = {
        "departure_time": "2024-05-15T08:00:00",
        "model_id": args.model,
        "baseline_eta_secs": 1500
    }

    print(f"Bắt đầu test API: {args.url}")
    print(f"Tổng số request: {args.requests} | Concurrency: {args.concurrency}")
    print(f"Payload: {payload}\n")

    latencies = []
    success_count = 0
    errors = {}

    start_total = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(send_request, args.url, payload) for _ in range(args.requests)]
        for future in concurrent.futures.as_completed(futures):
            latency, success, error_msg = future.result()
            latencies.append(latency)
            if success:
                success_count += 1
            else:
                errors[error_msg] = errors.get(error_msg, 0) + 1

    end_total = time.perf_counter()
    total_time = end_total - start_total

    print("-" * 40)
    print("KẾT QUẢ BENCHMARK")
    print("-" * 40)
    print(f"Tổng thời gian test: {total_time:.2f} giây")
    print(f"Requests thành công: {success_count}/{args.requests}")
    
    if errors:
        print("\n[Chi tiết lỗi - Error Details]")
        for err, count in errors.items():
            print(f"  - {err}: {count} lần")

    if latencies:
        print(f"Requests per second (RPS): {args.requests / total_time:.2f} req/s")
        print("\n[Độ trễ - Latency (ms)]")
        print(f"  Trung bình (Mean): {statistics.mean(latencies):.2f} ms")
        print(f"  Trung vị (P50):    {statistics.median(latencies):.2f} ms")
        
        # P90, P95, P99
        latencies_sorted = sorted(latencies)
        p90 = latencies_sorted[int(len(latencies) * 0.90)]
        p95 = latencies_sorted[int(len(latencies) * 0.95)]
        p99 = latencies_sorted[int(len(latencies) * 0.99)]
        
        print(f"  Phân vị 90 (P90):  {p90:.2f} ms")
        print(f"  Phân vị 95 (P95):  {p95:.2f} ms")
        print(f"  Phân vị 99 (P99):  {p99:.2f} ms")
        print(f"  Chậm nhất (Max):   {max(latencies):.2f} ms")
        print(f"  Nhanh nhất (Min):  {min(latencies):.2f} ms")
    else:
        print("Không có request nào được thực hiện thành công.")

if __name__ == "__main__":
    main()
