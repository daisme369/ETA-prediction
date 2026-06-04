# Vietmap Route Frontend

Minimal frontend and backend scaffold for a Vietmap-powered route viewer where users enter full addresses and the browser never sees the Vietmap API key.

## What it does

- serves a map UI from `public/`
- resolves origin and destination addresses into `lat/lng` through the backend
- exposes `POST /api/route` as a backend proxy to Vietmap Route v3
- exposes `POST /api/resolve-location` for address-to-coordinate extraction
- reads Vietmap settings from `.env`
- optionally proxies map tiles if you provide a separate tile API key and tile URL template

## Setup

1. Copy `.env.example` to `.env`
2. Fill in `VIETMAP_API_KEY`
3. Optionally fill in `VIETMAP_TILE_API_KEY` and `VIETMAP_TILE_URL_TEMPLATE`
4. Start the app:

```bash
npm start
```

Open `http://localhost:3000`.

Start the ETA model API in a second terminal when you need model predictions:

```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

## Suggested env values

### Route API

```env
VIETMAP_API_KEY=your_service_api_key
VIETMAP_ROUTE_URL=https://maps.vietmap.vn/api/route/v3
VIETMAP_SEARCH_URL=https://maps.vietmap.vn/api/search/v4
VIETMAP_PLACE_URL=https://maps.vietmap.vn/api/place/v4
VIETMAP_DISPLAY_TYPE=5
```

### Optional tile proxy

Vietmap documents a dedicated tile key separate from the service key. If you want the map tiles to go through your backend too, add:

```env
VIETMAP_TILE_API_KEY=your_tile_api_key
VIETMAP_TILE_URL_TEMPLATE=https://maps.vietmap.vn/maps/tiles/st/{z}/{x}/{y}.png
```

If these are not set, the frontend falls back to the public OpenStreetMap tile layer.

## API shape

### `POST /api/resolve-location`

```json
{
  "address": "197 Tran Phu, Phuong 4, Quan 5, Thanh pho Ho Chi Minh"
}
```

### `POST /api/route`

```json
{
  "origin": {
    "lat": 10.776889,
    "lng": 106.700806,
    "address": "197 Tran Phu, Phuong 4, Quan 5, Thanh pho Ho Chi Minh"
  },
  "destination": {
    "lat": 10.802640,
    "lng": 106.714221,
    "address": "292 Dinh Bo Linh, Phuong 26, Quan Binh Thanh, Thanh pho Ho Chi Minh"
  },
  "vehicle": "car",
  "capacityKg": 2000,
  "departureTime": "2026-05-26T10:00:00Z",
  "alternative": false
}
```

## Notes

- Vietmap recommends backend integration to avoid exposing API credentials.
- The frontend resolves full addresses on blur or when the user presses `Extract coordinates`.
- Vietmap currently issues separate keys for tile display and service APIs.
- The app requests `points_encoded=false` so the frontend can render route geometry without a polyline decoder.

## Mock ETA residual data (Hanoi MVP)

Use the generator in `data/` to create realistic Hanoi origin/destination trips and mock ETA residuals.

```bash
npm run generate:mock
```

The generated rows include `request_timestamp` in `Asia/Ho_Chi_Minh` ISO-8601 format. Time features such as `hour_of_day`, `day_of_week`, `is_weekend`, and `is_rush_hour` are derived from that timestamp.

Useful generator options:

```bash
node data/generate_mock_eta.js --start-date 2026-05-01 --date-range-days 30 --format both
```

The XGBoost training pipeline supports either random or chronological splitting:

```bash
python model/model_baseline.py --split-strategy time
```

To fill `data/output_log.csv`, keep both APIs running and execute:

```bash
node data/fill_output_log.js
```

The script calls the Vietmap proxy with each row's `lat/lng`, `destination_lat/destination_lng`, and timestamp-aligned `departureTime`. If `data/output_log.csv` does not include `timestamp`, the script hydrates it from `data/processed_data.csv` and corrects `hour` to match that timestamp. It then calls the model API with the same `departure_time` and writes both `estimate_time` and `predict_time` in seconds.

For a fresh Python environment:

```bash
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python model\model_baseline.py --split-strategy time --n-trials 16 --cv-folds 3
.\.venv\Scripts\mlflow ui --backend-store-uri .\mlruns
```

The MLflow run logs dataset metrics, split metadata, nested tuning trials, trace spans, best hyperparameters, model metrics, feature importance, and the serialized sklearn/XGBoost pipeline.
