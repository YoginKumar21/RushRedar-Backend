# TrafficFlow — Backend (API & Models)

Lightweight Flask API that loads short-term (LSTM) and long-term (ensemble) traffic prediction models and serves configuration & prediction endpoints consumed by the frontend dashboard.

---

## Highlights
- Loads models from backend/models (lstm + ensemble)
- Exposes endpoints used by frontend:
  - GET /api/config — returns Mapbox API key (MY_API_KEY)
  - GET /api/get-recent-data — returns recent short-term rows for prediction
  - POST /api/predict?horizon=... — run short-term LSTM prediction
  - GET /api/trends?start=YYYY-MM-DD&end=YYYY-MM-DD — long-term trends
- Uses .env (backend/.env) for secrets (Mapbox token)
- CSV data files live in backend root:
  - TrafficDataset.csv (short-term timeseries)
  - a.csv (daily historical for long-term trends)

---

## Repository layout (backend/)
- .env — environment variables (ignored by git)
- app.py — Flask app, model loading, API routes
- TrafficDataset.csv — short-term dataset (now inside backend/)
- a.csv — historical daily CSV for long-term model
- models/
  - lstm/ — LSTM model files (.h5, metadata)
  - enesmble/ or ensemble/ — ensemble model folder (app.py will try both names)
- app/ — package containing utils (predictor.py, longpredictor.py), API helpers, assets
- requirements.txt — Python dependencies (moved into backend/)

---

## Quickstart (Windows)

1. Create & activate virtual environment
   ```powershell
   cd C:\Users\ASUS\Desktop\trafficflow\backend
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

### Complete Quickstart & Usage

1. Create & activate virtual environment (Windows)
```powershell
cd C:\Users\ASUS\Desktop\trafficflow\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Create a backend/.env file (example)
```
FLASK_ENV=development
FLASK_APP=app.py
SECRET_KEY=change_this_to_a_random_value
MAPBOX_TOKEN=your_mapbox_token_here
MODEL_DIR=models
PORT=5000
```
Do NOT commit .env to git. Replace MAPBOX_TOKEN with your Mapbox token.

3. Model placement
- backend/models/lstm/ — place LSTM model files (.h5, metadata)
- backend/models/ensemble/ — place ensemble model files (if folder named enesmble, app.py tries both)
Ensure filenames expected by your predictor modules (check app/ predictor.py and longpredictor.py).

4. Data files (backend root)
- TrafficDataset.csv — short-term timeseries used by LSTM pipeline
- a.csv — daily historical CSV used by long-term/ensemble pipeline

5. Run the API (development)
```powershell
# from backend/
set FLASK_ENV=development
flask run --host=0.0.0.0 --port=5000
# or
python app.py
```
For production, use a WSGI server (gunicorn) and ensure proper environment variables are set.

API Endpoints
- GET /api/config
    - Returns configuration (e.g., Mapbox token). Response example:
    ```json
    { "mapboxToken": "MAPBOX_TOKEN_PLACEHOLDER" }
    ```

- GET /api/get-recent-data
    - Returns recent short-term rows used for predictions (JSON array). Useful for frontend to display inputs.

- POST /api/predict?horizon=<n>
    - Runs short-term LSTM prediction for horizon (integer, e.g., 6 for next 6 timesteps).
    - Accepts JSON body with recent rows OR falls back to backend/TrafficDataset.csv if not provided.
    - Example curl:
    ```bash
    curl -X POST "http://localhost:5000/api/predict?horizon=6" \
        -H "Content-Type: application/json" \
        -d '{"data": [[timestamp1, val1, ...], ...] }'
    ```
    - Response: JSON containing predicted values and optionally confidence/metadata.

- GET /api/trends?start=YYYY-MM-DD&end=YYYY-MM-DD
    - Returns long-term trends computed by the ensemble model between the provided dates.
    - Example:
    ```
    GET /api/trends?start=2024-01-01&end=2024-03-31
    ```

Configuration notes
- The app tries to read .env. You can also export env vars in your shell.
- Set MODEL_DIR if your models are in a different location.
- If models are large, ensure your environment has enough memory.

Troubleshooting
- 500 on startup: check logs for missing model files or mismatched model paths. Verify models exist under backend/models and names match expected in app code.
- Missing MAPBOX token in frontend: ensure MAPBOX_TOKEN is set in .env and app restarted.
- Model version mismatch: retrain or export models consistent with predictor code (check shapes & preprocessing steps).

Development tips
- Inspect app/routes and app/ predictor modules to see exact expected input formats.
- Add logging to app.py for easier debugging: Python logging or print in development.
- Use Postman or curl to test endpoints before connecting frontend.

Testing
- Add unit tests for predictor functions (app/ predictor.py and longpredictor.py).
- Create small sample CSVs to run integration tests against endpoints.

License & Contributions
- Add your preferred LICENSE file in repository root.
- Document contribution guidelines (CONTRIBUTING.md) if others will work on the project.

