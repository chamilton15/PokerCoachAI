# Poker Coach AI Web Application

Simple React frontend with Flask backend for analyzing poker sessions using the trained LSTM model.

## Setup

### Backend (Flask API)

```bash
cd web_app/backend
pip install flask flask-cors
python app.py
```

Backend runs on `http://localhost:5001`

**Note:** Make sure you have the trained model at `ml_pipeline/models/poker_coach_model.pt`

### Frontend (React)

```bash
cd web_app/frontend
npm install
npm start
```

Frontend runs on `http://localhost:3000` and automatically proxies API requests to the backend.

## Testing

### Test the API directly:

```bash
# Start the backend first, then:
cd web_app
python test_api.py
```

### Test the full stack:

1. Start backend: `cd web_app/backend && python app.py`
2. Start frontend: `cd web_app/frontend && npm start`
3. Open browser to `http://localhost:3000`
4. Enter a player name (e.g., "Qt5Yyd/Y121jtIk37c7TSg")
5. Select a sample file or upload your own .phh/.phhs file
6. Click "Analyze Session"

## Features

- Upload .phh/.phhs files or use preloaded samples
- Enter player name to analyze
- View analysis results including:
  - Agreement rate with optimal play
  - Total hands and decisions analyzed
  - High confidence mistakes count
  - Top mistake patterns
  - Sample decision comparisons table

## Preloaded Sample Files

The app includes 3 preloaded sample files from the Poker_Data_Set for easy testing:
- Sample 1 - 50NLH (ABS-2009-07-01_2009-07-23_50NLH_OBFU)
- Sample 2 - 100NLH (ABS-2009-07-01_2009-07-23_100NLH_OBFU)
- Sample 3 - 200NLH (ABS-2009-07-01_2009-07-23_200NLH_OBFU)

## API Endpoints

- `GET /api/health` - Health check
- `GET /api/sample-files` - Get list of preloaded sample files
- `POST /api/analyze` - Analyze a poker session
  ```json
  {
    "player_name": "PlayerName",
    "use_sample": true,
    "sample_index": 0
  }
  ```

## Troubleshooting

- **Model not found**: Make sure `ml_pipeline/models/poker_coach_model.pt` exists
- **No decisions found**: The player name must match exactly (case-sensitive) or be a substring of the player ID in the file
- **Import errors**: Make sure you're running from the correct directory and all dependencies are installed
