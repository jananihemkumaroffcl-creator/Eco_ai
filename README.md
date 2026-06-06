# 🌱 Eco-AI Carbon Intelligence Dashboard

Tracks and compares CO₂ emissions between a baseline and an optimized (Eco-AI) PyTorch model, with a live Dash dashboard.

## Project Structure
```
Eco_ai/
├── src/
│   ├── baseline_train.py   # Unoptimized model (high carbon)
│   ├── eco_train.py        # Optimized model (low carbon)
│   └── dashboard.py        # Dash dashboard
├── assets/
│   └── dashboard.css
├── data/                   # Auto-generated CSVs (gitignored)
├── models/                 # Saved model weights (gitignored)
├── run_all.py              # Run full pipeline
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
python run_all.py
```
This trains both models and launches the dashboard at http://127.0.0.1:8050
