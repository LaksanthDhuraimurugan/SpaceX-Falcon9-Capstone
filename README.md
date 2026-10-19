# SpaceX Falcon 9 Capstone Project

This project follows the supplied Coursera capstone template: data collection, scraping, wrangling, EDA, SQL, Folium, Dash, and classification.

## Data-source note
The original community SpaceX API (`https://api.spacexdata.com/v4/launches/past`) may be unreachable from some networks and is maintenance-only. `data_collection.py` tries the API, then downloads the course's public CSV from IBM Skills Network. It saves the actual source used in `data/source_log.txt`. This fallback is a prepared course dataset, not a live API.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python data_collection.py
python analysis_pipeline.py
python dashboard.py
```
Open the Dash URL printed in the terminal (usually http://127.0.0.1:8050).

Run from this folder. Generated files go in `data/`, `outputs/`, and `maps/`

## Important interpretation
The historical course dataset has a limited number of launches and features. Model scores are calculated at runtime and can vary with dataset version and train/test split. Do not paste invented metrics into the slides.
