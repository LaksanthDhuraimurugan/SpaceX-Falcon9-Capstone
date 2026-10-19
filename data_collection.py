"""Collect SpaceX data from API where reachable; fall back to public course data."""
from pathlib import Path
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
API_URL = "https://api.spacexdata.com/v4/launches/past"
# Public IBM Skills Network course data; fallback for blocked/unavailable API.
FALLBACK_URLS = [
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv",
    "https://raw.githubusercontent.com/Abhigyan-RA/SpaceX-capstone/main/dataset_part_2.csv",
    "https://raw.githubusercontent.com/DanielEduardoLopez/IBM-SpaceX/main/dataset_part_2_DEL.csv",
]
session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 Falcon9-Capstone-Student-Project"})

def get_json(url):
    response = session.get(url, timeout=20)
    response.raise_for_status()
    return response.json()

def main():
    log = []
    launches = None
    try:
        payload = get_json(API_URL)
        if not isinstance(payload, list) or not payload:
            raise ValueError("API response was not a non-empty list")
        launches = pd.json_normalize(payload)
        launches.to_csv(DATA / "spacex_api_raw.csv", index=False)
        log.append(f"API SUCCESS: {API_URL}; rows={len(launches)}")
        print(f"API connected: saved {len(launches)} launch records.")
    except Exception as exc:
        log.append(f"API FAILED: {API_URL}; reason={type(exc).__name__}: {exc}")
        print("SpaceX API unavailable; trying public course CSV fallback...")

    if launches is None:
        last_error = None
        for url in FALLBACK_URLS:
            try:
                response = session.get(url, timeout=30)
                response.raise_for_status()
                from io import BytesIO
                df = pd.read_csv(BytesIO(response.content))
                if df.empty:
                    raise ValueError("Downloaded CSV is empty")
                df.to_csv(DATA / "launches_fallback.csv", index=False)
                log.append(f"CSV FALLBACK SUCCESS: {url}; rows={len(df)}")
                print(f"Fallback connected: saved {len(df)} rows from {url}")
                launches = df
                break
            except Exception as exc:
                last_error = exc
                log.append(f"CSV FALLBACK FAILED: {url}; reason={type(exc).__name__}: {exc}")
        if launches is None:
            raise RuntimeError(
                "Could not reach API or fallback CSVs. Download dataset_part_2.csv from "
                "https://github.com/Abhigyan-RA/SpaceX-capstone and place it in data/ "
                f"as launches_fallback.csv. Last error: {last_error}"
            )

    # Also collect Wikipedia HTML/table as a separate, optional scraping artifact.
    wiki_url = "https://en.wikipedia.org/wiki/List_of_Falcon_9_and_Falcon_Heavy_launches"
    try:
        response = session.get(wiki_url, timeout=25)
        response.raise_for_status()
        (DATA / "wikipedia_launches.html").write_text(response.text, encoding="utf-8")
        tables = pd.read_html(response.text)
        if tables:
            tables[0].to_csv(DATA / "wikipedia_table_0.csv", index=False)
            log.append(f"WIKIPEDIA SUCCESS: {wiki_url}; tables={len(tables)}")
            print(f"Wikipedia scrape saved ({len(tables)} tables detected).")
    except Exception as exc:
        log.append(f"WIKIPEDIA FAILED: {wiki_url}; reason={type(exc).__name__}: {exc}")
        print(f"Wikipedia scraping skipped (network/site may block requests): {exc}")

    (DATA / "source_log.txt").write_text("\\n".join(log), encoding="utf-8")
    print("Source log:", DATA / "source_log.txt")

if __name__ == "__main__":
    main()
