# crime-prediction
Prophet-based crime prediction using Vancouver crime data


# Crime Rate Forecasting using Facebook Prophet

This project aims to forecast monthly crime rates in various neighborhoods using historical Vancouver crime data. The workflow includes spatial clustering using DBSCAN, time-series modeling using Facebook Prophet, and rich visualizations for trend analysis and forecasting.

---

## Project Structure

| File/Folder                | Description |
|---------------------------|-------------|
| `process_data.py`         | Reads raw crime data (`Train.csv`), applies DBSCAN clustering on location data, and summarizes crime counts into a processed CSV file. |
| `crime_predictor.py`      | Trains Prophet models on the processed data, forecasts future crime rates by type and neighbourhood, and displays interactive visualizations. |
| `dataset_modified/`       | Contains a preprocessed version of the dataset (`crime_rate_summary.csv`) for users who prefer not to run the full preprocessing pipeline. |
| `requirements.txt`        | Lists Python dependencies required to run the project. |
| `README.md`               | This documentation file. |

---

## ⚙️ How It Works

1. **Data Preprocessing**
   - Raw data is clustered spatially using DBSCAN.
   - Crime counts are grouped by location, type, and time to generate a summary dataset.

2. **Time Series Forecasting**
   - Prophet models are trained for each neighborhood and crime type.
   - Forecasts are made for a given month and year.

3. **Visualization**
   - Generates line charts, pie charts, heatmaps, trend comparisons, and uncertainty intervals.

---

## Quick Start

### 🔹 Option 1: Full Pipeline (requires more RAM)

If your system can handle large data processing:

```bash
python process_data.py
