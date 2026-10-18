# StudyStress Insights
Clustering-based student workload and well-being advisor (K-Means + Streamlit).

**Live app:** https://studystress-insights-6ngab2uukv88nxpapdwex3.streamlit.app/

## What it doeshttps://studystress-insights-6ngab2uukv88nxpapdwex3.streamlit.app/
Groups students by sleep, bedtime, screen time, caffeine, exercise, study hours and attendance into 4 lifestyle groups, shows how each group reports energy, sleep quality and concentration, and suggests workload-management habits.

## Tech
Python, pandas, scikit-learn (K-Means), Streamlit, Plotly. Notebook in Google Colab.

## Files
- `app.py`: Streamlit dashboard
- `recommendations.py`: plans for each group
- `models/`: saved scaler and K-Means model
- `data/`: raw and processed data
- `report/`: charts from the analysis

## Dataset
Student Sleep, Screen Time & Academic Performance (Kaggle, synthetic, CC0).

## Ethics
Informational only. Not a medical or mental-health diagnosis.
