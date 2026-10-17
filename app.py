import json
import math

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

from recommendations import PLANS, DESCRIPTIONS, SUPPORT_MESSAGE

st.set_page_config(page_title="StudyStress Insights", page_icon="📚", layout="wide")

LABELS = {
    "avg_sleep_hours_weekday": "Sleep on weekdays (hours/night)",
    "bedtime_adj": "Usual bedtime",
    "daily_screen_time_hours": "Daily screen time (hours)",
    "pre_sleep_screen_time_min": "Screen time before bed (minutes)",
    "caffeine_drinks_per_day": "Caffeinated drinks per day",
    "exercise_days_per_week": "Exercise days per week",
    "weekly_study_hours": "Study hours per week",
    "class_attendance_percent": "Class attendance (%)",
}
STEPS = {
    "avg_sleep_hours_weekday": 0.5, "daily_screen_time_hours": 0.5,
    "pre_sleep_screen_time_min": 5, "caffeine_drinks_per_day": 1,
    "exercise_days_per_week": 1, "weekly_study_hours": 1,
    "class_attendance_percent": 1,
}
OUTCOMES = {
    "energy_level": "Energy level (1-10)",
    "sleep_quality_score": "Sleep quality (1-10)",
    "concentration_difficulty": "Concentration difficulty (1-10)",
    "cgpa": "CGPA",
}


def clock(b):
    """Turn 25.5 into '1:30 AM'."""
    total = int(round((b % 24) * 60)) % (24 * 60)
    hh, mm = divmod(total, 60)
    suffix = "AM" if hh < 12 else "PM"
    return f"{(hh % 12) or 12}:{mm:02d} {suffix}"


@st.cache_resource
def load_model():
    scaler = joblib.load("models/scaler.pkl")
    km = joblib.load("models/kmeans_model.pkl")
    with open("models/config.json") as f:
        cfg = json.load(f)
    cfg["cluster_names"] = {int(k): v for k, v in cfg["cluster_names"].items()}
    return scaler, km, cfg


@st.cache_data
def load_data():
    return pd.read_csv("data/processed/clustered_students.csv")


@st.cache_data
def pca_points(_scaler, features, data):
    return PCA(n_components=2, random_state=42).fit_transform(
        _scaler.transform(data[list(features)]))


scaler, km, cfg = load_model()
data = load_data()
FEATURES, NAMES = cfg["features"], cfg["cluster_names"]

page = st.sidebar.radio("Go to", ["Home", "Data Insights", "Check My Group", "About & Ethics"])

# ------------------------------ HOME ------------------------------
if page == "Home":
    st.title("📚 StudyStress Insights")
    st.subheader("Clustering-Based Student Workload and Well-Being Advisor")
    st.write(
        "This app groups students by their sleep, screen time, caffeine, exercise, "
        "study and attendance habits, then shows how each group feels (energy, sleep "
        "quality, concentration) and suggests simple workload-management habits."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Students analysed", f"{len(data):,}")
    c2.metric("Lifestyle groups", len(NAMES))
    c3.metric("Habits used", len(FEATURES))
    st.info("Open **Check My Group** in the sidebar to try it. It is anonymous and "
            "informational only, not a diagnosis.")

# ------------------------- DATA INSIGHTS --------------------------
elif page == "Data Insights":
    st.title("Data Insights")
    t1, t2, t3, t4 = st.tabs(["Distributions", "Correlations", "Group map", "Group profiles"])

    with t1:
        col = st.selectbox("Habit", FEATURES, format_func=lambda c: LABELS[c])
        st.plotly_chart(px.histogram(data, x=col, color="cluster_name",
                                     nbins=25, barmode="overlay", opacity=0.6))
        if col == "bedtime_adj":
            st.caption("Bedtime is shown as hours past noon: 24 = midnight, 25 = 1 AM.")

    with t2:
        corr_cols = FEATURES + list(OUTCOMES.keys())
        st.plotly_chart(px.imshow(data[corr_cols].corr().round(2), text_auto=True,
                                  color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                  height=650))

    with t3:
        pcs = pca_points(scaler, tuple(FEATURES), data)
        plot_df = pd.DataFrame({"Component 1": pcs[:, 0], "Component 2": pcs[:, 1],
                                "Group": data["cluster_name"]})
        st.plotly_chart(px.scatter(plot_df, x="Component 1", y="Component 2",
                                   color="Group", opacity=0.7))
        st.caption("Groups overlap because student habits vary smoothly. They describe "
                   "tendencies, not strict boxes.")
        counts = data["cluster_name"].value_counts().reset_index()
        counts.columns = ["Group", "Students"]
        st.plotly_chart(px.bar(counts, x="Group", y="Students"))

    with t4:
        prof = data.groupby("cluster_name")[FEATURES + list(OUTCOMES.keys())].mean().round(2)
        st.dataframe(prof.T)
        st.write("Chronotype mix in each group")
        st.dataframe(pd.crosstab(data["cluster_name"], data["chronotype"],
                                 normalize="index").round(2))

# ------------------------ CHECK MY GROUP --------------------------
elif page == "Check My Group":
    st.title("Check My Group")
    st.write("Enter your typical habits. Nothing is stored.")

    vals = {}
    cols = st.columns(2)
    for i, f in enumerate(FEATURES):
        box = cols[i % 2]
        lo, hi, med = float(data[f].min()), float(data[f].max()), float(data[f].median())
        if f == "bedtime_adj":
            options = [x / 4 for x in range(int(math.floor(lo * 4)), int(math.ceil(hi * 4)) + 1)]
            default = min(options, key=lambda v: abs(v - med))
            vals[f] = box.select_slider(LABELS[f], options=options, value=default,
                                        format_func=clock)
        else:
            step = STEPS[f]
            lo_s = math.floor(lo / step) * step
            hi_s = math.ceil(hi / step) * step
            med_s = min(max(round(med / step) * step, lo_s), hi_s)
            vals[f] = box.slider(LABELS[f], min_value=float(lo_s), max_value=float(hi_s),
                                 value=float(med_s), step=float(step))

    stress = st.slider("How stressed do you feel lately? (1 = calm, 10 = very stressed). "
                       "Not used for grouping.", 1, 10, 5)

    st.divider()
    x = pd.DataFrame([vals])[FEATURES]
    dist = km.transform(scaler.transform(x))[0]
    order = np.argsort(dist)
    cid = int(order[0])
    name = NAMES[cid]

    st.success(f"Closest group: **{name}**")
    st.write(DESCRIPTIONS.get(name, ""))

    m1, m2 = st.columns(2)
    m1.metric("Distance to your group's centre", f"{dist[cid]:.2f}")
    m2.metric(f"Next closest: {NAMES[int(order[1])]}", f"{dist[order[1]]:.2f}")
    if dist[order[1]] - dist[order[0]] < 0.3:
        st.info("You sit close to the boundary between two groups, so the other group's "
                "tips may fit you too.")

    # radar: you vs group average, scaled 0-1
    lo, hi = data[FEATURES].min(), data[FEATURES].max()
    norm = lambda s: ((s - lo) / (hi - lo)).clip(0, 1)
    you = norm(pd.Series(vals))
    avg = norm(data[data["cluster"] == cid][FEATURES].mean())
    theta = [LABELS[f] for f in FEATURES]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=list(you) + [you.iloc[0]], theta=theta + [theta[0]],
                                  fill="toself", name="You"))
    fig.add_trace(go.Scatterpolar(r=list(avg) + [avg.iloc[0]], theta=theta + [theta[0]],
                                  fill="toself", name="Group average"))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), height=550)
    st.plotly_chart(fig)

    st.subheader("How students in this group report feeling")
    grp = data[data["cluster"] == cid]
    ocols = st.columns(len(OUTCOMES))
    for oc, (k, lab) in zip(ocols, OUTCOMES.items()):
        diff = grp[k].mean() - data[k].mean()
        oc.metric(lab, f"{grp[k].mean():.2f}", f"{diff:+.2f} vs all students",
                  delta_color="inverse" if k == "concentration_difficulty" else "normal")

    st.subheader("Suggested workload plan")
    for tip in PLANS.get(name, ["Keep tracking your habits weekly."]):
        st.markdown(f"- {tip}")

    if stress >= 8 or vals["avg_sleep_hours_weekday"] < 5:
        st.warning(SUPPORT_MESSAGE)

# ------------------------ ABOUT & ETHICS --------------------------
else:
    st.title("About & Ethics")
    st.markdown("""
- Results are **informational lifestyle and workload guidance**, not a mental-health diagnosis or treatment.
- Nothing you enter is stored, and no names or identifying details are used.
- Groups come from patterns in a dataset (K-Means clustering). They overlap, so they describe tendencies, not fixed categories.
- The dataset is **synthetic** (computer-generated), so findings illustrate the method and are not conclusions about real students.
- Energy, sleep quality and concentration are self-reported scores in the dataset, not clinical measures.
- If you feel overwhelmed, please talk to a counsellor, a trusted adult, or your college's wellness service.
""")
