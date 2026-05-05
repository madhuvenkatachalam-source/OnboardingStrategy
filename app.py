import math
import datetime as dt
from typing import Optional

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter


# =========================================================
# PAGE + STYLING
# =========================================================
st.set_page_config(page_title="Onboarding Strategy Engine", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0A46A7, #2F80ED);
        color: white;
    }

    h1, h2, h3, h4, h5, h6, p, label {
        color: white !important;
    }

    .stButton > button {
        background-color: #0A46A7;
        color: white;
        border: 1px solid white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }

    .stDownloadButton > button {
        background-color: #0A46A7;
        color: white;
        border: 1px solid white;
        border-radius: 10px;
    }

    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.10);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="margin-bottom:20px;">
        <div style="font-size:18px; font-weight:700; opacity:0.95;">project44</div>
        <h1 style="margin:0;">Onboarding Strategy Engine</h1>
        <p style="margin:0; opacity:0.85;">
            Flexible onboarding planning with wave-based tiering and overlap visualization
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CONFIG
# =========================================================
COLUMN_ALIASES = {
    "carrier": ["carrier", "carrier name", "name of carrier"],
    "country": ["country base", "country", "region", "geo", "base country"],
    "volume": ["yearly volume", "annual volume", "annual spend", "volume"],
    "connectivity": [
        "p44 network",
        "p44 connectivity",
        "connectivity",
        "existing/new",
        "network",
    ],
    "method": ["comments", "connection method", "method", "integration method"],
    "mode": ["mode", "shipment mode", "transport mode"],
}

REGION_MAP = {
    "hungary": "EMEA",
    "poland": "EMEA",
    "netherland": "EMEA",
    "netherlands": "EMEA",
    "denmark": "EMEA",
    "france": "EMEA",
    "austria": "EMEA",
    "srb": "EMEA",
    "serbia": "EMEA",
    "czech": "EMEA",
    "czech republic": "EMEA",
    "sweden": "EMEA",
    "belgium": "EMEA",
}

MODE_DEFAULT_DURATIONS = {
    "FTL": {"Existing": 3, "New": 8},
    "LTL": {"Existing": 4, "New": 12},
    "PARCEL": {"Existing": 4, "New": 12},
    "OCEAN": {"Existing": 4, "New": 16},
    "AIR": {"Existing": 4, "New": 16},
    "DRAYAGE": {"Existing": 5, "New": 12},
}

DEFAULT_MODE = "FTL"

TYPE_BATCH_SIZE_BALANCED = {
    "ELD": 12,
    "APP": 12,
    "API": 4,
    "NEW": 6,
}

TYPE_BATCH_SIZE_AGGRESSIVE = {
    "ELD": 18,
    "APP": 18,
    "API": 6,
    "NEW": 8,
}


# =========================================================
# HELPERS
# =========================================================
def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def pct_str(x: float) -> str:
    return f"{x:.1%}"


def month_label(period_str: str) -> str:
    return pd.Period(period_str, freq="M").strftime("%b %Y")


def months_between(start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    days = max((end_date - start_date).days, 0)
    return round(days / 30.4, 1)


def add_weeks(date_value: dt.date, weeks: int) -> dt.date:
    return date_value + dt.timedelta(weeks=int(weeks))


def find_best_column(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    normalized = {col: normalize_text(col).lower() for col in df.columns}

    for alias in aliases:
        alias_lower = alias.lower()
        for original, norm in normalized.items():
            if norm == alias_lower:
                return original

    for alias in aliases:
        alias_lower = alias.lower()
        for original, norm in normalized.items():
            if alias_lower in norm:
                return original

    return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for target, aliases in COLUMN_ALIASES.items():
        source = find_best_column(df, aliases)
        if source:
            rename_map[source] = target
    return df.rename(columns=rename_map)


def clean_connectivity(value: object) -> str:
    text = normalize_text(value).lower()
    if "exist" in text or "live" in text or "current" in text:
        return "Existing"
    return "New"


def clean_method(value: object) -> str:
    text = normalize_text(value).upper()

    if "ELD" in text or "TELEMATICS" in text:
        return "ELD"
    if "API" in text or "DI" in text or "DIRECT" in text:
        return "API"
    if "DRIVEVIEW" in text or "MOBILE" in text or "APP" in text:
        return "APP"
    return ""


def infer_type(connectivity: str, method: str) -> str:
    if method == "ELD":
        return "ELD"
    if method == "API":
        return "API"
    if method == "APP":
        return "APP"
    if connectivity == "New":
        return "NEW"
    return "ELD"


def infer_mode(value: object) -> str:
    text = normalize_text(value).lower()
    if "ltl" in text:
        return "LTL"
    if "parcel" in text:
        return "PARCEL"
    if "ocean" in text:
        return "OCEAN"
    if "air" in text:
        return "AIR"
    if "drayage" in text:
        return "DRAYAGE"
    if "ftl" in text or "truck" in text:
        return "FTL"
    return DEFAULT_MODE


def map_region(country_value: object) -> str:
    text = normalize_text(country_value).lower()
    return REGION_MAP.get(text, normalize_text(country_value) or "Unknown")


def get_duration_weeks(row: pd.Series) -> int:
    mode = row["mode"]
    connectivity = row["connectivity"]
    method = row["method"]

    if mode == "FTL":
        if method == "ELD":
            return 1 if connectivity == "Existing" else 2
        if method == "API":
            return 8 if connectivity == "Existing" else 16
        if method == "APP":
            return 1

    return MODE_DEFAULT_DURATIONS.get(mode, MODE_DEFAULT_DURATIONS[DEFAULT_MODE])[connectivity]


def rank_within_groups(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["priority_score"] = result["volume"] / result["duration_weeks"]
    return result.sort_values(
        by=["priority_score", "volume"],
        ascending=[False, False]
    ).reset_index(drop=True)


def assign_wave_tiers(
    df: pd.DataFrame,
    small_list_no_tier_threshold: int = 25,
    medium_list_two_wave_threshold: int = 75,
    max_carriers_per_tier: int = 40,
) -> pd.DataFrame:
    result = rank_within_groups(df)
    total_carriers = len(result)

    if total_carriers <= small_list_no_tier_threshold:
        result["tier"] = "Tier 1"
        return result

    if total_carriers <= medium_list_two_wave_threshold:
        new_df = result[result["connectivity"] == "New"].copy()
        existing_df = result[result["connectivity"] == "Existing"].copy()

        tier1_target = min(max_carriers_per_tier, math.ceil(total_carriers * 0.5))
        tier1_new_target = min(len(new_df), max(1, math.ceil(tier1_target * 0.5)))
        tier1_existing_target = min(len(existing_df), tier1_target - tier1_new_target)

        tier1_new = new_df.head(tier1_new_target)
        tier1_existing = existing_df.head(tier1_existing_target)

        tier1_ids = set(tier1_new.index.tolist() + tier1_existing.index.tolist())
        result["tier"] = result.index.map(lambda i: "Tier 1" if i in tier1_ids else "Tier 2")

        current_tier1 = (result["tier"] == "Tier 1").sum()
        if current_tier1 < tier1_target:
            needed = tier1_target - current_tier1
            remaining_idx = result[result["tier"] != "Tier 1"].head(needed).index
            result.loc[remaining_idx, "tier"] = "Tier 1"

        return result.reset_index(drop=True)

    result["tier"] = ""
    unassigned = result.copy()
    tier_num = 1

    while not unassigned.empty:
        tier_label = f"Tier {tier_num}"
        tier_target = min(max_carriers_per_tier, len(unassigned))

        new_pool = unassigned[unassigned["connectivity"] == "New"]
        existing_pool = unassigned[unassigned["connectivity"] == "Existing"]

        tier_new_target = min(len(new_pool), max(1, math.ceil(tier_target * 0.4)))
        tier_existing_target = min(len(existing_pool), tier_target - tier_new_target)

        selected_idx = []
        selected_idx.extend(new_pool.head(tier_new_target).index.tolist())
        selected_idx.extend(existing_pool.head(tier_existing_target).index.tolist())

        if len(selected_idx) < tier_target:
            extra_needed = tier_target - len(selected_idx)
            extra = unassigned.drop(index=selected_idx).head(extra_needed).index.tolist()
            selected_idx.extend(extra)

        result.loc[selected_idx, "tier"] = tier_label
        unassigned = unassigned.drop(index=selected_idx)
        tier_num += 1

    return result.reset_index(drop=True)


def schedule_by_tier_and_type(
    df: pd.DataFrame,
    project_start: dt.date,
    tier_stagger_weeks: int,
    batch_sizes: dict[str, int],
) -> pd.DataFrame:
    scheduled_rows = []

    tier_order = sorted(
        df["tier"].dropna().unique(),
        key=lambda t: int(str(t).replace("Tier", "").strip())
    )

    tier_start_map = {
        tier: add_weeks(project_start, idx * tier_stagger_weeks)
        for idx, tier in enumerate(tier_order)
    }

    for tier in tier_order:
        tier_df = df[df["tier"] == tier].copy()

        for carrier_type in ["ELD", "APP", "API", "NEW"]:
            type_df = tier_df[tier_df["type"] == carrier_type].copy()
            if type_df.empty:
                continue

            type_df = type_df.sort_values(
                by=["priority_score", "volume"],
                ascending=[False, False]
            ).reset_index(drop=True)

            batch_size = batch_sizes.get(carrier_type, 4)
            base_start = tier_start_map[tier]

            starts = []
            ends = []
            batch_numbers = []

            for idx, (_, row) in enumerate(type_df.iterrows()):
                batch_number = idx // batch_size
                start_date = add_weeks(base_start, batch_number)
                end_date = add_weeks(start_date, int(row["duration_weeks"]))

                starts.append(pd.to_datetime(start_date))
                ends.append(pd.to_datetime(end_date))
                batch_numbers.append(batch_number + 1)

            type_df["start_date"] = starts
            type_df["end_date"] = ends
            type_df["batch"] = batch_numbers
            scheduled_rows.append(type_df)

    result = pd.concat(scheduled_rows, ignore_index=True)
    return result.sort_values(
        by=["start_date", "tier", "type", "priority_score"],
        ascending=[True, True, True, False]
    ).reset_index(drop=True)


def build_mix_string(sub_df: pd.DataFrame) -> str:
    counts = {
        "ELD": int((sub_df["type"] == "ELD").sum()),
        "APP": int((sub_df["type"] == "APP").sum()),
        "API": int((sub_df["type"] == "API").sum()),
        "NEW": int((sub_df["type"] == "NEW").sum()),
    }
    return (
        f"ELD:{counts['ELD']} | "
        f"APP:{counts['APP']} | "
        f"API:{counts['API']} | "
        f"NEW:{counts['NEW']}"
    )


def build_timeline_task_label(row: pd.Series) -> str:
    return f"{row['tier']} | {row['type']}"


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Planning Inputs")

selected_start_date = st.sidebar.date_input(
    "Onboarding start date",
    value=dt.date.today()
)

selected_style = st.sidebar.selectbox(
    "Onboarding style",
    ["Balanced", "Aggressive"],
    index=0
)

desired_duration = st.sidebar.selectbox(
    "Desired onboarding duration (optional)",
    ["Not specified", "6 months", "9 months", "1 year", "2 years"],
    index=0
)

small_list_no_tier_threshold = st.sidebar.number_input(
    "No tier split at or below carrier count",
    min_value=5,
    max_value=100,
    value=25,
    step=5
)

medium_list_two_wave_threshold = st.sidebar.number_input(
    "Use 2-wave logic up to carrier count",
    min_value=10,
    max_value=150,
    value=75,
    step=5
)

max_carriers_per_tier = st.sidebar.number_input(
    "Max carriers per tier",
    min_value=10,
    max_value=100,
    value=40,
    step=5
)

tier_stagger_weeks = st.sidebar.number_input(
    "Tier start lag (weeks)",
    min_value=0,
    max_value=12,
    value=4,
    step=1
)

if selected_style == "Aggressive":
    st.sidebar.warning("Aggressive is heavier on internal teams.")
    batch_sizes = TYPE_BATCH_SIZE_AGGRESSIVE.copy()
else:
    st.sidebar.success("Balanced is the recommended planning mode.")
    batch_sizes = TYPE_BATCH_SIZE_BALANCED.copy()

uploaded_file = st.file_uploader("Upload carrier file", type=["csv", "xlsx"])
run_model = st.button("Run Strategy")

if not run_model:
    st.info("Upload file and click 'Run Strategy' to generate the onboarding plan.")
    st.stop()

if uploaded_file is None:
    st.error("Please upload a CSV or Excel file.")
    st.stop()


# =========================================================
# READ FILE
# =========================================================
try:
    if uploaded_file.name.lower().endswith(".csv"):
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = pd.read_excel(uploaded_file)
except Exception as exc:
    st.error(f"Could not read file: {exc}")
    st.stop()

st.subheader("Raw Data Preview")
st.dataframe(raw_df, use_container_width=True)


# =========================================================
# CLEAN + STANDARDIZE
# =========================================================
df = standardize_columns(raw_df.copy())

required_fields = ["carrier", "country", "volume", "connectivity"]
missing = [field for field in required_fields if field not in df.columns]

if missing:
    st.error("Missing required fields after mapping: " + ", ".join(missing))
    st.write("Detected columns:", list(raw_df.columns))
    st.stop()

if "method" not in df.columns:
    df["method"] = ""

if "mode" not in df.columns:
    df["mode"] = DEFAULT_MODE

df["carrier"] = df["carrier"].astype(str).str.strip()
df["country"] = df["country"].apply(normalize_text)
df["region"] = df["country"].apply(map_region)
df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
df["connectivity"] = df["connectivity"].apply(clean_connectivity)
df["method"] = df["method"].apply(clean_method)
df["mode"] = df["mode"].apply(infer_mode)
df["type"] = df.apply(lambda row: infer_type(row["connectivity"], row["method"]), axis=1)

df = df[(df["carrier"] != "") & (df["volume"] > 0)].copy()

if df.empty:
    st.error("No usable carrier rows found after cleaning.")
    st.stop()

df["duration_weeks"] = df.apply(get_duration_weeks, axis=1)
df["priority_score"] = df["volume"] / df["duration_weeks"]


# =========================================================
# TIERING + SCHEDULING
# =========================================================
df = assign_wave_tiers(
    df=df,
    small_list_no_tier_threshold=int(small_list_no_tier_threshold),
    medium_list_two_wave_threshold=int(medium_list_two_wave_threshold),
    max_carriers_per_tier=int(max_carriers_per_tier),
)

df = schedule_by_tier_and_type(
    df=df,
    project_start=selected_start_date,
    tier_stagger_weeks=int(tier_stagger_weeks),
    batch_sizes=batch_sizes,
)

df["timeline_task"] = df.apply(build_timeline_task_label, axis=1)

total_carriers = len(df)
total_volume = float(df["volume"].sum())
project_start = df["start_date"].min()
project_end = df["end_date"].max()
estimated_months = months_between(project_start, project_end)


# =========================================================
# FEASIBILITY
# =========================================================
requested_months = None
if desired_duration == "6 months":
    requested_months = 6
elif desired_duration == "9 months":
    requested_months = 9
elif desired_duration == "1 year":
    requested_months = 12
elif desired_duration == "2 years":
    requested_months = 24

st.subheader("Planning Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Carriers", f"{total_carriers}")
c2.metric("Total Volume", f"{int(total_volume):,}")
c3.metric("Estimated Timeline", f"{estimated_months} months")
c4.metric("Tier Lag", f"{int(tier_stagger_weeks)} weeks")

if requested_months is not None:
    if estimated_months <= requested_months:
        st.success(f"Requested duration is feasible. Estimated completion is about {estimated_months} months.")
    elif estimated_months <= requested_months + 1.0:
        st.warning(f"Requested duration is tight. Estimated completion is about {estimated_months} months.")
    else:
        st.error(f"Requested duration is not feasible. Best estimated achievable timeline is about {estimated_months} months.")


# =========================================================
# TIER SUMMARY
# =========================================================
st.subheader("Tier Summary")

tier_summary = (
    df.groupby("tier", as_index=False)
    .agg(
        carrier_count=("carrier", "count"),
        volume=("volume", "sum"),
        start_date=("start_date", "min"),
        end_date=("end_date", "max"),
    )
    .sort_values("tier")
)

tier_summary["volume_pct_num"] = tier_summary["volume"] / total_volume
tier_summary["duration_months"] = tier_summary.apply(
    lambda row: months_between(row["start_date"], row["end_date"]),
    axis=1
)
tier_summary["mix"] = tier_summary["tier"].apply(
    lambda tier: build_mix_string(df[df["tier"] == tier])
)
tier_summary["start_lag_weeks"] = tier_summary["start_date"].apply(
    lambda d: round((d - project_start).days / 7, 1)
)

tier_display = tier_summary.rename(
    columns={
        "tier": "Tier",
        "mix": "Mix",
        "carrier_count": "Carrier Count",
        "volume": "Volume",
        "start_date": "Start Date",
        "end_date": "End Date",
        "start_lag_weeks": "Start Lag (Weeks)",
        "duration_months": "Duration (Months)",
    }
).copy()

tier_display["Volume %"] = tier_summary["volume_pct_num"].apply(pct_str)
tier_display = tier_display[
    [
        "Tier",
        "Mix",
        "Carrier Count",
        "Volume",
        "Volume %",
        "Start Lag (Weeks)",
        "Start Date",
        "End Date",
        "Duration (Months)",
    ]
]

st.dataframe(tier_display, use_container_width=True)


# =========================================================
# PIVOT TABLES
# =========================================================
st.subheader("Pivot View")

pivot_col1, pivot_col2 = st.columns(2)

with pivot_col1:
    st.markdown("**Carrier Count by Tier and Type**")
    pivot_count = pd.pivot_table(
        df,
        index="tier",
        columns="type",
        values="carrier",
        aggfunc="count",
        fill_value=0,
    ).reset_index()
    st.dataframe(pivot_count, use_container_width=True)

with pivot_col2:
    st.markdown("**Volume by Tier and Type**")
    pivot_volume = pd.pivot_table(
        df,
        index="tier",
        columns="type",
        values="volume",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    st.dataframe(pivot_volume, use_container_width=True)

st.markdown("**Estimated Completion by Tier and Month**")
tier_month_count = pd.pivot_table(
    df.assign(
        end_month_label=df["end_date"].dt.to_period("M").astype(str).map(month_label)
    ),
    index="tier",
    columns="end_month_label",
    values="carrier",
    aggfunc="count",
    fill_value=0,
).reset_index()
st.dataframe(tier_month_count, use_container_width=True)


# =========================================================
# FUNNEL VIEW
# =========================================================
st.subheader("Funnel View")

funnel_col1, funnel_col2 = st.columns(2)

tier_order = sorted(
    tier_summary["tier"].tolist(),
    key=lambda t: int(str(t).replace("Tier", "").strip())
)
funnel_summary = tier_summary.set_index("tier").loc[tier_order].reset_index()

with funnel_col1:
    st.markdown("**Carrier Funnel by Tier**")
    fig_f1, ax_f1 = plt.subplots(figsize=(8, 4))
    y_labels = funnel_summary["tier"]
    counts = funnel_summary["carrier_count"]

    ax_f1.barh(y_labels, counts)
    ax_f1.invert_yaxis()
    ax_f1.set_xlabel("Carrier Count")
    ax_f1.set_ylabel("Tier")
    ax_f1.set_title("Estimated Carriers by Tier")

    for i, v in enumerate(counts):
        ax_f1.text(v, i, f" {int(v)}", va="center")

    plt.tight_layout()
    st.pyplot(fig_f1)

with funnel_col2:
    st.markdown("**Volume Funnel by Tier**")
    fig_f2, ax_f2 = plt.subplots(figsize=(8, 4))
    y_labels = funnel_summary["tier"]
    volumes = funnel_summary["volume"]

    ax_f2.barh(y_labels, volumes)
    ax_f2.invert_yaxis()
    ax_f2.set_xlabel("Volume")
    ax_f2.set_ylabel("Tier")
    ax_f2.set_title("Estimated Volume by Tier")

    for i, v in enumerate(volumes):
        ax_f2.text(v, i, f" {int(v):,}", va="center")

    plt.tight_layout()
    st.pyplot(fig_f2)


# =========================================================
# STRATEGY TABLE
# =========================================================
st.subheader("Strategy Table")

strategy_table = (
    df.groupby(["tier", "type"], as_index=False)
    .agg(
        carrier_count=("carrier", "count"),
        volume=("volume", "sum"),
        start_date=("start_date", "min"),
        end_date=("end_date", "max"),
    )
)

strategy_table["volume_pct"] = strategy_table["volume"] / total_volume
strategy_table["duration_weeks"] = (
    (strategy_table["end_date"] - strategy_table["start_date"]).dt.days / 7
).round(1)
strategy_table["volume_pct"] = strategy_table["volume_pct"].apply(pct_str)

strategy_display = strategy_table.rename(
    columns={
        "tier": "Tier",
        "type": "Type",
        "carrier_count": "Carrier Count",
        "volume": "Volume",
        "volume_pct": "Volume %",
        "start_date": "Start Date",
        "end_date": "End Date",
        "duration_weeks": "Duration (Weeks)",
    }
)

st.dataframe(strategy_display, use_container_width=True)


# =========================================================
# CARRIER DETAIL
# =========================================================
st.subheader("Carrier-Level Plan")

carrier_view = df[
    [
        "carrier",
        "country",
        "region",
        "mode",
        "connectivity",
        "method",
        "type",
        "volume",
        "duration_weeks",
        "priority_score",
        "tier",
        "batch",
        "start_date",
        "end_date",
    ]
].copy()

carrier_view = carrier_view.rename(
    columns={
        "carrier": "Carrier",
        "country": "Country Base",
        "region": "Mapped Region",
        "mode": "Mode",
        "connectivity": "P44 Connectivity",
        "method": "Connection Method",
        "type": "Type",
        "volume": "Volume",
        "duration_weeks": "Duration (Weeks)",
        "priority_score": "Priority Score",
        "tier": "Tier",
        "batch": "Batch",
        "start_date": "Start Date",
        "end_date": "End Date",
    }
)

st.dataframe(carrier_view, use_container_width=True)


# =========================================================
# GANTT
# =========================================================
st.subheader("Onboarding Timeline")

gantt_data = (
    df.groupby(["timeline_task"], as_index=False)
    .agg(
        start_date=("start_date", "min"),
        end_date=("end_date", "max"),
        carrier_count=("carrier", "count"),
        volume=("volume", "sum"),
    )
    .sort_values(["start_date", "end_date"])
    .reset_index(drop=True)
)

fig, ax = plt.subplots(figsize=(11, max(4, len(gantt_data) * 0.6)))

for _, row in gantt_data.iterrows():
    left = mdates.date2num(row["start_date"])
    width = (row["end_date"] - row["start_date"]).days
    label = f"{row['timeline_task']} ({int(row['carrier_count'])})"
    ax.barh(label, width, left=left)

ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b-%Y"))
plt.xticks(rotation=45)
ax.set_xlabel("Timeline")
ax.set_ylabel("Tier / Type")
ax.set_title("Overlap View by Tier and Connection Type")
plt.tight_layout()
st.pyplot(fig)


# =========================================================
# MONTHLY PROGRESS
# =========================================================
df["end_month"] = df["end_date"].dt.to_period("M").astype(str)

monthly_carriers = (
    df.groupby("end_month", as_index=False)["carrier"]
    .count()
    .sort_values("end_month")
)
monthly_carriers["cumulative_carrier_pct"] = monthly_carriers["carrier"].cumsum() / total_carriers
monthly_carriers["Month"] = monthly_carriers["end_month"].apply(month_label)
monthly_carriers["Estimated Carrier Count"] = monthly_carriers["carrier"]
monthly_carriers["Estimated % of Carriers"] = monthly_carriers["cumulative_carrier_pct"].apply(pct_str)

monthly_volume = (
    df.groupby("end_month", as_index=False)["volume"]
    .sum()
    .sort_values("end_month")
)
monthly_volume["cumulative_volume_pct"] = monthly_volume["volume"].cumsum() / total_volume
monthly_volume["Month"] = monthly_volume["end_month"].apply(month_label)
monthly_volume["Estimated Volume"] = monthly_volume["volume"]
monthly_volume["Estimated % of Volume"] = monthly_volume["cumulative_volume_pct"].apply(pct_str)

st.subheader("Onboarding Progress")
col_a, col_b = st.columns(2)

with col_a:
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    x_vals = list(range(len(monthly_carriers)))
    ax1.plot(x_vals, monthly_carriers["cumulative_carrier_pct"], marker="o")
    ax1.set_xticks(x_vals)
    ax1.set_xticklabels(monthly_carriers["Month"], rotation=45)
    ax1.set_title("% of Carriers Estimated to Be Completed by Month")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Percent of Carriers")
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    st.pyplot(fig1)

with col_b:
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    x_vals = list(range(len(monthly_volume)))
    ax2.plot(x_vals, monthly_volume["cumulative_volume_pct"], marker="o")
    ax2.set_xticks(x_vals)
    ax2.set_xticklabels(monthly_volume["Month"], rotation=45)
    ax2.set_title("% of Volume Estimated to Be Completed by Month")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Percent of Volume")
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    st.pyplot(fig2)

st.subheader("Monthly Estimated Completion Summary")

month_col1, month_col2 = st.columns(2)

with month_col1:
    carrier_monthly_display = monthly_carriers[
        ["Month", "Estimated Carrier Count", "Estimated % of Carriers"]
    ]
    st.dataframe(carrier_monthly_display, use_container_width=True)

with month_col2:
    volume_monthly_display = monthly_volume[
        ["Month", "Estimated Volume", "Estimated % of Volume"]
    ]
    st.dataframe(volume_monthly_display, use_container_width=True)


# =========================================================
# QUARTERLY + REGION SUMMARY
# =========================================================
st.subheader("Quarterly Summary")

df["end_quarter"] = df["end_date"].dt.to_period("Q").astype(str)
quarterly = (
    df.groupby("end_quarter", as_index=False)
    .agg(
        carriers_estimated_to_be_completed=("carrier", "count"),
        volume_estimated_to_be_completed=("volume", "sum"),
    )
    .sort_values("end_quarter")
)

quarterly["cumulative_volume_pct"] = quarterly["volume_estimated_to_be_completed"].cumsum() / total_volume
quarterly["cumulative_volume_pct"] = quarterly["cumulative_volume_pct"].apply(pct_str)

quarterly_display = quarterly.rename(
    columns={
        "end_quarter": "Quarter",
        "carriers_estimated_to_be_completed": "Carriers Estimated to Be Completed",
        "volume_estimated_to_be_completed": "Volume Estimated to Be Completed",
        "cumulative_volume_pct": "Cumulative Volume %",
    }
)

st.dataframe(quarterly_display, use_container_width=True)

st.subheader("Region Summary")
region_summary = (
    df.groupby("region", as_index=False)
    .agg(
        carrier_count=("carrier", "count"),
        volume=("volume", "sum"),
        start_date=("start_date", "min"),
        end_date=("end_date", "max"),
    )
)

region_summary["volume_pct"] = region_summary["volume"] / total_volume
region_summary["duration_months"] = region_summary.apply(
    lambda row: months_between(row["start_date"], row["end_date"]),
    axis=1
)
region_summary["volume_pct"] = region_summary["volume_pct"].apply(pct_str)

region_display = region_summary.rename(
    columns={
        "region": "Region",
        "carrier_count": "Carrier Count",
        "volume": "Volume",
        "volume_pct": "Volume %",
        "start_date": "Start Date",
        "end_date": "End Date",
        "duration_months": "Duration (Months)",
    }
)

st.dataframe(region_display, use_container_width=True)


# =========================================================
# PLAN SUMMARY (CALC VIEW)
# =========================================================
st.subheader("Plan Summary (Calc View)")

summary_rows = []

for _, row in funnel_summary.iterrows():
    tier_name = row["tier"]
    tier_df = df[df["tier"] == tier_name].copy()
    tier_total = len(tier_df)

    eld_count = int((tier_df["type"] == "ELD").sum())
    app_count = int((tier_df["type"] == "APP").sum())
    api_count = int((tier_df["type"] == "API").sum())
    new_count = int((tier_df["type"] == "NEW").sum())

    eld_pct = f"{(eld_count / tier_total):.0%}" if tier_total else "0%"
    app_pct = f"{(app_count / tier_total):.0%}" if tier_total else "0%"
    api_pct = f"{(api_count / tier_total):.0%}" if tier_total else "0%"
    new_pct = f"{(new_count / tier_total):.0%}" if tier_total else "0%"

    summary_rows.append(
        {
            "Tier": tier_name,
            "Mix Calc": (
                f"ELD:{eld_count} ({eld_pct}) | "
                f"APP:{app_count} ({app_pct}) | "
                f"API:{api_count} ({api_pct}) | "
                f"NEW:{new_count} ({new_pct})"
            ),
            "Volume Coverage": pct_str(row["volume_pct_num"]),
            "Start Lag (Weeks)": round((row["start_date"] - project_start).days / 7, 1),
            "Timeline Calc": f"{months_between(row['start_date'], row['end_date'])} months",
        }
    )

plan_summary_df = pd.DataFrame(summary_rows)
st.dataframe(plan_summary_df, use_container_width=True)


# =========================================================
# DISCLAIMER
# =========================================================
st.info(
    "This estimate is directional and based on carrier mix, connection assumptions, "
    "selected onboarding style, wave logic, and current planning rules. "
    "Actual timelines may vary based on carrier responsiveness, technical readiness, "
    "team bandwidth, and final scope."
)
