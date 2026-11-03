
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

# Dashboard styling: works without external image or font downloads.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
:root { --navy:#10243d; --navy2:#1b3552; --gold:#f4b83f; --ink:#14233b; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(135deg,#f4f7fc 0%,#eef2fa 100%); color:var(--ink); }
.block-container { padding-top:1.2rem; padding-bottom:2rem; max-width:1500px; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#10213b 0%,#193858 100%); }
[data-testid="stSidebar"] * { color:#f7f9ff !important; }
[data-testid="stSidebar"] .stMarkdown h2 { color:#f4b83f !important; }
.hero { background:linear-gradient(110deg,rgba(10,26,48,.98),rgba(24,54,83,.91)), linear-gradient(45deg,#10243d,#315477); color:white; padding:30px 34px; border-radius:18px; margin-bottom:20px; box-shadow:0 10px 28px rgba(18,39,67,.18); }
.hero h1 { color:white; font-size:clamp(2rem,4vw,3rem); margin:0 0 5px 0; font-weight:800; letter-spacing:-1px; }
.hero p { color:#e7edf7; font-size:1.05rem; margin:0; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:8px; border-bottom:1px solid #dce3f0; }
[data-testid="stTabs"] button[role="tab"] { background:#e8edf8; border-radius:12px 12px 0 0; padding:12px 22px; font-weight:700; color:#14233b; }
[data-testid="stTabs"] button[aria-selected="true"] { background:#fff; border-bottom:4px solid #f4b83f; }
[data-testid="stMetric"] { background:#fff; border:1px solid #e0e6f1; padding:14px 16px; border-radius:13px; box-shadow:0 3px 12px rgba(25,48,78,.06); }
[data-testid="stMetricLabel"] { color:#52627a; font-weight:600; }
[data-testid="stMetricValue"] { color:#14233b; font-weight:800; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius:14px; }
.chart-card { background:#fff; border:1px solid #dce4f0; border-radius:14px; padding:12px 14px 4px; box-shadow:0 4px 14px rgba(20,42,70,.06); margin-bottom:12px; }
.chart-title { background:#172d46; color:#fff; font-weight:700; padding:9px 13px; border-radius:9px; margin-bottom:8px; }
div.stButton > button[kind="primary"] { background:#f4b83f; color:#12243c; border:0; font-weight:800; border-radius:10px; }
div.stButton > button[kind="primary"]:hover { background:#e9a923; color:#10213b; }
[data-testid="stAlert"] { border-radius:12px; }
</style>
""", unsafe_allow_html=True)


# ==================================================
# 2. LOAD TRAINED MODEL
# ==================================================

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "house_price_xgb_model.pkl")


try:
    model = load_model()
    model_error = None
except Exception as error:
    model = None
    model_error = str(error)


# ==================================================
# 3. NEIGHBORHOOD MAPPING
# ==================================================

neighborhood_mapping = {
    "Blmngtn": "Bloomington Heights",
    "Blueste": "Bluestem",
    "BrDale": "Briardale",
    "BrkSide": "Brookside",
    "ClearCr": "Clear Creek",
    "CollgCr": "College Creek",
    "Crawfor": "Crawford",
    "Edwards": "Edwards",
    "Gilbert": "Gilbert",
    "IDOTRR": "Iowa DOT and Rail Road",
    "MeadowV": "Meadow Village",
    "Mitchel": "Mitchell",
    "NAmes": "North Ames",
    "NPkVill": "Northpark Villa",
    "NWAmes": "Northwest Ames",
    "NoRidge": "Northridge",
    "NridgHt": "Northridge Heights",
    "OldTown": "Old Town",
    "SWISU": "South and West of Iowa State University",
    "Sawyer": "Sawyer",
    "SawyerW": "Sawyer West",
    "Somerst": "Somerset",
    "StoneBr": "Stone Brook",
    "Timber": "Timberland",
    "Veenker": "Veenker"
}


# ==================================================
# 4. CATEGORY LABELS
# ==================================================

CATEGORY_LABELS = {

    "zoning_classification": {
        "RL": "Residential - Low Density",
        "RM": "Residential - Medium Density",
        "C (all)": "Commercial",
        "FV": "Floating Village Residential",
        "RH": "Residential - High Density"
    },

    "road_access_type": {
        "Pave": "Paved Road",
        "Grvl": "Gravel Road"
    },

    "lot_shape": {
        "Reg": "Regular",
        "IR1": "Slightly Irregular",
        "IR2": "Moderately Irregular",
        "IR3": "Very Irregular"
    },

    "land_contour": {
        "Lvl": "Level",
        "Bnk": "Banked",
        "Low": "Depression",
        "HLS": "Hillside"
    },

    "building_type": {
        "1Fam": "Single-Family Detached",
        "2fmCon": "Two-Family Conversion",
        "Duplex": "Duplex",
        "TwnhsE": "Townhouse - End Unit",
        "Twnhs": "Townhouse - Inside Unit"
    },

    "house_style": {
        "1Story": "One Story",
        "1.5Fin": "One and a Half Stories - Finished",
        "1.5Unf": "One and a Half Stories - Unfinished",
        "2Story": "Two Stories",
        "2.5Fin": "Two and a Half Stories - Finished",
        "2.5Unf": "Two and a Half Stories - Unfinished",
        "SFoyer": "Split Foyer",
        "SLvl": "Split Level"
    },

    "quality": {
        "Ex": "Excellent",
        "Gd": "Good",
        "TA": "Typical / Average",
        "Fa": "Fair",
        "Po": "Poor"
    },

    "central_air_conditioning": {
        "Y": "Yes",
        "N": "No"
    },

    "garage_type": {
        "Attchd": "Attached Garage",
        "Detchd": "Detached Garage",
        "BuiltIn": "Built-in Garage",
        "CarPort": "Car Port",
        "Basment": "Basement Garage",
        "2Types": "More Than One Garage Type"
    },

    "garage_finish": {
        "Fin": "Finished",
        "RFn": "Rough Finished",
        "Unf": "Unfinished"
    },

    "basement_exposure": {
        "Gd": "Good Exposure",
        "Av": "Average Exposure",
        "Mn": "Minimum Exposure",
        "No": "No Exposure"
    },

    "kitchen_quality": {
        "Ex": "Excellent",
        "Gd": "Good",
        "TA": "Typical / Average",
        "Fa": "Fair"
    }
}

CATEGORY_LABELS["neighborhood"] = neighborhood_mapping


# Features that share the same quality codes
QUALITY_FEATURES = [
    "exterior_quality",
    "exterior_condition",
    "basement_quality",
    "basement_condition",
    "heating_quality",
    "fireplace_quality",
    "garage_quality",
    "garage_condition",
    "pool_quality"
]


# ==================================================
# 5. FRIENDLY FEATURE LABELS
# ==================================================

FEATURE_LABELS = {
    "overall_quality": "Overall Quality",
    "above_ground_living_area": "Living Area (sq ft)",
    "neighborhood": "Neighborhood",
    "year_built": "Year Built",
    "bedrooms_above_ground": "Bedrooms",
    "garage_capacity": "Garage Capacity (cars)",
    "total_basement_area": "Total Basement Area (sq ft)",
    "garage_area": "Garage Area (sq ft)",
    "full_bathrooms": "Full Bathrooms",
    "first_floor_area": "First Floor Area (sq ft)",
    "year_remodeled": "Year Remodeled",
    "number_of_fireplaces": "Number of Fireplaces",
    "lot_area": "Lot Area (sq ft)",
    "street_frontage": "Street Frontage (ft)",
    "second_floor_area": "Second Floor Area (sq ft)",
    "wood_deck_area": "Wood Deck Area (sq ft)",
    "open_porch_area": "Open Porch Area (sq ft)",
    "screen_porch_area": "Screen Porch Area (sq ft)",
    "pool_area": "Pool Area (sq ft)",
    "half_bathrooms": "Half Bathrooms",
    "kitchen_quality": "Kitchen Quality",
    "central_air_conditioning": "Central Air Conditioning",
    "garage_type": "Garage Type",
    "garage_finish": "Garage Finish",
    "basement_quality": "Basement Quality",
    "basement_exposure": "Basement Exposure",
    "building_type": "Building Type",
    "house_style": "House Style",
    "zoning_classification": "Zoning Classification",
    "roof_style": "Roof Style",
    "roof_material": "Roof Material",
    "foundation_type": "Foundation Type",
    "heating_type": "Heating Type",
    "heating_quality": "Heating Quality",
    "sale_condition": "Sale Condition",
    "sale_type": "Sale Type"
}


def friendly_label(feature):
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]

    return feature.replace("_", " ").title()


def get_category_label(feature, code):

    if feature in QUALITY_FEATURES:
        return CATEGORY_LABELS["quality"].get(code, code)

    return CATEGORY_LABELS.get(feature, {}).get(code, code)


# ==================================================
# 6. EXTRACT MODEL FEATURE INFORMATION
# ==================================================

feature_columns = []
numeric_columns = []
categorical_columns = []
categorical_options = {}

if model is not None:

    preprocessor = model.named_steps["preprocessor"]

    feature_columns = (
        preprocessor.feature_names_in_.tolist()
    )

    for name, transformer, columns in preprocessor.transformers_:

        if isinstance(transformer, str):
            continue

        encoder = None

        if isinstance(transformer, Pipeline):

            for step in transformer.named_steps.values():

                if isinstance(step, OneHotEncoder):
                    encoder = step
                    break

        elif isinstance(transformer, OneHotEncoder):

            encoder = transformer

        if encoder is not None:

            categorical_columns.extend(list(columns))

            for feature, categories in zip(
                columns,
                encoder.categories_
            ):
                categorical_options[feature] = [
                    value for value in categories
                    if not pd.isna(value)
                ]

        else:

            numeric_columns.extend(list(columns))


# ==================================================
# 7. REUSABLE DROPDOWN FUNCTION
# ==================================================

def category_dropdown(label, feature_name, options):

    return st.selectbox(
        label,
        options=options,
        index=None,
        placeholder="Select an option",
        format_func=lambda code: get_category_label(
            feature_name, code
        ),
        key=f"select_{feature_name}"
    )


# ==================================================
# 8. APPLICATION HEADER
# ==================================================

with st.sidebar:
    st.markdown("## 🏠 House Price\nPrediction")
    st.caption("ANALYZE  •  EXPLORE  •  PREDICT")
    st.divider()
    st.markdown("### 💡 Discover Insights")
    st.write("Explore housing data, understand market patterns, and make informed predictions.")
    st.divider()
    st.caption("Machine Learning • Real Estate Analytics")

st.markdown("""
<div class="hero">
  <h1>House Price Prediction</h1>
  <p>Data-driven insights for smarter real estate decisions</p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# 9. CREATE THREE TABS
# ==================================================

tab1, tab2, tab3 = st.tabs([
    "🏠  About Project",
    "📊  EDA & Visualizations",
    "🧮  Prediction Engine"
])


# ==================================================
# TAB 1: ABOUT PROJECT
# ==================================================

with tab1:

    st.header("About the Project")

    st.write(
        """
        The House Price Prediction project uses Machine Learning
        to estimate residential property prices based on various
        characteristics of a house.
        """
    )

    st.subheader("🎯 Project Objective")

    st.write(
        """
        To develop a predictive model that learns the relationship
        between house characteristics and their selling prices.
        """
    )

    st.subheader("📂 Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Training Records", "1,460")
    col2.metric("Testing Records", "1,459")
    col3.metric("Input Features", "79")

    st.write(
        """
        The dataset includes information about property quality,
        living area, neighborhood, construction, garage facilities,
        basement characteristics, and other housing attributes.
        """
    )

    st.subheader("🤖 Machine Learning Model")

    st.markdown(
        """
        - **Algorithm:** XGBoost Regressor
        - **Target transformation:** `log1p()`
        - **Numerical preprocessing:** Imputation and scaling
        - **Categorical preprocessing:** Imputation and One-Hot Encoding
        - **Hyperparameter tuning:** RandomizedSearchCV
        - **Evaluation:** RMSE, MAE, R² and RMSLE
        """
    )

    st.subheader("⚙️ Project Workflow")

    st.markdown(
        """
        1. Data collection
        2. Data cleaning and preprocessing
        3. Exploratory Data Analysis
        4. Feature analysis
        5. Model training and tuning
        6. Model evaluation
        7. Price prediction
        """
    )

    st.subheader("🛠️ Technologies Used")

    st.write(
        "Python | Pandas | NumPy | Scikit-learn | "
        "XGBoost | Matplotlib | Streamlit"
    )

    st.info(
        "This is an educational Machine Learning project. "
        "Predictions are estimates, not official property valuations."
    )


# ==================================================
# TAB 2: EDA AND VISUALIZATIONS
# ==================================================

@st.cache_data(show_spinner="Loading training dataset...")
def load_eda_dataset(csv_path, modified_time):
    """Load the EDA CSV. modified_time invalidates cache after file updates."""
    del modified_time
    # utf-8-sig handles CSVs saved with a BOM; latin-1 is a safe fallback.
    try:
        return pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="latin-1")


def find_column(dataframe, candidates):
    """Return the first matching column, ignoring capitalization and spaces."""
    normalized = {str(col).strip().lower(): col for col in dataframe.columns}
    for candidate in candidates:
        found = normalized.get(candidate.strip().lower())
        if found is not None:
            return found
    return None


def show_chart(title, fig):
    st.markdown(f'<div class="chart-title">{title}</div>', unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


with tab2:
    st.header("📊 Exploratory Data Analysis")
    st.write("Explore sale-price patterns, property categories, neighborhood differences, and the relationship between construction year and value.")

    train_file = BASE_DIR / "train.csv"
    if not train_file.is_file():
        st.error("EDA dataset not found: train.csv")
        st.warning("Upload or commit **train.csv** to the same GitHub folder as app.py, then redeploy the Streamlit app. Check that the filename is exactly train.csv (including capitalization).")
        st.code("your-repository/\n├── app.py\n├── train.csv\n├── house_price_xgb_model.pkl\n└── requirements.txt")
    else:
        try:
            eda_data = load_eda_dataset(str(train_file), train_file.stat().st_mtime)
            if eda_data.empty:
                st.error("train.csv was found, but it contains no rows.")
            else:
                target_column = find_column(eda_data, ["SalePrice", "sale_price"])
                if target_column is None:
                    st.error("The CSV loaded, but the target column was not found. Expected SalePrice or sale_price.")
                    st.caption("Available columns: " + ", ".join(map(str, eda_data.columns)))
                else:
                    prices = pd.to_numeric(eda_data[target_column], errors="coerce").dropna()
                    if prices.empty:
                        st.error(f"Column '{target_column}' exists, but it has no valid numeric prices.")
                    else:
                        st.caption(f"Dataset loaded successfully • {len(eda_data):,} rows • {len(eda_data.columns):,} columns • Source: {train_file.name}")

                        # KPI row
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Average Sale Price", f"${prices.mean():,.0f}")
                        m2.metric("Median Sale Price", f"${prices.median():,.0f}")
                        m3.metric("Minimum Price", f"${prices.min():,.0f}")
                        m4.metric("Maximum Price", f"${prices.max():,.0f}")

                        st.markdown("### Market overview")
                        left, right = st.columns(2, gap="large")

                        # 1. Price distribution
                        with left:
                            fig, ax = plt.subplots(figsize=(7.2, 3.5))
                            ax.hist(prices, bins=30, color="#3687df", edgecolor="white", linewidth=.6)
                            ax.set_xlabel("Sale Price ($)")
                            ax.set_ylabel("Number of Properties")
                            ax.grid(axis="y", alpha=.22)
                            ax.spines[["top", "right"]].set_visible(False)
                            show_chart("📊  Sale Price Distribution", fig)

                        # 2. Property type distribution
                        with right:
                            type_col = find_column(eda_data, ["BldgType", "building_type", "property_type"])
                            if type_col:
                                type_labels = {
                                    "1Fam": "Single Family", "2fmCon": "Other", "Duplex": "Duplex",
                                    "Twnhs": "Townhouse", "TwnhsE": "Townhouse"
                                }
                                type_data = eda_data[type_col].fillna("Unknown").astype(str).map(lambda x: type_labels.get(x, x))
                                type_counts = type_data.value_counts()
                                fig, ax = plt.subplots(figsize=(7.2, 3.5))
                                ax.pie(type_counts.values, labels=None, autopct="%1.1f%%", startangle=90,
                                       pctdistance=.72, wedgeprops={"linewidth": 1, "edgecolor": "white"})
                                ax.legend(type_counts.index, loc="center left", bbox_to_anchor=(1.0, .5), frameon=False)
                                ax.set_aspect("equal")
                                show_chart("🏘️  Property Types", fig)
                            else:
                                st.info("Property-type chart unavailable: expected BldgType, building_type, or property_type.")

                        left, right = st.columns(2, gap="large")

                        # 3. Neighborhood price comparison
                        with left:
                            neighborhood_col = find_column(eda_data, ["Neighborhood", "neighborhood"])
                            if neighborhood_col:
                                temp = eda_data[[neighborhood_col, target_column]].copy()
                                temp[target_column] = pd.to_numeric(temp[target_column], errors="coerce")
                                neighborhood_avg = temp.dropna().groupby(neighborhood_col)[target_column].mean().nlargest(8).sort_values()
                                labels = [neighborhood_mapping.get(str(x), str(x)) for x in neighborhood_avg.index]
                                fig, ax = plt.subplots(figsize=(7.2, 3.7))
                                ax.barh(labels, neighborhood_avg.values, color="#3687df")
                                ax.set_xlabel("Average Sale Price ($)")
                                ax.grid(axis="x", alpha=.22)
                                ax.spines[["top", "right", "left"]].set_visible(False)
                                ax.tick_params(axis="y", labelsize=8)
                                show_chart("🏡  Average Price by Neighborhood", fig)
                            else:
                                st.info("Neighborhood chart unavailable: expected Neighborhood or neighborhood.")

                        # 4. Average price by construction period
                        with right:
                            year_col = find_column(eda_data, ["YearBuilt", "year_built"])
                            if year_col:
                                temp = eda_data[[year_col, target_column]].copy()
                                temp[year_col] = pd.to_numeric(temp[year_col], errors="coerce")
                                temp[target_column] = pd.to_numeric(temp[target_column], errors="coerce")
                                temp = temp.dropna()
                                temp["Build Decade"] = (temp[year_col] // 10 * 10).astype(int)
                                trend = temp.groupby("Build Decade")[target_column].mean().sort_index()
                                fig, ax = plt.subplots(figsize=(7.2, 3.7))
                                ax.plot(trend.index, trend.values, marker="o", linewidth=2.2, color="#2878dc")
                                ax.set_xlabel("Year Built (decade)")
                                ax.set_ylabel("Average Sale Price ($)")
                                ax.grid(alpha=.22)
                                ax.spines[["top", "right"]].set_visible(False)
                                show_chart("📈  Price Trend by Year Built", fig)
                            else:
                                st.info("Year trend chart unavailable: expected YearBuilt or year_built.")

                        # Additional analytical view
                        with st.expander("Explore feature correlations"):
                            numeric_data = eda_data.select_dtypes(include=np.number)
                            if target_column in numeric_data.columns and numeric_data.shape[1] > 1:
                                corr = numeric_data.corr(numeric_only=True)[target_column].drop(target_column).dropna()
                                corr = corr.reindex(corr.abs().sort_values(ascending=False).head(12).index).sort_values()
                                fig, ax = plt.subplots(figsize=(9, 4.5))
                                corr.plot(kind="barh", ax=ax, color="#3687df")
                                ax.set_xlabel("Correlation with Sale Price")
                                ax.grid(axis="x", alpha=.2)
                                ax.spines[["top", "right"]].set_visible(False)
                                st.pyplot(fig, use_container_width=True)
                                plt.close(fig)
                                st.caption("Correlation indicates linear association; it does not establish causation.")
                            else:
                                st.info("Not enough numeric columns for correlation analysis.")

                        with st.expander("View dataset preview"):
                            st.dataframe(eda_data.head(10), use_container_width=True)
                        with st.expander("View descriptive statistics"):
                            st.dataframe(eda_data.describe(include="all").T, use_container_width=True)

        except Exception as error:
            st.error("The EDA dataset was found, but Streamlit could not load or visualize it.")
            st.exception(error)
            st.info("Check the CSV format and the Streamlit Cloud logs for the full traceback.")


# ==================================================
# TAB 3: PREDICTION ENGINE
# ==================================================

with tab3:

    st.header("🏡 House Price Prediction Engine")

    st.write(
        "Enter the house information below. "
        "Additional fields are optional."
    )

    if model is None:

        st.error(
            "The trained model could not be loaded."
        )

        st.code(model_error or "Unknown model loading error.")

    else:

        # ------------------------------------------
        # ESSENTIAL INPUTS
        # ------------------------------------------

        st.subheader("1. Essential House Information")

        st.caption(
            "These are the main details used to estimate "
            "the property price."
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            overall_quality = st.number_input(
                "Overall Quality",
                min_value=1,
                max_value=10,
                value=None,
                placeholder="Enter quality (1-10)",
                key="essential_quality"
            )

            above_ground_living_area = st.number_input(
                "Living Area (sq ft)",
                min_value=0,
                value=None,
                placeholder="Enter living area",
                step=100,
                key="essential_living_area"
            )

        with col2:

            neighborhood = category_dropdown(
                "Neighborhood",
                "neighborhood",
                list(neighborhood_mapping.keys())
            )

            year_built = st.number_input(
                "Year Built",
                min_value=1800,
                max_value=2010,
                value=None,
                placeholder="Enter construction year",
                key="essential_year"
            )

        with col3:

            bedrooms_above_ground = st.number_input(
                "Bedrooms",
                min_value=0,
                value=None,
                placeholder="Enter bedrooms",
                key="essential_bedrooms"
            )

            garage_capacity = st.number_input(
                "Garage Capacity (cars)",
                min_value=0,
                value=None,
                placeholder="Enter garage capacity",
                key="essential_garage"
            )

        essential_values = {
            "overall_quality": overall_quality,
            "above_ground_living_area":
                above_ground_living_area,
            "neighborhood": neighborhood,
            "year_built": year_built,
            "bedrooms_above_ground":
                bedrooms_above_ground,
            "garage_capacity": garage_capacity
        }

        essential_features = set(essential_values.keys())

        st.divider()

        # ------------------------------------------
        # ADDITIONAL NUMERICAL AND CATEGORICAL INPUTS
        # ------------------------------------------

        st.subheader("2. Additional House Information")

        st.caption(
            "Expand the sections below to provide more details. "
            "Leave fields blank when information is unavailable."
        )

        optional_values = {}

        # Additional numerical features
        optional_numeric = [
            col for col in numeric_columns
            if col not in essential_features
        ]

        with st.expander(
            "📐 Additional Numerical Features",
            expanded=False
        ):

            st.write(
                "Enter measurements, areas, counts and years."
            )

            num_cols = st.columns(3)

            for index, feature in enumerate(optional_numeric):

                with num_cols[index % 3]:

                    value = st.number_input(
                        friendly_label(feature),
                        value=None,
                        placeholder="Leave blank if unknown",
                        step=1.0,
                        key=f"optional_num_{feature}"
                    )

                    optional_values[feature] = value

        # Additional categorical features
        optional_categorical = [
            col for col in categorical_columns
            if col not in essential_features
        ]

        with st.expander(
            "🏠 Additional Categorical Features",
            expanded=False
        ):

            st.write(
                "Select the categories that describe the property."
            )

            cat_cols = st.columns(3)

            for index, feature in enumerate(optional_categorical):

                options = categorical_options.get(
                    feature, []
                )

                if not options:
                    continue

                with cat_cols[index % 3]:

                    value = category_dropdown(
                        friendly_label(feature),
                        feature,
                        options
                    )

                    optional_values[feature] = value

        st.divider()

        # ------------------------------------------
        # PREDICTION BUTTON
        # ------------------------------------------

        predict_button = st.button(
            "🔮 Predict House Price",
            type="primary",
            use_container_width=True
        )

        if predict_button:

            # Initialize every expected feature
            input_data = {}

            for col in numeric_columns:
                input_data[col] = [np.nan]

            for col in categorical_columns:
                input_data[col] = pd.Series(
                    [np.nan],
                    dtype="object"
                )

            # Build complete 79-feature DataFrame
            user_input = pd.DataFrame(
                input_data,
                columns=feature_columns
            )

            # Combine essential and optional inputs
            all_values = {
                **essential_values,
                **optional_values
            }

            # Insert values provided by the user
            for feature, value in all_values.items():

                if value is not None:
                    user_input.loc[0, feature] = value

            # --------------------------------------
            # MAKE PREDICTION
            # --------------------------------------

            try:

                log_prediction = model.predict(
                    user_input
                )[0]

                predicted_price = np.expm1(
                    log_prediction
                )

                predicted_price = max(
                    0,
                    predicted_price
                )

                st.success(
                    "Prediction generated successfully!"
                )

                st.divider()

                st.subheader("💰 Estimated House Price")

                st.metric(
                    "Predicted Selling Price",
                    f"${predicted_price:,.0f}"
                )

                st.caption(
                    "This is an estimated price generated "
                    "by your trained XGBoost model."
                )

                with st.expander(
                    "View submitted house information"
                ):

                    submitted_data = {
                        feature: value
                        for feature, value in all_values.items()
                        if value is not None
                    }

                    st.json(submitted_data)

                    st.write(
                        f"Features provided: "
                        f"{len(submitted_data)} / "
                        f"{len(feature_columns)}"
                    )

            except Exception as error:

                st.error(
                    f"Prediction failed: {error}"
                )
