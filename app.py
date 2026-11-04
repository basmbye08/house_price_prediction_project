import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ==================================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==================================================

st.set_page_config(
    page_title="House Price Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern corporate aesthetic
st.markdown("""
    <style>
    /* Main Layout Styling */
    .stApp {
        background-color: #f8fafc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Typography */
    h1 {
        color: #0f172a !important;
        font-weight: 800 !important;
        letter-spacing: -0.025em;
    }
    h2, h3 {
        color: #1e293b !important;
        font-weight: 700 !important;
    }
    p, label {
        color: #334155 !important;
    }

    /* Metric Cards Styling */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 20px 24px;
        border-radius: 12px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    div[data-testid="stMetric"] label {
        color: #64748b !important;
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #0284c7 !important;
        font-weight: 800 !important;
        font-size: 1.875rem !important;
    }
    
    /* Custom Card Containers */
    .content-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
    }
    
    /* Primary Action Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: #ffffff !important;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 0.75rem 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(2, 132, 199, 0.25);
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%);
        box-shadow: 0 6px 12px -1px rgba(2, 132, 199, 0.35);
        transform: translateY(-1px);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Custom Radio Navigation */
    div[data-testid="stSidebar"] div.stRadio > label {
        display: none;
    }
    div[data-testid="stSidebar"] div.stRadio > div {
        gap: 8px;
    }
    </style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent


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

QUALITY_FEATURES = [
    "exterior_quality", "exterior_condition", "basement_quality",
    "basement_condition", "heating_quality", "fireplace_quality",
    "garage_quality", "garage_condition", "pool_quality"
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
    feature_columns = preprocessor.feature_names_in_.tolist()

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
            for feature, categories in zip(columns, encoder.categories_):
                categorical_options[feature] = [
                    value for value in categories if not pd.isna(value)
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
        format_func=lambda code: get_category_label(feature_name, code),
        key=f"select_{feature_name}"
    )


# ==================================================
# 8. SIDEBAR INTERACTIVE NAVIGATION (About first)
# ==================================================

with st.sidebar:
    st.markdown("### 🏠 **HOUSE PRICE PREDICTION**")
    st.caption("Real Estate Valuation & Analytics Suite")
    st.divider()
    
    st.markdown("#### **Navigation**")
    page = st.radio(
        "Go to",
        options=["📌 About Project", "📊 EDA & Visualizations", "🏡 Prediction Engine"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.divider()
    st.caption("House price prediction Engine Version 2.5 | by Basiru Mbye")


# ==================================================
# PAGE 1: ABOUT PROJECT (Comprehensive & Non-Technical)
# ==================================================

if page == "📌 About Project":
    st.title("📌 About the Project")
    st.write("Understand the background, purpose, and capability behind this real estate pricing platform.")
    st.divider()

    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.subheader("Welcome to House Price Intelligence")
    st.write(
        "Navigating the real estate market can be challenging—whether you are looking to purchase your first home, "
        "estimating the sale price of a property you own, or simply analyzing neighborhood trends. "
        "This application is designed to bridge the gap between complex housing data and clear, actionable insights."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.subheader("🎯 What This Project Solves")
    st.write(
        "Property values are shaped by dozens of moving parts: total square footage, construction quality, "
        "neighborhood desirability, age, and garage size. Traditionally, evaluating these factors requires sifting through "
        "cumbersome spreadsheets or relying solely on manual appraisals. "
        "This interactive dashboard gives users an intuitive workspace to explore historical property trends "
        "and dynamically test how changes to a home's features affect its overall market value."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.subheader("📊 Dataset & Methodology Overview")
    st.write(
        "The application is built on a comprehensive benchmark repository containing thousands of historical property records "
        "spanning up to 79 unique structural and geographical attributes. "
        "To translate these records into accurate market projections, the system utilizes the **XGBoost Regressor** machine learning model. "
        "By analyzing patterns from past sales data, the model understands the underlying weight of features (like living space and build quality) "
        "to instantly generate reliable price estimations for new properties."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.info("💡 **Disclaimer:** Valuations and insights provided by this system are algorithmic estimates derived from historical data and are purely for educational purposes.")


# ==================================================
# PAGE 2: EDA & VISUALIZATIONS
# ==================================================

elif page == "📊 EDA & Visualizations":
    st.title("📊 Exploratory Data Analysis")
    st.write("Visual inspection of underlying distributions, correlations, and key valuation metrics.")
    st.divider()

    eda_data = None
    train_file = BASE_DIR / "train.csv"

    if train_file.exists():
        try:
            eda_data = pd.read_csv(train_file)
        except Exception as error:
            st.error(f"Could not read train.csv: {error}")
    else:
        st.error("⚠️ `train.csv` was not found in the project root directory. Please place the training set alongside `app.py` to enable visualizations.")

    if eda_data is not None:
        target_candidates = ["sale_price", "SalePrice"]
        target_column = next((col for col in target_candidates if col in eda_data.columns), None)

        if target_column is None:
            st.error("The dataset must contain a 'SalePrice' or 'sale_price' target column.")
        else:
            prices = eda_data[target_column].dropna()

            # High Level Metrics Row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Average Price", f"${prices.mean():,.0f}")
            c2.metric("Median Price", f"${prices.median():,.0f}")
            c3.metric("Minimum Price", f"${prices.min():,.0f}")
            c4.metric("Maximum Price", f"${prices.max():,.0f}")

            st.divider()

            # CHART 1: HISTOGRAM
            st.markdown("<div class='content-card'>", unsafe_allow_html=True)
            st.subheader("1. Price Distribution Spectrum")
            fig_hist = px.histogram(
                eda_data, 
                x=target_column, 
                nbins=40,
                labels={target_column: "Sale Price ($)"},
                template="plotly_white",
                color_discrete_sequence=["#0284c7"]
            )
            fig_hist.update_traces(
                marker_line_color="#0369a1", 
                marker_line_width=1,
                opacity=0.85
            )
            fig_hist.update_layout(
                height=380,
                margin=dict(t=20, b=20, l=20, r=20),
                xaxis=dict(tickprefix="$", gridcolor="#f1f5f9"),
                yaxis=dict(title="Property Count", gridcolor="#f1f5f9"),
                font=dict(family="Inter", color="#1e293b")
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            st.caption("Distribution demonstrates right-skewness typical of real estate markets, with density concentrated around $150k–$200k.")
            st.markdown("</div>", unsafe_allow_html=True)

            # CHART 2: CORRELATION
            st.markdown("<div class='content-card'>", unsafe_allow_html=True)
            st.subheader("2. Top 10 Drivers of Property Value")
            numeric_data = eda_data.select_dtypes(include=np.number)

            if target_column in numeric_data.columns:
                correlations = numeric_data.corr()[target_column].drop(target_column).dropna()
                top_corrs = correlations.abs().sort_values(ascending=False).head(10)
                top_corrs_signed = correlations[top_corrs.index].sort_values()

                labels = [friendly_label(c) for c in top_corrs_signed.index]
                
                fig_corr = px.bar(
                    x=top_corrs_signed.values,
                    y=labels,
                    orientation='h',
                    labels={'x': 'Correlation Factor (r)', 'y': 'Feature'},
                    template="plotly_white",
                    color=top_corrs_signed.values,
                    color_continuous_scale="Viridis"
                )
                fig_corr.update_layout(
                    height=420,
                    margin=dict(t=20, b=20, l=20, r=20),
                    xaxis=dict(range=[0, 1.0], gridcolor="#f1f5f9"),
                    yaxis=dict(autorange="reversed"),
                    coloraxis_showscale=False,
                    font=dict(family="Inter", color="#1e293b")
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("No numeric columns found for correlation analysis.")
            st.markdown("</div>", unsafe_allow_html=True)

            # CHART 3 & 4: SCATTER AND QUALITY IN 2 COLUMNS
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("<div class='content-card'>", unsafe_allow_html=True)
                st.subheader("3. Living Area vs Price")
                area_candidates = ["above_ground_living_area", "GrLivArea"]
                area_column = next((col for col in area_candidates if col in eda_data.columns), None)

                if area_column is not None:
                    fig_scatter = px.scatter(
                        eda_data,
                        x=area_column,
                        y=target_column,
                        opacity=0.6,
                        labels={area_column: "Living Area (sq ft)", target_column: "Sale Price ($)"},
                        template="plotly_white",
                        color_discrete_sequence=["#0d9488"]
                    )
                    fig_scatter.update_layout(
                        height=360,
                        margin=dict(t=20, b=20, l=20, r=20),
                        yaxis=dict(tickprefix="$", gridcolor="#f1f5f9"),
                        xaxis=dict(gridcolor="#f1f5f9"),
                        font=dict(family="Inter", color="#1e293b")
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("Living area column missing.")
                st.markdown("</div>", unsafe_allow_html=True)

            with col_right:
                st.markdown("<div class='content-card'>", unsafe_allow_html=True)
                st.subheader("4. Average Price by Quality")
                quality_candidates = ["overall_quality", "OverallQual"]
                quality_column = next((col for col in quality_candidates if col in eda_data.columns), None)

                if quality_column is not None:
                    qual_group = eda_data.groupby(quality_column)[target_column].mean().reset_index()
                    fig_qual = px.bar(
                        qual_group,
                        x=quality_column,
                        y=target_column,
                        labels={quality_column: "Quality (1-10)", target_column: "Avg Price ($)"},
                        template="plotly_white",
                        color_discrete_sequence=["#6366f1"]
                    )
                    fig_qual.update_layout(
                        height=360,
                        margin=dict(t=20, b=20, l=20, r=20),
                        yaxis=dict(tickprefix="$", gridcolor="#f1f5f9"),
                        xaxis=dict(gridcolor="#f1f5f9", dtick=1),
                        font=dict(family="Inter", color="#1e293b")
                    )
                    st.plotly_chart(fig_qual, use_container_width=True)
                else:
                    st.info("Quality column missing.")
                st.markdown("</div>", unsafe_allow_html=True)

            # Data Preview Expanders
            with st.expander("🔍 Explore Raw Dataset Records"):
                st.dataframe(eda_data.head(20), use_container_width=True)

            with st.expander("📈 Comprehensive Statistical Breakdown"):
                st.dataframe(eda_data.describe().T, use_container_width=True)


# ==================================================
# PAGE 3: PREDICTION ENGINE
# ==================================================

elif page == "🏡 Prediction Engine":
    st.title("🏡 Valuation Engine")
    st.write("Configure property attributes below to obtain a real-time price estimation.")
    st.divider()

    if model is None:
        st.error("⚠️ The trained machine learning model could not be initialized.")
        st.code(model_error or "Model file unreadable or missing.")
    else:
        # Mandatory Details Section
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.subheader("1. Core Property Specifications")
        st.caption("Essential parameters carrying high weighting in structural valuation.")

        col1, col2, col3 = st.columns(3)

        with col1:
            overall_quality = st.number_input(
                "Overall Quality (1-10 Scale)",
                min_value=1,
                max_value=10,
                value=None,
                placeholder="e.g., 7",
                key="essential_quality"
            )
            above_ground_living_area = st.number_input(
                "Above Ground Area (sq ft)",
                min_value=0,
                value=None,
                placeholder="e.g., 1800",
                step=50,
                key="essential_living_area"
            )

        with col2:
            neighborhood = category_dropdown(
                "Neighborhood Location",
                "neighborhood",
                list(neighborhood_mapping.keys())
            )
            year_built = st.number_input(
                "Construction Year",
                min_value=1800,
                max_value=2026,
                value=None,
                placeholder="e.g., 2005",
                key="essential_year"
            )

        with col3:
            bedrooms_above_ground = st.number_input(
                "Bedrooms",
                min_value=0,
                value=None,
                placeholder="e.g., 3",
                key="essential_bedrooms"
            )
            garage_capacity = st.number_input(
                "Garage Car Capacity",
                min_value=0,
                value=None,
                placeholder="e.g., 2",
                key="essential_garage"
            )
        st.markdown("</div>", unsafe_allow_html=True)

        essential_values = {
            "overall_quality": overall_quality,
            "above_ground_living_area": above_ground_living_area,
            "neighborhood": neighborhood,
            "year_built": year_built,
            "bedrooms_above_ground": bedrooms_above_ground,
            "garage_capacity": garage_capacity
        }

        essential_features = set(essential_values.keys())

        # Optional Details Section
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.subheader("2. Supplementary Characteristics (Optional)")
        st.caption("Refine valuation precision by providing additional measurements and structural options.")

        optional_values = {}

        optional_numeric = [col for col in numeric_columns if col not in essential_features]
        with st.expander("📐 Additional Measurements & Areas", expanded=False):
            num_cols = st.columns(3)
            for index, feature in enumerate(optional_numeric):
                with num_cols[index % 3]:
                    value = st.number_input(
                        friendly_label(feature),
                        value=None,
                        placeholder="Optional",
                        step=1.0,
                        key=f"optional_num_{feature}"
                    )
                    optional_values[feature] = value

        optional_categorical = [col for col in categorical_columns if col not in essential_features]
        with st.expander("🏠 Additional Structural Categorical Attributes", expanded=False):
            cat_cols = st.columns(3)
            for index, feature in enumerate(optional_categorical):
                options = categorical_options.get(feature, [])
                if not options:
                    continue
                with cat_cols[index % 3]:
                    value = category_dropdown(
                        friendly_label(feature),
                        feature,
                        options
                    )
                    optional_values[feature] = value
        st.markdown("</div>", unsafe_allow_html=True)

        # Action Trigger & Output
        predict_button = st.button("🔮 Generate Valuation Estimate", use_container_width=True)

        if predict_button:
            input_data = {}
            for col in numeric_columns:
                input_data[col] = [np.nan]
            for col in categorical_columns:
                input_data[col] = pd.Series([np.nan], dtype="object")

            user_input = pd.DataFrame(input_data, columns=feature_columns)

            all_values = {**essential_values, **optional_values}
            for feature, value in all_values.items():
                if value is not None:
                    user_input.loc[0, feature] = value

            try:
                log_prediction = model.predict(user_input)[0]
                predicted_price = np.expm1(log_prediction)
                predicted_price = max(0, predicted_price)

                st.success("✅ Valuation complete!")
                
                # Metric Result Card
                st.markdown("<div class='content-card' style='border-left: 6px solid #0284c7;'>", unsafe_allow_html=True)
                res_col1, res_col2 = st.columns([1, 2])
                with res_col1:
                    st.metric(
                        label="Estimated Price",
                        value=f"${predicted_price:,.0f}"
                    )
                with res_col2:
                    st.markdown("#### **Valuation Summary**")
                    st.write(
                        "Estimated standard market value generated using gradient boosted trees trained "
                        "on verified transactional housing records."
                    )
                
                with st.expander("📄 Review Submitted Input Payload"):
                    submitted_data = {f: v for f, v in all_values.items() if v is not None}
                    st.json(submitted_data)
                    st.caption(f"Active inputs provided: {len(submitted_data)} / {len(feature_columns)}")
                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as error:
                st.error(f"Valuation pipeline execution failed: {error}")