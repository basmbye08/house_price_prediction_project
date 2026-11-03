
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

st.title("🏠 HOUSE PRICE PREDICTION")

st.write(
    "Explore housing data, understand important features, "
    "and estimate the selling price of a property using "
    "Machine Learning."
)

st.divider()


# ==================================================
# 9. CREATE THREE TABS
# ==================================================

tab1, tab2, tab3 = st.tabs([
    "📌 About Project",
    "📊 EDA & Visualizations",
    "🏡 Prediction Engine"
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

with tab2:

    st.header("📊 Exploratory Data Analysis")

    st.write(
        "Explore the distribution of house prices and "
        "relationships between important housing features."
    )

    # Load the training dataset directly from the project folder
    eda_data = None
    train_file = BASE_DIR / "train.csv"

    if train_file.exists():
        try:
            eda_data = pd.read_csv(train_file)
            st.caption(f"Loaded dataset: {train_file.name}")
        except Exception as error:
            st.error(f"Could not read train.csv: {error}")
    else:
        st.error(
            "train.csv not found. Make sure it is in the same "
            "folder as app.py."
        )

    if eda_data is not None:
        # Support original and renamed column names
        target_candidates = ["sale_price", "SalePrice"]
        target_column = next(
            (col for col in target_candidates if col in eda_data.columns),
            None
        )

        if target_column is None:
            st.error(
                "The dataset must contain 'SalePrice' or 'sale_price'."
            )
        else:
            prices = eda_data[target_column].dropna()

            st.success(
                f"Dataset loaded: {eda_data.shape[0]} rows "
                f"and {eda_data.shape[1]} columns."
            )

            # SUMMARY STATISTICS
            st.subheader("📈 Sale Price Summary")
            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Average Price", f"${prices.mean():,.0f}")
            c2.metric("Median Price", f"${prices.median():,.0f}")
            c3.metric("Minimum Price", f"${prices.min():,.0f}")
            c4.metric("Maximum Price", f"${prices.max():,.0f}")

            st.divider()

            # PRICE DISTRIBUTION
            st.subheader("1. Sale Price Distribution")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(prices, bins=30, edgecolor="black")
            ax.set_xlabel("Sale Price ($)")
            ax.set_ylabel("Number of Houses")
            ax.set_title("Distribution of House Sale Prices")
            st.pyplot(fig)
            plt.close(fig)

            st.write(
                "This histogram helps us understand the "
                "distribution and skewness of house prices."
            )

            st.divider()

            # CORRELATION ANALYSIS
            st.subheader("2. Feature Correlation")
            numeric_data = eda_data.select_dtypes(include=np.number)

            if target_column in numeric_data.columns:
                correlations = (
                    numeric_data.corr()[target_column]
                    .drop(target_column)
                    .dropna()
                )

                top_correlations = (
                    correlations
                    .reindex(
                        correlations.abs()
                        .sort_values(ascending=False)
                        .head(10)
                        .index
                    )
                    .sort_values()
                )

                fig, ax = plt.subplots(figsize=(10, 5))
                top_correlations.plot(kind="barh", ax=ax)
                ax.set_xlabel("Correlation with Sale Price")
                ax.set_ylabel("Feature")
                ax.set_title("Top 10 Correlated Features")
                st.pyplot(fig)
                plt.close(fig)

                st.write(
                    "Correlation measures the strength and "
                    "direction of a linear relationship. "
                    "It does not establish causation."
                )
            else:
                st.info("No numeric target column was available for correlation.")

            st.divider()

            # LIVING AREA VS SALE PRICE
            st.subheader("3. Living Area vs Sale Price")
            area_candidates = [
                "above_ground_living_area",
                "GrLivArea"
            ]
            area_column = next(
                (col for col in area_candidates if col in eda_data.columns),
                None
            )

            if area_column is not None:
                scatter_data = eda_data[
                    [area_column, target_column]
                ].dropna()

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.scatter(
                    scatter_data[area_column],
                    scatter_data[target_column],
                    alpha=0.5
                )
                ax.set_xlabel("Living Area (sq ft)")
                ax.set_ylabel("Sale Price ($)")
                ax.set_title("Living Area vs House Sale Price")
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Living area feature was not found.")

            st.divider()

            # OVERALL QUALITY VS SALE PRICE
            st.subheader("4. Overall Quality vs Sale Price")
            quality_candidates = [
                "overall_quality",
                "OverallQual"
            ]
            quality_column = next(
                (col for col in quality_candidates if col in eda_data.columns),
                None
            )

            if quality_column is not None:
                quality_data = (
                    eda_data
                    .groupby(quality_column)[target_column]
                    .mean()
                    .sort_index()
                )

                fig, ax = plt.subplots(figsize=(10, 4))
                quality_data.plot(kind="bar", ax=ax)
                ax.set_xlabel("Overall Quality")
                ax.set_ylabel("Average Sale Price ($)")
                ax.set_title("Average Sale Price by Overall Quality")
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Overall quality feature was not found.")

            st.divider()

            # DATASET PREVIEW
            with st.expander("View Dataset Preview"):
                st.dataframe(
                    eda_data.head(10),
                    use_container_width=True
                )

            with st.expander("View Descriptive Statistics"):
                st.dataframe(
                    eda_data.describe().T,
                    use_container_width=True
                )

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
