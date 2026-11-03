# House Price Prediction

An interactive machine learning application that explores housing data and estimates residential sale prices. The project includes a Streamlit dashboard, exploratory data analysis visualizations, and an XGBoost regression model trained on house characteristics.

## Features

- Interactive Streamlit web application
- Exploratory analysis of sale-price distributions and feature relationships
- Visualizations for:
  - Sale price distribution
  - Feature correlation
  - Living area versus sale price
  - Average sale price by overall quality
- House-price predictions from essential and optional property details
- Pre-trained XGBoost model with numerical imputation/scaling and categorical one-hot encoding

## Model workflow

1. Load and clean the housing data.
2. Explore distributions, correlations, and important features.
3. Preprocess numerical and categorical features.
4. Train and tune an XGBoost regression model.
5. Apply a `log1p` transformation to the target during training.
6. Generate predictions and convert them back with `expm1`.

The application reports the estimate produced by the saved model; it is not an official property valuation.

## Tech stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Matplotlib
- Streamlit

## Project structure

```text
.
├── app.py                         # Streamlit application
├── house_price_xgb_model.pkl      # Saved trained model pipeline
├── requirements.txt               # Python dependencies
├── train.csv                      # Training data used by the app's EDA tab
├── test.csv                       # Test data
├── data/
│   ├── train.csv                  # Data copy used for analysis/training
│   └── test.csv
├── predictions/
│   └── kaggle_submission.csv      # Example prediction output
└── notebook/
    └── House_Price_Prediction.ipynb
```

## Installation

Use Python 3.10 or newer, create a virtual environment, and install the dependencies:

```bash
git clone https://github.com/basmbye08/house_price_prediction_project.git
cd house_price_prediction_project

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run the application

Start the Streamlit app from the project root:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually `http://localhost:8501`.

The app expects `train.csv` and `house_price_xgb_model.pkl` to be in the same directory as `app.py`.

## Using the app

1. Open **About Project** to review the dataset and model workflow.
2. Open **EDA & Visualizations** to inspect price statistics and feature relationships.
3. Open **Prediction Engine** and provide the required house details.
4. Optionally add more numerical or categorical features.
5. Select **Predict House Price** to generate an estimated selling price.

## Notebook

The training and analysis notebook is available at `notebook/House_Price_Prediction.ipynb`. The saved model used by the dashboard is `house_price_xgb_model.pkl`.

## License

No license has been specified for this project.
