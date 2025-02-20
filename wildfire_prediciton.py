import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import lightgbm as lgb
import os
import logging
import kagglehub  

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

SYNTHETIC_PATH_CSV = r"C:\Users\taha_\OneDrive\Desktop\wildfire_prediction\synthetic_wildfire_data.csv"


def load_data():
    """Loads data from Kaggle using kagglehub."""
    dataset_id = "ayoubjadouli/morocco-wildfire-predictions-2010-2022-ml-dataset"
    try:
        path = kagglehub.dataset_download(dataset_id) # Download dataset from KaggleHub
        file_path = os.path.join(path, "Date_final_dataset_balanced_float32.parquet") 
        df = pd.read_parquet(file_path) 
        logging.info(f"Data loaded successfully from KaggleHub dataset: {dataset_id}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from KaggleHub: {e}")
        raise

def initial_inspection(df):
    """Performs initial inspection of the DataFrame."""
    logging.info("Initial Data Inspection:")
    logging.info(f"First 5 rows:\n{df.head()}")
    logging.info(f"DataFrame Info:\n{df.info(memory_usage='deep')}")
    logging.info(f"Descriptive Statistics:\n{df.describe()}")
    logging.info(f"Missing Values:\n{df.isnull().sum()}")
    logging.info(f"Target Variable 'is_fire' Value Counts:\n{df['is_fire'].value_counts()}")
    logging.info(f"Target Variable 'is_fire' Normalized Value Counts:\n{df['is_fire'].value_counts(normalize=True)}")
    logging.info(f"Data type of 'acq_date' before conversion: {df['acq_date'].dtype}")
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    logging.info(f"Data type of 'acq_date' after conversion: {df['acq_date'].dtype}")

def handle_missing_values(df):
    """Imputes missing values using median for numerical columns."""
    logging.info("Handling Missing Values...")
    for col in df.columns:
        if df[col].isnull().any() and pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logging.info(f"Imputed missing values in column '{col}' with median.")
    logging.info(f"Total missing values remaining: {df.isnull().sum().sum()}")
    return df

def select_features(df, feature_columns):
    """Selects specified feature columns for modeling."""
    df_model = df[feature_columns].copy()
    logging.info(f"DataFrame shape after feature selection: {df_model.shape}")
    logging.info(f"First 5 rows of modeling DataFrame:\n{df_model.head()}")
    return df_model

def time_based_split(df, date_column, train_ratio, val_ratio):
    """Splits data into train, validation, and test sets based on time."""
    df_sorted = df.sort_values(by=date_column)
    n_rows = len(df_sorted)
    train_end = int(n_rows * train_ratio)
    val_end = train_end + int(n_rows * val_ratio)
    df_train = df_sorted[:train_end].copy()
    df_val = df_sorted[train_end:val_end].copy()
    df_test = df_sorted[val_end:].copy()
    logging.info(f"Train set size: {len(df_train)}, Validation set size: {len(df_val)}, Test set size: {len(df_test)}")
    return df_train, df_val, df_test

def prepare_data_for_modeling(df_model, df_train, df_val, df_test):
    """Prepares X and y for train, validation, and test sets from processed DataFrames."""
    X_train = df_model.loc[df_train.index].drop('is_fire', axis=1)
    y_train = df_model.loc[df_train.index]['is_fire']
    X_val = df_model.loc[df_val.index].drop('is_fire', axis=1)
    y_val = df_model.loc[df_val.index]['is_fire']
    X_test = df_model.loc[df_test.index].drop('is_fire', axis=1)
    y_test = df_model.loc[df_test.index]['is_fire']
    logging.info("Data prepared for modeling.")
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_evaluate_random_forest(X_train, y_train, X_val, y_val):
    """Trains and evaluates a Random Forest model."""
    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    y_val_pred_rf = rf_model.predict(X_val)
    y_val_prob_rf = rf_model.predict_proba(X_val)[:, 1]
    logging.info("Random Forest - Validation Report:\n%s", classification_report(y_val, y_val_pred_rf))
    logging.info(f"Random Forest AUC-ROC: {roc_auc_score(y_val, y_val_prob_rf):.4f}")
    feature_importances_rf = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    logging.info(f"Top Feature Importances (Random Forest):\n{feature_importances_rf.head(10)}")
    return rf_model

def train_evaluate_lightgbm(X_train, y_train, X_val, y_val):
    """Trains and evaluates a LightGBM model."""
    lgbm_model = lgb.LGBMClassifier(objective='binary', metric='auc', random_state=42, n_jobs=-1, boosting_type='gbdt', class_weight='balanced')
    lgbm_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=10)])
    y_val_pred_lgbm = lgbm_model.predict(X_val)
    y_val_prob_lgbm = lgbm_model.predict_proba(X_val)[:, 1]
    logging.info("LightGBM - Validation Report:\n%s", classification_report(y_val, y_val_pred_lgbm))
    logging.info(f"LightGBM AUC-ROC: {roc_auc_score(y_val, y_val_prob_lgbm):.4f}")
    feature_importances_lgbm = pd.Series(lgbm_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    logging.info(f"Top Feature Importances (LightGBM):\n{feature_importances_lgbm.head(10)}")
    return lgbm_model, y_val_prob_lgbm

def plot_roc_curve(y_val, y_prob_lgbm):
    """Plots the ROC curve for the LightGBM model on the validation set."""
    fpr, tpr, _ = roc_curve(y_val, y_prob_lgbm)
    auc_score = roc_auc_score(y_val, y_prob_lgbm)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'LightGBM (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Baseline')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Validation Set')
    plt.legend()
    plt.grid(True)
    plt.show()

def predict_synthetic_data(model, expected_columns):
    """Predicts 'is_fire' on synthetic data from CSV using a trained model and hardcoded path."""
    synthetic_csv_path = SYNTHETIC_PATH_CSV  # Using hardcoded path
    try:
        synthetic_df = pd.read_csv(synthetic_csv_path)
        if set(synthetic_df.columns) != set(expected_columns):
            logging.error(f"Synthetic data columns mismatch. Expected: {expected_columns}, Got: {synthetic_df.columns.tolist()}")
            return None
        synthetic_predictions_proba = model.predict_proba(synthetic_df)[:, 1]
        synthetic_predictions_class = model.predict(synthetic_df)
        synthetic_df['predicted_is_fire_proba'] = synthetic_predictions_proba
        synthetic_df['predicted_is_fire_class'] = synthetic_predictions_class
        logging.info("Predictions on synthetic data complete.")
        return synthetic_df
    except FileNotFoundError:
        logging.error(f"Synthetic data file not found at: {synthetic_csv_path}")
        return None
    except Exception as e:
        logging.error(f"Error predicting synthetic data: {e}")
        return None

if __name__ == "__main__":
    config = {
        'feature_columns': [
            'day_of_week', 'day_of_year', 'is_holiday', 'is_weekend',
            'latitude', 'longitude', 'sea_distance',
            'NDVI', 'SoilMoisture',
            'average_temperature_lag_1', 'average_temperature_lag_3', 'average_temperature_lag_7', 'average_temperature_lag_15',
            'precipitation_lag_1', 'precipitation_lag_3', 'precipitation_lag_7', 'precipitation_lag_15',
            'wind_speed_lag_1', 'wind_speed_lag_3', 'wind_speed_lag_7', 'wind_speed_lag_15',
            'dew_point_lag_1', 'dew_point_lag_3', 'dew_point_lag_7', 'dew_point_lag_15',
            'thunder_lag_1', 'thunder_lag_3', 'thunder_lag_7', 'thunder_lag_15',
            'average_temperature_weekly_mean', 'precipitation_weekly_mean',
            'average_temperature_monthly_mean', 'precipitation_monthly_mean',
            'average_temperature_yearly_mean', 'precipitation_yearly_mean',
            'is_fire'
        ],
        'date_column': 'acq_date',
        'train_ratio': 0.7,
        'val_ratio': 0.15,
    }


    try:
        df = load_data()  # Load data from KaggleHub
        initial_inspection(df)
        df = handle_missing_values(df)
        df_model = select_features(df, config['feature_columns'])
        df_train, df_val, df_test = time_based_split(df, config['date_column'], config['train_ratio'], config['val_ratio'])
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_data_for_modeling(df_model, df_train, df_val, df_test)

        rf_model = train_evaluate_random_forest(X_train, y_train, X_val, y_val)
        lgbm_model, y_val_prob_lgbm = train_evaluate_lightgbm(X_train, y_train, X_val, y_val)
        plot_roc_curve(y_val, y_val_prob_lgbm)

        synthetic_df_predicted = predict_synthetic_data(lgbm_model, X_train.columns.tolist()) # Predict synthetic data with hardcoded path
        if synthetic_df_predicted is not None:
            logging.info(f"Predicted synthetic data included in: '{SYNTHETIC_PATH_CSV}'")
            synthetic_df_predicted.to_csv(SYNTHETIC_PATH_CSV, index=False) # Save to the hardcoded synthetic path

        logging.info("Workflow completed successfully.")

    except Exception as e:
        logging.error(f"Critical error during workflow execution: {e}")
        logging.error("Workflow terminated.")