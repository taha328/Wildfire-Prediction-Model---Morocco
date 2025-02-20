# Wildfire Prediction Model - Morocco

This Python script uses machine learning to predict wildfire risk in Morocco.  It trains and evaluates models to understand wildfire patterns and identify important risk factors.

## What the Model Does

*   **Predicts Wildfire Likelihood:**  Estimates the probability of wildfire occurrence based on environmental and temporal data.
*   **Uses Machine Learning:**  Employs LightGBM (Gradient Boosting) and Random Forest models, with LightGBM showing strong performance.
*   **Identifies Key Risk Factors:**  Reveals which factors are most influential in predicting wildfires, such as:
    *   Location (latitude, longitude, sea distance)
    *   Vegetation health (NDVI, Soil Moisture)
    *   Temperature patterns
*   **Provides Performance Insights:** Evaluates model accuracy, AUC-ROC, and feature importance to understand model behavior.
*   **Test Predictions:** Allows for testing the model on new, example data via `synthetic_wildfire_data.csv`.

## Explore the Findings

After running, look at these outputs:

*   **Console Logs:** Examine the validation reports for Random Forest and LightGBM to compare model performance metrics like Accuracy and AUC-ROC.  LightGBM should show better results.
*   **Feature Importances:** The script outputs the top features for both models. Notice the consistent importance of location, vegetation, and temperature factors. This helps understand what drives wildfire risk in the model.
*   **ROC Curve:**  A graph visualizing LightGBM's performance (True Positive Rate vs. False Positive Rate). Higher AUC-ROC score (closer to 1.0) indicates better model discrimination.
*   **Synthetic Data (Optional):** If you provided `synthetic_wildfire_data.csv`, check the updated file to see predicted wildfire probabilities and classifications for your examples.

## Notes

*   **LightGBM is the Stronger Model:**  Validation results show LightGBM outperforming Random Forest for this wildfire prediction task.
*   **Key Factors Identified:** Location, vegetation condition, and temperature emerge as crucial predictors.
*   **Kaggle Data Source:**  Uses a readily available Kaggle dataset for convenience.

This script provides a foundation for wildfire risk analysis in Morocco. Further improvements can be made by using more extensive data, fine-tuning the models, or incorporating additional relevant features.
