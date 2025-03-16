# Regression on Insurance Premium Prediction

## Overview
This project focuses on building regression models to predict insurance premium amounts based on various customer attributes. The dataset includes features such as age, income, occupation, health score, vehicle age, credit score, and other factors that influence insurance pricing.

## Dataset
- **Train Dataset:** Used to train the model, containing insurance attributes and premium amounts.
- **Test Dataset:** Used for final predictions (without target values).

## Data Preprocessing
- Handled missing values using mean imputation for numerical columns and mode imputation for categorical columns.
- Encoded categorical variables using `LabelEncoder`.
- Standardized numerical features using `StandardScaler`.
- Dropped irrelevant columns such as `id` and `Policy Start Date`.

## Model Implementation
Implemented and evaluated two regression models:
1. **Linear Regression**
2. **SGD Regressor** (Stochastic Gradient Descent)

## Model Evaluation Metrics
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**

## Predictions & Submission
- The trained model predicts the insurance premium amount for the test dataset.
- The results are stored in a CSV file (`submission.csv`).

## Dependencies
- Python 3.x
- Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/insurance-regression.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script in Jupyter Notebook or Google Colab.
4. Generate predictions and save them as `submission.csv`.

## Future Enhancements
- Experiment with more regression models (e.g., Decision Trees, Random Forest, XGBoost).
- Perform feature engineering to improve model performance.
- Hyperparameter tuning for better optimization.

---
Feel free to contribute and improve the project!

