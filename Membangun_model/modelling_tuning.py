import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import os

def tune_model():
    # Load Preprocessed Data
    # Path relative to this file in the submission structure
    data_path = 'mobile_price_preprocessing'
    X_train = pd.read_csv(os.path.join(data_path, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv'))

    # Define Parameter Grid
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    # Grid Search
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train.values.ravel())

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Score: {grid_search.best_score_}")

    # Save Tuned Model
    joblib.dump(grid_search.best_estimator_, 'model_tuned.joblib')
    print("Tuned model saved as model_tuned.joblib")

if __name__ == "__main__":
    tune_model()
